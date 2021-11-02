from typing import Optional, List, Callable, Dict
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import tensorboard
from torch import autograd
from tqdm import tqdm
from pprint import pprint

from ..common import GCLBase, Buffer
from ...envs.base import MetaGoalReachingEnv
from ...models import MetaGoalReachAgent
from ..common import utils as U


class GCML(GCLBase):
    def __init__(
        self,
        experiment_name: str,
        meta_env: MetaGoalReachingEnv,
        learner_in_size: int,
        learner_out_size: int,
        learner_n_layers: int,
        learner_activation: str,
        learner_hidden: List[int],
        device: torch.device,
        optimizer_name: str,
        outer_lr: float,
        inner_lr: float,
        loss_fn: Callable,
        optim_kwargs: dict,
        n_shots: int = 1,
        n_tasks: int = 5,
        n_query_traj: int = 1,
        n_outer_steps: int = 1e6,
        n_inner_steps: int = 1,
        trajectory_len: int = 20,
        n_exploration_trajectory: int = 1,
        boltzmann_exploration_coeff: float = 0.0,
        goal_threshold: float = 0.05,
        log_interval: int = 5,
        val_interval: int = 100,
        n_test_episodes_per_task: int = 10,
        tb_log_dir: str = "tblogs",
    ):
        self._device = device

        # meta env
        self._meta_env = meta_env

        # create meta parameters
        meta_parameters = {}
        in_size = learner_in_size
        for i in range(learner_n_layers - 1):
            meta_parameters[f"w{i}"] = nn.init.xavier_normal_(
                torch.empty(
                    learner_hidden[i], in_size, requires_grad=True, device=device,
                )
            )
            meta_parameters[f"b{i}"] = nn.init.zeros_(
                torch.empty(learner_hidden[i], requires_grad=True, device=device,)
            )
            in_size = learner_hidden[i]
        meta_parameters[f"w{learner_n_layers - 1}"] = nn.init.xavier_normal_(
            torch.empty(learner_out_size, in_size, requires_grad=True, device=device,)
        )
        meta_parameters[f"b{learner_n_layers - 1}"] = nn.init.zeros_(
            torch.empty(learner_out_size, in_size, requires_grad=True, device=device,)
        )
        self._meta_parameters = meta_parameters

        activation_fn = getattr(F, learner_activation)
        if activation_fn is None:
            raise ValueError(f"Unknown activation provided {learner_activation}")
        self._meta_learner = MetaGoalReachAgent(
            n_layers=learner_n_layers, activation=activation_fn
        )
        self._loss_fn = loss_fn

        optimizer_class = getattr(optim, optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer name provided {optimizer_name}")

        # initialize optimizer
        self._optimizer = optimizer_class(
            params=meta_parameters, lr=outer_lr, **optim_kwargs,
        )
        self._inner_lr = inner_lr

        # task relevant parameters
        assert n_shots >= 1, f"Invalid number of shots provided {n_shots}"
        self._n_shots = n_shots
        assert n_tasks >= 1, f"Invalid number of tasks provided {n_tasks}"
        self._n_tasks = n_tasks
        assert (
            n_query_traj >= 1
        ), f"Invalid number of query trajectories provided {n_query_traj}"
        self._n_query_traj = n_query_traj
        assert (
            n_outer_steps >= 1
        ), f"Invalid number of outer steps provided {n_outer_steps}"
        self._n_outer_steps = n_outer_steps
        assert (
            n_inner_steps >= 1
        ), f"Invalid number of inner steps provided {n_inner_steps}"
        self._n_inner_steps = n_inner_steps
        assert (
            trajectory_len >= 1
        ), f"Invalid trajectory length provided {trajectory_len}"
        self._trajectory_len = trajectory_len
        assert (
            n_exploration_trajectory > 0
        ), f"Invalid number of exploration trajectories provided {n_exploration_trajectory}"
        self._n_exp_trajectory = n_exploration_trajectory
        assert (
            boltzmann_exploration_coeff > 0
        ), f"Invalid exploration coeff provided {boltzmann_exploration_coeff}"
        self._exp_coeff = boltzmann_exploration_coeff
        assert (
            goal_threshold >= 0
        ), f"Invalid value of goal threshold provided {goal_threshold}"
        self._goal_threshold = goal_threshold

        self._log_interval = log_interval
        self._val_interval = val_interval
        self._n_test_episodes_per_task = n_test_episodes_per_task

        # instantiate buffers
        self._inner_loop_buffer = Buffer()

        # tensorboard
        tb_dir = os.path.join(tb_log_dir, experiment_name)
        self._writer = tensorboard.SummaryWriter(log_dir=tb_dir)

    def _inner_loop_adaptation(
        self, obs_dict: Dict[str, torch.Tensor], targets: torch.Tensor, train: bool,
    ):
        parameters = {k: torch.clone(v) for k, v in self._meta_parameters.items()}

        # forward the learner to get actions
        actions = self._meta_learner.act(
            input_dict=obs_dict,
            parameters=parameters,
            noise_coeff=self._exp_coeff,
            greedy=True,
        )  # (N, action_dim)

        # now start `self._n_inner_steps` times adaptation
        for _ in range(self._n_inner_steps):
            # calculate the adaptation loss
            adapt_loss = self._compute_loss(predictions=actions, targets=targets)
            # calculate the adaptation gradients with `autograd.grad`
            adapt_grad = autograd.grad(
                outputs=adapt_loss,
                inputs=list(parameters.values()),
                create_graph=train,
            )
            # now compute the adapted parameters
            parameters = {
                k: initial_param - self._inner_lr * each_grad
                for each_grad, (k, initial_param) in zip(adapt_grad, parameters.items())
            }

        return parameters

    def _outer_step(self, train: bool):
        outer_loss_batch = []
        pre_adapt_success_rate_batch = []
        post_adapt_success_rate_batch = []
        query_success_rate_batch = []

        for task_i in range(self._n_tasks):
            # sample a new task
            self._meta_env.sample_task()

            # clear inner loop buffer
            self._inner_loop_buffer.clear_buffer()

            # collect data
            # exploration trajectory
            with torch.no_grad():
                for _ in range(self._n_exp_trajectory):
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=self._meta_parameters,
                        greedy=False,
                        exp_coeff=1,
                        sample_new_goal=True,
                    )
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)
                # execution trajectory
                used_goals = []
                for _ in range(self._n_shots):
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=self._meta_parameters,
                        greedy=True,
                        exp_coeff=self._exp_coeff,
                        sample_new_goal=True,
                    )
                    used_goals.append(self._meta_env.current_goal)
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)

                # measure success rate before the adaptation
                pre_adapt_success_rate_batch.append(
                    U.evaluate_batch_trajectories(
                        self._inner_loop_buffer.all_trajectories, self._goal_threshold,
                    )
                )

            # generate expert demonstrations with hindsight relabeling
            expert_demo_dict = self._inner_loop_buffer.generate_expert_demo()
            expert_demo_dict = U.np_dict_to_tensor_dict(
                expert_demo_dict, device=self._device
            )
            # pop expert actions
            targets = expert_demo_dict.pop("action")

            # do inner loop adaptation
            adapted_parameters = self._inner_loop_adaptation(
                obs_dict=expert_demo_dict, targets=targets, train=train,
            )

            # measure post-adaptation success rate
            trajectories_to_be_evaluated = []
            for goal in used_goals:
                with torch.no_grad():
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=adapted_parameters,
                        greedy=True,
                        exp_coeff=self._exp_coeff,
                        sample_new_goal=False,
                        predefined_goal=goal,
                    )
                trajectories_to_be_evaluated.append(generated_trajectory)
            post_adapt_success_rate_batch.append(
                U.evaluate_batch_trajectories(
                    trajectories_to_be_evaluated, self._goal_threshold,
                )
            )

            # clear inner loop buffer
            self._inner_loop_buffer.clear_buffer()

            # generate query data with adapted parameters
            with torch.no_grad():
                for _ in range(self._n_query_traj):
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=adapted_parameters,
                        greedy=True,
                        exp_coeff=self._exp_coeff,
                        sample_new_goal=True,
                    )
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)
            # measure success rate for query data
            query_success_rate_batch.append(
                U.evaluate_batch_trajectories(
                    self._inner_loop_buffer.all_trajectories, self._goal_threshold,
                )
            )

            # generate expert demonstrations with hindsight relabeling
            expert_demo_dict = self._inner_loop_buffer.generate_expert_demo()
            expert_demo_dict = U.np_dict_to_tensor_dict(
                expert_demo_dict, device=self._device
            )
            # pop expert actions
            targets = expert_demo_dict.pop("action")

            # get predicted actions on the query data with adapted parameters
            predicted_actions = self._meta_learner.act(
                input_dict=expert_demo_dict,
                parameters=adapted_parameters,
                noise_coeff=self._exp_coeff,
                greedy=True,
            )
            # calculate loss on query data
            query_loss = self._compute_loss(
                predictions=predicted_actions, targets=targets
            )
            outer_loss_batch.append(query_loss)

        # finish the iteration over task
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        pre_adapt_success_rate = np.mean(pre_adapt_success_rate_batch)
        post_adapt_success_rate = np.mean(post_adapt_success_rate_batch)
        query_success_rate = np.mean(query_success_rate_batch)
        return (
            outer_loss,
            pre_adapt_success_rate,
            post_adapt_success_rate,
            query_success_rate,
        )

    def _take_gradient_step(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        return self._loss_fn(predicts=predictions, targets=targets)

    def _sample_trajectory(
        self,
        agent_parameters,
        greedy: bool,
        exp_coeff: float,
        sample_new_goal,
        predefined_goal=None,
    ):
        return U.sample_trajectory(
            agent=self._meta_learner,
            agent_parameters=agent_parameters,
            env=self._meta_env,
            greedy=greedy,
            exploration_coeff=exp_coeff,
            sample_new_goal=sample_new_goal,
            predefined_goal=predefined_goal,
        )

    def train(self):
        """
        Train GCML.
        """
        for step_idx in tqdm(range(self._n_outer_steps)):
            self._optimizer.zero_grad()
            (
                outer_loss,
                pre_adapt_success_rate,
                post_adapt_success_rate,
                query_success_rate,
            ) = self._outer_step(train=True)
            self._take_gradient_step(outer_loss)

            if step_idx % self._log_interval == 0:
                printed_info = {
                    "Iteration:": step_idx,
                    "Loss:": outer_loss.item(),
                    "Pre-adaptation success rate:": pre_adapt_success_rate,
                    "Post-adaptation success rate:": post_adapt_success_rate,
                    "Post-adaptation query success rate": query_success_rate,
                }
                pprint(printed_info)

                # write tensorboard log
                self._writer.add_scalar(
                    "loss/train", outer_loss.item(), step_idx,
                )
                self._writer.add_scalar(
                    "train_success_rate/pre_adapt_support",
                    pre_adapt_success_rate,
                    step_idx,
                )
                self._writer.add_scalar(
                    "train_success_rate/post_adapt_support",
                    post_adapt_success_rate,
                    step_idx,
                )
                self._writer.add_scalar(
                    "train_success_rate/post_adapt_query", query_success_rate, step_idx,
                )

            if step_idx % self._val_interval == 0:
                (
                    outer_loss,
                    pre_adapt_success_rate,
                    post_adapt_success_rate,
                    query_success_rate,
                ) = self._outer_step(train=False)
                printed_info = {
                    "Val_Loss:": outer_loss.item(),
                    "Val_Pre-adaptation success rate:": pre_adapt_success_rate,
                    "Val_Post-adaptation success rate:": post_adapt_success_rate,
                    "Val_Post-adaptation query success rate": query_success_rate,
                }
                pprint(printed_info)

                # write tensorboard log
                self._writer.add_scalar(
                    "loss/val", outer_loss.item(), step_idx,
                )
                self._writer.add_scalar(
                    "val_success_rate/pre_adapt_support",
                    pre_adapt_success_rate,
                    step_idx,
                )
                self._writer.add_scalar(
                    "val_success_rate/post_adapt_support",
                    post_adapt_success_rate,
                    step_idx,
                )
                self._writer.add_scalar(
                    "val_success_rate/post_adapt_query", query_success_rate, step_idx,
                )

    def test(self, *args, **kwargs):
        pass
