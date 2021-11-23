from typing import List, Callable, Dict, Union
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard
from torch import autograd
from tqdm import tqdm
from pprint import pprint

from ..common import GCLBase, Buffer
from ...envs.base import MetaGoalReachingEnv
from ...models import MetaGoalReachAgentDiscrete, MetaGoalReachAgentContinuous
from ..common import utils as U
from ...models.utils import np_dict_to_tensor_dict


class GCML(GCLBase):
    def __init__(
        self,
        experiment_name: str,
        meta_env: MetaGoalReachingEnv,
        meta_learner: Union[MetaGoalReachAgentDiscrete, MetaGoalReachAgentContinuous],
        learner_in_size: int,
        learner_out_size: int,
        learner_n_layers: int,
        learner_hidden: List[int],
        device: torch.device,
        optimizer_name: str,
        outer_lr: float,
        inner_lr: float,
        learn_inner_lr: bool,
        loss_fn: Callable,
        optim_kwargs: dict,
        goal_threshold: float,
        n_shots: int = 1,
        n_tasks: int = 5,
        n_query_traj: int = 1,
        n_outer_steps: int = int(1e6),
        n_inner_steps: int = 1,
        trajectory_len: int = 20,
        n_exploration_trajectory: int = 1,
        log_interval: int = 5,
        val_interval: int = 100,
        n_test_episodes_per_task: int = 10,
        tb_log_dir: str = "tblogs",
        save_model_dir: str = "saved_model",
        save_model_interval: int = 500,
    ):
        self._device = device

        # meta env
        self._meta_env = meta_env

        # create meta parameters
        meta_parameters = {}
        in_size = learner_in_size
        learner_hidden.append(learner_out_size)
        assert learner_n_layers == len(
            learner_hidden
        ), f"Inconsistent n_layers and hidden sizes provided"
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
            torch.empty(learner_out_size, requires_grad=True, device=device,)
        )
        self._meta_parameters = meta_parameters

        self._meta_learner = meta_learner
        self._loss_fn = loss_fn

        optimizer_class = getattr(optim, optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer name provided {optimizer_name}")

        # initialize optimizer
        self._optimizer = optimizer_class(
            params=list(meta_parameters.values()), lr=outer_lr, **optim_kwargs,
        )
        self._inner_lr = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lr, device=self._device)
            for k in self._meta_parameters.keys()
        }

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
            goal_threshold >= 0
        ), f"Invalid value of goal threshold provided {goal_threshold}"
        self._goal_threshold = goal_threshold

        self._log_interval = log_interval
        self._val_interval = val_interval
        self._n_test_episodes_per_task = n_test_episodes_per_task

        # instantiate buffers
        self._inner_loop_buffer = Buffer()
        # this buffer store trajectories before the adaptation for evaluation purpose only
        # because we do not evaluate exploration trajectories, we separate buffers
        self._pre_adapt_eval_buffer = Buffer()

        # tensorboard
        tb_dir = os.path.join(tb_log_dir, experiment_name)
        self._writer = tensorboard.SummaryWriter(log_dir=tb_dir)

        # save model
        save_dir = os.path.join(save_model_dir, experiment_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._save_dir = save_dir
        self._save_model_interval = save_model_interval

    def _inner_loop_adaptation(
        self, obs_dict: Dict[str, torch.Tensor], targets: torch.Tensor, train: bool,
    ):
        parameters = {k: torch.clone(v) for k, v in self._meta_parameters.items()}

        # learner act, we dont use the returned actions to calculate loss
        # we use logits actually
        _ = self._meta_learner.act(
            input_dict=obs_dict, parameters=parameters, greedy=True,
        )  # (N, action_dim)
        logits = self._meta_learner.logits

        # now start `self._n_inner_steps` times adaptation
        for _ in range(self._n_inner_steps):
            # calculate the adaptation loss
            adapt_loss = self._compute_loss(input=logits, target=targets)
            # calculate the adaptation gradients with `autograd.grad`
            adapt_grad = autograd.grad(
                outputs=adapt_loss,
                inputs=list(parameters.values()),
                create_graph=train,
            )
            # now compute the adapted parameters
            parameters = {
                k: initial_param - self._inner_lr[k] * each_grad
                for each_grad, (k, initial_param) in zip(adapt_grad, parameters.items())
            }

        return parameters

    def _outer_step(self, train: bool):
        outer_loss_batch = []
        pre_adapt_distance_batch = []
        post_adapt_distance_batch = []
        query_distance_batch = []

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
                        sample_new_goal=True,
                    )
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)
                # execution trajectory
                # clear evaluation buffer
                self._pre_adapt_eval_buffer.clear_buffer()
                used_goals = []
                for _ in range(self._n_shots):
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=self._meta_parameters,
                        greedy=True,
                        sample_new_goal=True,
                    )
                    used_goals.append(self._meta_env.current_goal)
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)
                    self._pre_adapt_eval_buffer.add_trajectory(generated_trajectory)

                # measure distance before the adaptation
                pre_adapt_distance_batch.append(
                    U.evaluate_batch_trajectories(
                        self._pre_adapt_eval_buffer.all_trajectories,
                        self._goal_threshold,
                        self._meta_env.metric_fn,
                    )
                )

            # generate expert demonstrations with hindsight relabeling
            expert_demo_dict = self._inner_loop_buffer.generate_expert_demo()
            expert_demo_dict = np_dict_to_tensor_dict(
                expert_demo_dict, device=self._device
            )
            # pop expert actions
            targets = expert_demo_dict.pop("action")

            # do inner loop adaptation
            adapted_parameters = self._inner_loop_adaptation(
                obs_dict=expert_demo_dict, targets=targets, train=train,
            )

            # measure post-adaptation distance
            trajectories_to_be_evaluated = []
            for goal in used_goals:
                with torch.no_grad():
                    generated_trajectory = self._sample_trajectory(
                        agent_parameters=adapted_parameters,
                        greedy=True,
                        sample_new_goal=False,
                        predefined_goal=goal,
                    )
                trajectories_to_be_evaluated.append(generated_trajectory)
            post_adapt_distance_batch.append(
                U.evaluate_batch_trajectories(
                    trajectories_to_be_evaluated,
                    self._goal_threshold,
                    self._meta_env.metric_fn,
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
                        sample_new_goal=True,
                    )
                    self._inner_loop_buffer.add_trajectory(generated_trajectory)
            # measure distance for query data
            query_distance_batch.append(
                U.evaluate_batch_trajectories(
                    self._inner_loop_buffer.all_trajectories,
                    self._goal_threshold,
                    self._meta_env.metric_fn,
                )
            )

            # generate expert demonstrations with hindsight relabeling
            expert_demo_dict = self._inner_loop_buffer.generate_expert_demo()
            expert_demo_dict = np_dict_to_tensor_dict(
                expert_demo_dict, device=self._device
            )
            # pop expert actions
            targets = expert_demo_dict.pop("action")

            # get logits on the query data with adapted parameters
            _ = self._meta_learner.act(
                input_dict=expert_demo_dict, parameters=adapted_parameters, greedy=True,
            )
            logits = self._meta_learner.logits
            # calculate loss on query data
            query_loss = self._compute_loss(input=logits, target=targets)
            outer_loss_batch.append(query_loss)

        # finish the iteration over task
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        pre_adapt_distance = np.mean(pre_adapt_distance_batch)
        post_adapt_distance = np.mean(post_adapt_distance_batch)
        query_distance = np.mean(query_distance_batch)
        return (
            outer_loss,
            pre_adapt_distance,
            post_adapt_distance,
            query_distance,
        )

    def _take_gradient_step(self, loss: torch.Tensor):
        loss.backward()
        self._optimizer.step()

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        return self._loss_fn(input=input, target=target)

    def _sample_trajectory(
        self, agent_parameters, greedy: bool, sample_new_goal, predefined_goal=None,
    ):
        return U.sample_trajectory(
            agent=self._meta_learner,
            agent_parameters=agent_parameters,
            env=self._meta_env,
            episode_length=self._trajectory_len,
            greedy=greedy,
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
                pre_adapt_distance,
                post_adapt_distance,
                query_distance,
            ) = self._outer_step(train=True)
            self._take_gradient_step(outer_loss)

            if step_idx % self._log_interval == 0:
                printed_info = {
                    "Iteration:": step_idx,
                    "Loss:": outer_loss.item(),
                    "Pre-adaptation distance:": pre_adapt_distance,
                    "Post-adaptation distance:": post_adapt_distance,
                    "Post-adaptation query distance": query_distance,
                }
                pprint(printed_info)

                # write tensorboard log
                self._writer.add_scalar(
                    "loss/train", outer_loss.item(), step_idx,
                )
                self._writer.add_scalar(
                    "train_distance/pre_adapt_support", pre_adapt_distance, step_idx,
                )
                self._writer.add_scalar(
                    "train_distance/post_adapt_support", post_adapt_distance, step_idx,
                )
                self._writer.add_scalar(
                    "train_distance/post_adapt_query", query_distance, step_idx,
                )

            if step_idx % self._val_interval == 0:
                (
                    outer_loss,
                    pre_adapt_distance,
                    post_adapt_distance,
                    query_distance,
                ) = self._outer_step(train=False)
                printed_info = {
                    "Val_Loss:": outer_loss.item(),
                    "Val_Pre-adaptation distance:": pre_adapt_distance,
                    "Val_Post-adaptation distance:": post_adapt_distance,
                    "Val_Post-adaptation query distance": query_distance,
                }
                pprint(printed_info)

                # write tensorboard log
                self._writer.add_scalar(
                    "loss/val", outer_loss.item(), step_idx,
                )
                self._writer.add_scalar(
                    "val_distance/pre_adapt_support", pre_adapt_distance, step_idx,
                )
                self._writer.add_scalar(
                    "val_distance/post_adapt_support", post_adapt_distance, step_idx,
                )
                self._writer.add_scalar(
                    "val_distance/post_adapt_query", query_distance, step_idx,
                )

            if step_idx % self._save_model_interval == 0:
                self._save(step_idx)

    def _save(self, checkpoint_step):
        optimizer_state_dict = self._optimizer.state_dict()
        save_file_name = os.path.join(self._save_dir, f"{checkpoint_step}.pt")
        torch.save(
            dict(
                meta_parameters=self._meta_parameters,
                inner_lrs=self._inner_lr,
                optimizer_state_dict=optimizer_state_dict,
            ),
            save_file_name,
        )

    def test(self, *args, **kwargs):
        pass
