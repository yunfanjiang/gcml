import argparse

import torch.nn.functional as F

from gcml.algorithms import GCML
from gcml.envs import MetaPendulumEnv
import gcml.models.utils as U
import gcml.models.loss as L
from gcml.models import MetaGoalReachAgentContinuous


def main(arg):
    # goal threshold doesn't matter in GCML because we don't care about rewards
    meta_env = MetaPendulumEnv(goal_threshold=0)
    meta_learner = MetaGoalReachAgentContinuous(
        n_layers=3,
        activation=F.relu,
        device=U.get_device(),
        std=0.1,
        use_task_config=arg.use_task_config,
    )

    if arg.exp_name == "auto":
        exp_name = (
            f"pendulum_outer_lr:{arg.outer_lr}_inner_lr:{arg.inner_lr}_learn_inner_lr:{arg.learn_inner_lr}_"
            f"use_task_config:{arg.use_task_config}_n_tasks:{arg.n_tasks}_traj_len:{arg.traj_len}"
        )
        exp_name = exp_name + f"_run{arg.n_run}"
    else:
        exp_name = arg.exp_name

    learner_in_size = 12 if arg.use_task_config else 9
    trainer = GCML(
        experiment_name=exp_name,
        meta_env=meta_env,
        meta_learner=meta_learner,
        learner_in_size=learner_in_size,
        learner_out_size=meta_env.action_space.shape[0],
        learner_n_layers=3,
        learner_hidden=[64, 64],
        device=U.get_device(),
        optimizer_name="Adam",
        outer_lr=arg.outer_lr,
        inner_lr=arg.inner_lr,
        learn_inner_lr=arg.learn_inner_lr,
        loss_fn=L.mse_loss,
        optim_kwargs={},
        goal_threshold=0,  # doesn't matter
        n_tasks=arg.n_tasks,
        trajectory_len=arg.traj_len,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="auto")
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--inner_lr", type=float, default=0.4)
    parser.add_argument("--learn_inner_lr", default=True, action="store_false")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--traj_len", type=int, default=50)
    parser.add_argument("--use_task_config", default=False, action="store_true")
    parser.add_argument("--n_run", default=0)
    main_args = parser.parse_args()
    main(main_args)
