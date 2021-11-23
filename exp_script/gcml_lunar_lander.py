import argparse

import torch.nn.functional as F

from gcml.algorithms import GCML
from gcml.envs import LunarAnyLander
import gcml.models.utils as U
from gcml.models import MetaGoalReachAgentDiscrete


def main(arg):
    # goal threshold doesn't matter in GCML because we don't care about rewards
    meta_env = LunarAnyLander(goal_threshold=0)
    meta_learner = MetaGoalReachAgentDiscrete(
        n_layers=3, activation=F.relu, device=U.get_device(),
    )

    if arg.exp_name == "auto":
        exp_name = f"lander_outer_lr:{arg.outer_lr}_inner_lr:{arg.inner_lr}_n_tasks:{arg.n_tasks}_traj_len:{arg.traj_len}"
    else:
        exp_name = arg.exp_name

    trainer = GCML(
        experiment_name=exp_name,
        meta_env=meta_env,
        meta_learner=meta_learner,
        learner_in_size=2 * 4 + 8,
        learner_out_size=meta_env.action_space.n,
        learner_n_layers=3,
        learner_hidden=[64, 64],
        device=U.get_device(),
        optimizer_name="Adam",
        outer_lr=arg.outer_lr,
        inner_lr=arg.inner_lr,
        loss_fn=F.cross_entropy,
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
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--traj_len", type=int, default=50)
    main_args = parser.parse_args()
    main(main_args)
