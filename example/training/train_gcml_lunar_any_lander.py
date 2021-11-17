import torch.nn.functional as F

from gcml.algorithms import GCML
from gcml.envs import LunarAnyLander
import gcml.models.utils as U
from gcml.models import MetaGoalReachAgentDiscrete


if __name__ == "__main__":
    goal_threshold = 0.05
    meta_env = LunarAnyLander(goal_threshold)
    meta_learner = MetaGoalReachAgentDiscrete(
        n_layers=3, activation=F.relu, device=U.get_device(),
    )

    trainer = GCML(
        experiment_name="gcml_lunar_lander_example",
        meta_env=meta_env,
        meta_learner=meta_learner,
        learner_in_size=2 * 4 + 8,
        learner_out_size=meta_env.action_space.n,
        learner_n_layers=3,
        learner_hidden=[64, 64],
        device=U.get_device(),
        optimizer_name="Adam",
        outer_lr=1e-3,
        inner_lr=1e-3,
        loss_fn=F.cross_entropy,
        optim_kwargs={},
        goal_threshold=goal_threshold,
    )

    trainer.train()
