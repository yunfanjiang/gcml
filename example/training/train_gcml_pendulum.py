import torch.nn.functional as F

from gcml.algorithms import GCML
from gcml.envs import MetaPendulumEnv
import gcml.models.utils as U
import gcml.models.loss as L
from gcml.models import MetaGoalReachAgentContinuous


if __name__ == "__main__":
    goal_threshold = 0.05
    meta_env = MetaPendulumEnv(goal_threshold)
    meta_learner = MetaGoalReachAgentContinuous(
        n_layers=3, activation=F.relu, device=U.get_device(), std=0.1
    )

    trainer = GCML(
        experiment_name="gcml_example",
        meta_env=meta_env,
        meta_learner=meta_learner,
        learner_in_size=9,
        learner_out_size=meta_env.action_space.shape[0],
        learner_n_layers=3,
        learner_hidden=[64, 64],
        device=U.get_device(),
        optimizer_name="Adam",
        outer_lr=1e-3,
        inner_lr=1e-3,
        loss_fn=L.mse_loss,
        optim_kwargs={},
        goal_threshold=goal_threshold,
    )

    trainer.train()