from ...models import MetaGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory


def sample_trajectory(
        agent: MetaGoalReachAgent,
        env: MetaGoalReachingEnv,
) -> Trajectory:
    raise NotImplementedError


def evaluate_batch_trajectories(
        batch_trajectories,
        batch_goals,
        goal_threshold,
):
    raise NotImplementedError
