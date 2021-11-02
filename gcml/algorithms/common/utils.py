from ...models import MetaGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory


def sample_trajectory(
        agent: MetaGoalReachAgent,
        env: MetaGoalReachingEnv,
        greedy: bool,
        exploration_coeff: float,
) -> Trajectory:
    raise NotImplementedError


def evaluate_batch_trajectories(
        batch_trajectories,
        goal_threshold,
):
    raise NotImplementedError
