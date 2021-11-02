from ...models import MetaGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory


def sample_trajectory(
        agent: MetaGoalReachAgent,
        env: MetaGoalReachingEnv,
) -> Trajectory:
    raise NotImplementedError
