from ...models import BaseGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory


def sample_trajectory(
        agent: BaseGoalReachAgent,
        env: MetaGoalReachingEnv,
) -> Trajectory:
    raise NotImplementedError
