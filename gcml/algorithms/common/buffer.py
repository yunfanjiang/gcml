from typing import Dict

import numpy as np

from .hindsight_relabel import hindsight_relabel
from .constants import Trajectory


class Buffer(object):
    def __init__(self):
        """
        Initialize a buffer. A buffer itself can be a list of trajectories, a deque, and so on.
        """
        raise NotImplementedError

    def add_trajectory(self, trajectory: Trajectory):
        """
        Add the generated trajectories in to the buffer.
        """
        raise NotImplementedError

    def generate_expert_demo(self) -> Dict[str, np.ndarray]:
        """
        Call `hindsight_relabel()` to generate expert transitions using all added trajectories.
        The returned dict should contain keys
        `observation`, `achieved_goal`, `achieved_state_goal`, `desired_goal`, `desired_state_goal`, and `task_config`.
        Each value should be a numpy array with shape (N, *)
        """
        raise NotImplementedError
