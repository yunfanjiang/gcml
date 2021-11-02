from typing import Dict

import numpy as np

from .hindsight_relabel import hindsight_relabel
from .constants import Trajectory, Transition


class Buffer(object):
    def __init__(self):
        """
        Initialize a buffer. A buffer itself can be a list of trajectories, a deque, and so on.
        """
        self._internal_buffer = []

    def add_trajectory(self, trajectory: Trajectory):
        """
        Add the generated trajectories in to the buffer.
        """
        self._internal_buffer.append(trajectory)

    def generate_expert_demo(self) -> Dict[str, np.ndarray]:
        """
        Call `hindsight_relabel()` to generate expert transitions using all added trajectories.
        The returned dict should contain keys
        `action`, `observation`, `achieved_goal`, `achieved_state_goal`,
        `desired_goal`, `desired_state_goal`, and `task_config`.
        Each value should be a numpy array with shape (N, *)
        """
        batched_demos = {key: [] for key in Transition._fields}
        for each_trajectory in self._internal_buffer:
            for expert_transition in hindsight_relabel(each_trajectory):
                for key in batched_demos.keys():
                    batched_demos[key].append(getattr(expert_transition, key))
        batched_demos = {k: np.stack(v, axis=0) for k, v in batched_demos.items()}
        return batched_demos

    def clear_buffer(self):
        """
        clear all added trajectories.
        """
        self._internal_buffer = []

    @property
    def all_trajectories(self):
        return self._internal_buffer
