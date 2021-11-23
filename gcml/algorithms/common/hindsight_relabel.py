from typing import Iterable
from copy import deepcopy

from .constants import Trajectory, Transition


def hindsight_relabel(trajectory: Trajectory,) -> Iterable[Transition]:
    trajectory_len = len(trajectory)
    for ptr_start in range(trajectory_len - 1):
        ptr_end = trajectory_len - 1
        start_transition = trajectory[ptr_start]
        end_transition = trajectory[ptr_end]
        new_transition = Transition(
            observation=start_transition.observation.copy(),
            achieved_goal=start_transition.achieved_goal.copy(),
            achieved_state_goal=start_transition.achieved_state_goal.copy(),
            action=deepcopy(start_transition.action),
            next_obs=start_transition.next_obs.copy(),
            next_achieved_goal=start_transition.next_achieved_goal.copy(),
            next_achieved_s_goal=start_transition.next_achieved_s_goal.copy(),
            task_config=start_transition.task_config.copy(),
            desired_goal=end_transition.next_achieved_goal.copy(),
            desired_state_goal=end_transition.next_achieved_s_goal.copy(),
        )
        yield new_transition
