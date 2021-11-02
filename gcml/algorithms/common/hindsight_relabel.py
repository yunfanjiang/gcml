from typing import Iterable

from .constants import Trajectory, Transition


def hindsight_relabel(trajectory: Trajectory,) -> Iterable[Transition]:
    trajectory_len = len(trajectory)
    for ptr_start in range(trajectory_len - 1):
        for ptr_end in range(ptr_start + 1, trajectory_len):
            start_transition = trajectory[ptr_start]
            end_transition = trajectory[ptr_end]
            new_transition = Transition(
                cur_obs=start_transition.cur_obs.copy(),
                cur_achieved_goal=start_transition.cur_achieved_goal.copy(),
                cur_achieved_s_goal=start_transition.cur_achieved_s_goal.copy(),
                action=start_transition.action.copy(),
                next_obs=start_transition.next_obs.copy(),
                next_achieved_goal=start_transition.next_achieved_goal.copy(),
                next_achieved_s_goal=start_transition.next_achieved_s_goal.copy(),
                task_config=start_transition.task_config.copy(),
                goal=end_transition.next_achieved_goal.copy(),
                s_goal=end_transition.next_achieved_s_goal.copy(),
            )
            yield new_transition
