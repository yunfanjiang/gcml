from typing import NamedTuple, List

import numpy as np


class Transition(NamedTuple):
    cur_obs: np.ndarray
    cur_achieved_goal: np.ndarray
    cur_achieved_s_goal: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    next_achieved_goal: np.ndarray
    next_achieved_s_goal: np.ndarray
    task_config: np.ndarray
    goal: np.ndarray
    s_goal: np.ndarray


class RLTransition(Transition):
    reward: float
    done: bool


Trajectory = List[Transition]
RLTrajectory = List[RLTransition]
