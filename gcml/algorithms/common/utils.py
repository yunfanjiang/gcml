from typing import Dict

import torch
import numpy as np

from ...models import MetaGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory


def sample_trajectory(
    agent: MetaGoalReachAgent,
    agent_parameters: Dict[str, torch.Tensor],
    env: MetaGoalReachingEnv,
    greedy: bool,
    exploration_coeff: float,
    sample_new_goal: bool,
    predefined_goal=None,
) -> Trajectory:
    raise NotImplementedError


def evaluate_batch_trajectories(
    batch_trajectories, goal_threshold,
):
    raise NotImplementedError


def np_dict_to_tensor_dict(np_dict: dict, device):
    tensor_dict = {}
    for k, v in np_dict.items():
        if isinstance(v, dict):
            sub_tensor_dict = np_dict_to_tensor_dict(v, device)
            tensor_dict[k] = sub_tensor_dict
        elif isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        else:
            raise ValueError(f"Unknown type {type(v)}")
    return tensor_dict
