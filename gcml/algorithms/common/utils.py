from typing import Dict, List, Callable

import torch
import numpy as np

from ...models import MetaGoalReachAgent
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory, Transition


def sample_trajectory(
    agent: MetaGoalReachAgent,
    agent_parameters: Dict[str, torch.Tensor],
    env: MetaGoalReachingEnv,
    episode_length: int,
    greedy: bool,
    exploration_coeff: float,
    sample_new_goal: bool,
    predefined_goal=None,
) -> Trajectory:
    trajectory = []
    obs = env.reset(sample_new_goal=sample_new_goal, predefined_goal=predefined_goal)
    for _ in range(episode_length):
        action = agent.act(
            input_dict=obs,
            parameters=agent_parameters,
            noise_coeff=exploration_coeff,
            greedy=greedy,
        )
        next_obs, _, _, _ = env.step(action)
        transition = Transition(
            cur_obs=obs["observation"],
            cur_achieved_goal=obs["achieved_goal"],
            cur_achieved_s_goal=obs["achieved_state_goal"],
            action=action,
            next_obs=next_obs["observation"],
            next_achieved_goal=next_obs["achieved_goal"],
            next_achieved_s_goal=next_obs["achieved_state_goal"],
            task_config=obs["task_config"],
            goal=obs["goal"],
            s_goal=obs["state_goal"],
        )
        trajectory.append(transition)
        obs = next_obs
    return trajectory


def evaluate_batch_trajectories(
    batch_trajectories: List[Trajectory], goal_threshold: float, metric_fn: Callable,
):
    achieved_state_goal = [trajectory[-1].next_achieved_s_goal for trajectory in batch_trajectories]
    desired_state_goal = [trajectory[-1].s_goal for trajectory in batch_trajectories]
    achieved_state_goal = np.stack(achieved_state_goal, axis=0)
    desired_state_goal = np.stack(desired_state_goal, axis=0)
    distance = metric_fn(achieved_state_goal, desired_state_goal)
    is_successful = distance <= goal_threshold
    return np.mean(is_successful.float())


def np_dict_to_tensor_dict(np_dict: dict, device):
    tensor_dict = {}
    for k, v in np_dict.items():
        if isinstance(v, dict):
            sub_tensor_dict = np_dict_to_tensor_dict(v, device)
            tensor_dict[k] = sub_tensor_dict
        elif isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            tensor_dict[k] = v.to(device)
        else:
            raise ValueError(f"Unknown type {type(v)}")
    return tensor_dict
