from typing import Dict, List, Callable

import torch
import numpy as np

from ...models import MetaGoalReachAgentBase
from ...envs import MetaGoalReachingEnv
from .constants import Trajectory, Transition


def sample_trajectory(
    agent: MetaGoalReachAgentBase,
    agent_parameters: Dict[str, torch.Tensor],
    env: MetaGoalReachingEnv,
    episode_length: int,
    greedy: bool,
    sample_new_goal: bool,
    predefined_goal=None,
) -> Trajectory:
    trajectory = []
    obs = env.reset(sample_new_goal=sample_new_goal, predefined_goal=predefined_goal)
    for _ in range(episode_length):
        action = agent.act(input_dict=obs, parameters=agent_parameters, greedy=greedy,)
        # convert tensor action to float
        action = action.detach().cpu().numpy().item()
        next_obs, _, _, _ = env.step(action)
        transition = Transition(
            observation=obs["observation"],
            achieved_goal=obs["achieved_goal"],
            achieved_state_goal=obs["achieved_state_goal"],
            action=action,
            next_obs=next_obs["observation"],
            next_achieved_goal=next_obs["achieved_goal"],
            next_achieved_s_goal=next_obs["achieved_state_goal"],
            task_config=obs["task_config"],
            desired_goal=obs["desired_goal"],
            desired_state_goal=obs["desired_state_goal"],
        )
        trajectory.append(transition)
        obs = next_obs
    return trajectory


def evaluate_batch_trajectories(
    batch_trajectories: List[Trajectory], goal_threshold: float, metric_fn: Callable,
):
    achieved_state_goal = [
        trajectory[-1].next_achieved_s_goal for trajectory in batch_trajectories
    ]
    desired_state_goal = [
        trajectory[-1].achieved_state_goal for trajectory in batch_trajectories
    ]
    achieved_state_goal = np.stack(achieved_state_goal, axis=0)
    desired_state_goal = np.stack(desired_state_goal, axis=0)
    distance = metric_fn(achieved_state_goal, desired_state_goal)
    is_successful = distance <= goal_threshold
    return np.mean(is_successful)
