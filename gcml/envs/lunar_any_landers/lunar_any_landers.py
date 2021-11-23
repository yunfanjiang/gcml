from typing import Dict, Any

import numpy as np
import gym

from gcml.envs.lunar_any_landers.lunar_lander_base import (
    LunarLander,
    VIEWPORT_H,
    SCALE,
    CHUNKS,
)
from gcml.envs.base import MetaGoalReachingEnv
from gcml.envs.lunar_any_landers.metric import metric_fn


class LunarAnyLander(MetaGoalReachingEnv):
    def __init__(self, goal_threshold):
        # create base env
        base_env = LunarLander(goal_threshold)

        # prepare task config space
        task_config_space = gym.spaces.Dict(
            {
                "height": gym.spaces.Box(
                    low=0, high=VIEWPORT_H / SCALE / 4, shape=(CHUNKS + 1,)
                ),
                "main_engine_power": gym.spaces.Box(low=13, high=20, shape=(1,)),
                "side_engine_power": gym.spaces.Box(low=0.6, high=1, shape=(1,)),
                "leg_height": gym.spaces.Box(low=8, high=12, shape=(1,)),
                "leg_spring_torque": gym.spaces.Box(low=40, high=60, shape=(1,)),
                "lander_density": gym.spaces.Box(low=2, high=5, shape=(1,)),
            }
        )
        self._task_config_space = task_config_space

        super(LunarAnyLander, self).__init__(
            base_env=base_env,
            metric_fn=metric_fn,
            task_config_space=task_config_space,
            obs_key="base_obs",
            achieved_goal_key="achieved_goal",
            achieved_state_goal_key="achieved_state_goal",
        )

    def _sample_task(self):
        sampled_task_config = {
            key: np.random.uniform(
                low=self._task_config_space[key].low,
                high=self._task_config_space[key].high,
                size=self._task_config_space[key].shape,
            )
            for key in self._task_config_space
        }
        return sampled_task_config

    def reset(
        self, sample_new_goal: bool = True, predefined_goal=None
    ) -> Dict[str, np.ndarray]:
        if sample_new_goal:
            self._base_env.sample_goal(height=self._task_config["height"])
        elif predefined_goal is not None:
            self._base_env.set_goal(predefined_goal)
        else:
            raise ValueError(f"Either sample a new goal or provide a predefined goal")
        base_obs = self._base_env.reset(
            height=self._task_config["height"],
            main_engine_power=self._task_config["main_engine_power"].item(),
            side_engine_power=self._task_config["side_engine_power"].item(),
            leg_height=self._task_config["leg_height"].item(),
            leg_spring_torque=self._task_config["leg_spring_torque"].item(),
            lander_density=self._task_config["lander_density"].item(),
        )
        obs = self._prepare_obs(base_obs)
        return obs

    def step(self, action: np.ndarray):
        action = self._preprocess_action(action)
        # we omit reward and done here because they don't matter in GCML/GCSL
        next_base_obs, _, _, info = self._base_env.step(action)
        next_obs = self._prepare_obs(next_base_obs)
        info.update(
            {
                "metric": self._metric_fn(
                    next_obs["achieved_state_goal"], self._base_env.goal,
                )
            }
        )
        return next_obs, 0, False, info

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        assert self.action_space.contains(action)
        return action

    def _base_obs_to_obs(
        self, base_obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {
            "observation": base_obs[self._obs_key],
            "achieved_goal": base_obs[self._achieved_goal_key],
            "achieved_state_goal": base_obs[self._achieved_state_goal_key],
        }

    def _goal_to_desired_goal_and_desired_state_goal(
        self, goal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        return {
            "desired_goal": goal,
            "desired_state_goal": goal,
        }

    def _task_config_to_obs(self) -> Dict[str, np.ndarray]:
        task_config = list(self._task_config.values())
        task_config = np.concatenate(task_config)
        return {
            "task_config": task_config,
        }

    def render(self):
        self._base_env.render()
