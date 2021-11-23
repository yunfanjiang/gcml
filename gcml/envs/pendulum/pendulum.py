import numpy as np
import gym
from typing import Dict
from gcml.envs.base import MetaGoalReachingEnv
from gcml.envs.pendulum.pendulum_base import PendulumEnv
from gcml.envs.pendulum.metric import metric_fn


class MetaPendulumEnv(MetaGoalReachingEnv):
    def __init__(self, goal_threshold):
        base_env = PendulumEnv(goal_threshold)

        task_config_space = gym.spaces.Dict(
            {
                "gravity": gym.spaces.Box(
                    low=0.16 * 9.8, high=2.64 * 9.8, dtype=np.float32, shape=(1,)
                ),
                "mass": gym.spaces.Box(low=0.5, high=1.5, dtype=np.float32, shape=(1,)),
                "length": gym.spaces.Box(
                    low=0.5, high=1.5, dtype=np.float32, shape=(1,)
                ),
            }
        )
        self._task_config_space = task_config_space

        super(MetaPendulumEnv, self).__init__(
            base_env=base_env,
            metric_fn=metric_fn,
            task_config_space=task_config_space,
            obs_key="base_obs",
            achieved_goal_key="achieved_goal",
            achieved_state_goal_key="achieved_state_goal",
        )

    def _sample_task(self):
        sampled_task_config = {
            key: self._task_config_space[key].sample()
            for key in self._task_config_space
        }
        return sampled_task_config

    def reset(
        self, sample_new_goal: bool = True, predefined_goal=None
    ) -> Dict[str, np.ndarray]:
        if sample_new_goal:
            self._base_env.sample_goal()
        elif predefined_goal is not None:
            self._base_env.set_goal(predefined_goal)
        else:
            raise ValueError(f"Either sample a new goal or provide a predefined goal")
        base_obs = self._base_env.reset(
            g=self._task_config["gravity"].item(),
            m=self._task_config["mass"].item(),
            l=self._task_config["length"].item(),
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
            "desired_goal": np.array([np.cos(goal), np.sin(goal)], dtype=np.float32),
            "desired_state_goal": np.array([goal], dtype=np.float32),
        }

    def _task_config_to_obs(self) -> Dict[str, np.ndarray]:
        task_config = list(self._task_config.values())
        task_config = np.concatenate(task_config)
        return {
            "task_config": task_config,
        }

    def render(self):
        self._base_env.render()
