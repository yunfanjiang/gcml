import numpy as np
import gym
from typing import Dict
from gcml.envs.base import MetaGoalReachingEnv
from gcml.envs.pendulum.pendulum_base import PendulumEnv


def _metric_fn(curr_theta, goal_theta):
    return goal_theta - curr_theta


class MetaPendulumEnv(MetaGoalReachingEnv):
    def __init__(self):
        base_env = PendulumEnv()

        task_config_space = gym.spaces.Dict(
            {
                "gravity": gym.spaces.Box(low=0.16*9.8, high=2.64*9.8, dtype=np.float32),
                "mass": gym.spaces.Box(low=0.5, high=2.0, dtype=np.float32),
                "length": gym.spaces.Box(low=0.5, high=2.0, dtype=np.float32),
            }
        )
        self._task_config_space = task_config_space

        super(MetaPendulumEnv, self).__init__(
            base_env=base_env,
            metric_fn=_metric_fn,
            task_config_space=task_config_space,
            obs_key="base_obs",
            achieved_goal_key="achieved_goal",
            achieved_state_goal_key="achieved_state_goal"
        )

    def _sample_task(self):
        # g values on different planets
        gravities = np.array(
            [0.37*9.8, 0.98*9.8, 1*9.8, 0.16*9.8, 0.38*9.8, 2.64*9.8, 1.15*9.8, 0.93*9.8, 1.22*9.8], dtype=np.float32
        )
        sampled_task_config = {
            "gravity": np.random.choice(gravities, replace=False),
            "mass": np.random.uniform(low=0.5, high=2.0),
            "length": np.random.uniform(low=0.5, high=2.0),
        }
        return sampled_task_config

    def reset(self, sample_new_goal: bool = True,) -> Dict[str, np.ndarray]:
        if sample_new_goal:
            self._base_env.sample_goal()
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
            "desired_goal": np.array([np.cos(goal), np.sin(goal)], dtype=np.float32),
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




















