import gym
import numpy as np

from gcml.envs.lunar_any_landers.lunar_lander_base import (
    LunarLander,
    VIEWPORT_H,
    SCALE,
    CHUNKS,
)


class LunarLanderGoalEnv(gym.GoalEnv):
    def __init__(self, goal_threshold: float, episode_len: int):
        self._env = LunarLander(goal_threshold=goal_threshold)
        self._episode_len = episode_len
        self._goal_threshold = goal_threshold

        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self._env.observation_space["base_obs"],
                "achieved_goal": gym.spaces.Box(
                    low=np.array([-1, -1]),
                    high=np.array([1, 1]),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "desired_goal": gym.spaces.Box(
                    low=np.array([-1, -1]),
                    high=np.array([1, 1]),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self._env.seed()
        self._height = gym.spaces.Box(
            low=VIEWPORT_H / SCALE / 4, high=VIEWPORT_H / SCALE / 4, shape=(CHUNKS + 1,)
        ).sample()
        self._main_engine_power = 13
        self._side_engine_power = 0.6
        self._leg_height = 8
        self._leg_spring_torque = 40
        self._lander_density = 5
        self._env.sample_goal(self._height)

        self._step_counter = None

    def step(self, action):
        raw_obs_dict, _, _, _ = self._env.step(action)
        self._step_counter += 1
        obs_dict = {
            "observation": raw_obs_dict["base_obs"],
            "achieved_goal": raw_obs_dict["achieved_state_goal"],
            "desired_goal": self._env.goal,
        }
        reward = self.compute_reward(
            obs_dict["achieved_goal"], obs_dict["desired_goal"], {}
        )
        if self._step_counter >= self._episode_len:
            done = True
        else:
            done = False
        return obs_dict, reward, done, {}

    def reset(self):
        self._env.sample_goal(self._height)
        self._step_counter = 0
        raw_obs_dict = self._env.reset(
            height=self._height,
            main_engine_power=self._main_engine_power,
            side_engine_power=self._side_engine_power,
            leg_height=self._leg_height,
            leg_spring_torque=self._leg_spring_torque,
            lander_density=self._lander_density,
        )
        obs_dict = {
            "observation": raw_obs_dict["base_obs"],
            "achieved_goal": raw_obs_dict["achieved_state_goal"],
            "desired_goal": self._env.goal,
        }
        return obs_dict

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        if achieved_goal.ndim == 1 and desired_goal.ndim == 1:
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            if distance <= self._goal_threshold:
                return 1.0
            else:
                return 0.0
        elif achieved_goal.ndim == 2 and desired_goal.ndim == 2:
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            reward = np.float32(distance <= self._goal_threshold)
            return reward
        else:
            raise ValueError
