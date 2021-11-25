import gym
import numpy as np
from gym.utils import seeding
from gcml.envs.lunar_any_landers.lunar_lander_base import (
    VIEWPORT_H,
    SCALE,
    CHUNKS,
)

class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.seed()

        self.height = self.np_random.uniform(low=0, high=VIEWPORT_H / SCALE / 4, size=(CHUNKS + 1,))
        self.main_engine_power = self.np_random.uniform(low=13, high=20)
        self.side_engine_power = self.np_random.uniform(low=0.6, high=1)
        self.leg_height = self.np_random.uniform(low=8, high=12)
        self.leg_spring_torque = self.np_random.uniform(low=40, high=60)
        self.lander_density = self.np_random.uniform(low=2, high=5)
        self.env.sample_goal(self.height)
        self.env.reset(
            self.height, 
            self.main_engine_power, 
            self.side_engine_power, 
            self.leg_height, 
            self.leg_spring_torque, 
            self.lander_density,
            )
        
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps == self._max_episode_steps and reward == 1:
            reward = 1
        else:
            reward = 0
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.env.sample_goal(self.height)
        return self.env.reset(
            self.height,
            self.main_engine_power,
            self.side_engine_power,
            self.leg_height,
            self.leg_spring_torque,
            self.lander_density,
            )
