import gym
import numpy as np
from gym.utils import seeding


class PendulumWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.seed()
        self.env.sample_goal()

        self.g = self.np_random.uniform(low=0.16 * 9.8, high=2.64 * 9.8)
        self.m = self.np_random.uniform(low=0.5, high=1.5)
        self.l = self.np_random.uniform(low=0.5, high=1.5)
        self.env.reset(self.g, self.m, self.l)
        
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
        self.env.sample_goal()
        return self.env.reset(self.g, self.m, self.l)
