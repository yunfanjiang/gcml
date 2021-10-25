import numpy as np
import gym
from gcml.envs.base import MetaGoalReachingEnv
from gcml.envs.pendulum.pendulum_base import PendulumEnv

class MetaPendulumEnv(MetaGoalReachingEnv):
    def __init__(self):
        base_env = PendulumEnv()
        


