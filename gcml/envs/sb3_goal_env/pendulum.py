import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gcml.envs.pendulum.metric import metric_fn


class PendulumGoalEnv(gym.GoalEnv):
    def __init__(self, goal_threshold: float, episode_len: int):
        super().__init__()

        self.max_angular_speed = 8.0
        self.max_torque = 2.0
        self.action_scale = 2.0
        self.dt = 0.05
        self.g = np.random.uniform(low=0.16 * 9.8, high=2.64 * 9.8)
        self.m = np.random.uniform(low=0.5, high=1.5)
        self.l = np.random.uniform(low=0.5, high=1.5)
        self.state = None
        self.last_action = None
        self.viewer = None
        self.goal_threshold = goal_threshold
        self._episode_len = episode_len

        obs_high = np.array([1.0, 1.0, self.max_angular_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-obs_high, high=obs_high, dtype=np.float32, shape=(3,)
                ),
                "achieved_goal": spaces.Box(
                    low=np.array([-np.pi,], dtype=np.float32),
                    high=np.array([np.pi,], dtype=np.float32),
                    dtype=np.float32,
                    shape=(1,),
                ),
                "desired_goal": spaces.Box(
                    low=np.array([-np.pi,], dtype=np.float32),
                    high=np.array([np.pi,], dtype=np.float32),
                    dtype=np.float32,
                    shape=(1,),
                ),
            }
        )
        self.seed()
        self._sample_goal()
        self._step_counter = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_goal(self):
        base_goal = self.np_random.uniform(low=-np.pi, high=np.pi)
        self._goal = base_goal

    def step(self, action):
        theta, angular_speed = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action *= self.action_scale
        action = np.clip(action, -self.max_torque, self.max_torque)
        self.last_action = action
        new_angular_speed = (
            angular_speed
            + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l ** 2) * action) * dt
        )
        new_angular_speed = np.clip(
            new_angular_speed, -self.max_angular_speed, self.max_angular_speed
        )
        new_theta = theta + new_angular_speed * dt
        new_theta = np.arctan(np.sin(new_theta) / np.cos(new_theta))
        self.state = np.array([new_theta, new_angular_speed])

        self._step_counter += 1
        if self._step_counter >= self._episode_len:
            done = True
        else:
            done = False

        obs_dict = self._get_obs_dict()
        reward = self.compute_reward(
            achieved_goal=obs_dict["achieved_goal"],
            desired_goal=obs_dict["desired_goal"],
            info={},
        )

        return obs_dict, reward, done, {}

    def reset(self):
        self._sample_goal()
        self._step_counter = 0
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_obs_dict()

    def _get_obs_dict(self):
        theta, angular_speed = self.state
        if theta.ndim == 0:
            theta = theta[np.newaxis, ...]
        obs_dict = {
            "observation": np.array(
                [np.cos(theta), np.sin(theta), angular_speed], dtype=np.float32
            ).reshape(3,),
            "achieved_goal": theta,
            "desired_goal": np.array([self._goal,], dtype=np.float32),
        }
        return obs_dict

    def compute_reward(self, achieved_goal, desired_goal, info):
        # rollout case
        # if achieved_goal.ndim == 1 and desired_goal.ndim == 1:
        if metric_fn([achieved_goal[0]], [desired_goal[0]]) <= self.goal_threshold:
            return 1.0
        else:
            return 0.0
        # elif achieved_goal.ndim == 2 and desired_goal.ndim == 2:
        #     distance = metric_fn(achieved_goal[:, 0], desired_goal[:, 0])[0]
        #     reward = np.float32(distance <= self.goal_threshold)
        #     return reward
        # else:
        #     print(achieved_goal.shape, desired_goal.shape)
        #     raise ValueError
