import gym
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from ...pendulum.pendulum_base import PendulumEnv


class PendulumRLlibMetaEnv(TaskSettableEnv):
    def __init__(
        self, goal_threshold: float, episode_len: int,
    ):
        task_config_space = gym.spaces.Dict(
            {
                "g": gym.spaces.Box(
                    low=0.16 * 9.8, high=2.64 * 9.8, dtype=np.float32, shape=(1,)
                ),
                "m": gym.spaces.Box(low=0.5, high=1.5, dtype=np.float32, shape=(1,)),
                "l": gym.spaces.Box(low=0.5, high=1.5, dtype=np.float32, shape=(1,)),
            }
        )
        self._task_config_space = task_config_space

        self._base_env = PendulumEnv(goal_threshold=goal_threshold)
        base_obs_space = self._base_env.observation_space
        base_obs_space_dict = {k: base_obs_space[k] for k in base_obs_space}
        goal_high = np.array([1.0, 1.0], dtype=np.float32)
        base_obs_space_dict.update(
            {
                "desired_goal": gym.spaces.Box(
                    low=-goal_high, high=goal_high, dtype=np.float32, shape=(2,)
                ),
                "desired_state_goal": gym.spaces.Box(
                    low=-np.pi, high=np.pi, dtype=np.float32, shape=(1,)
                ),
            }
        )
        self.observation_space = gym.spaces.Dict(base_obs_space_dict)
        self._episode_len = episode_len
        self._task_config = {
            key: self._task_config_space[key].sample()
            for key in self._task_config_space
        }
        self._step_counter = None

    @property
    def action_space(self):
        return self._base_env.action_space

    def seed(self, seed=None):
        return self._base_env.seed(seed)

    def sample_tasks(self, n_tasks: int):
        return [
            {
                key: self._task_config_space[key].sample()
                for key in self._task_config_space
            }
            for _ in range(n_tasks)
        ]

    def set_task(self, task):
        self._task_config = task

    def get_task(self):
        return self._task_config

    def reset(self):
        self._step_counter = 0

        self._base_env.sample_goal()
        task_config = {k: v.item() for k, v in self._task_config.items()}
        base_obs = self._base_env.reset(**task_config)
        goal = self._base_env.goal
        goal_obs = {
            "desired_goal": np.array([np.cos(goal), np.sin(goal)], dtype=np.float32),
            "desired_state_goal": np.array([goal], dtype=np.float32),
        }
        base_obs.update(goal_obs)
        return base_obs

    def step(self, action):
        next_base_obs, reward, _, info = self._base_env.step(action)
        done = False
        if self._step_counter >= self._episode_len:
            done = True
        # according to GCSL goal-reaching problem formulation,
        # the agent is rewarded iff it reaches the goal at the LAST time step
        if not done:
            reward = 0
        self._step_counter += 1

        goal = self._base_env.goal
        goal_obs = {
            "desired_goal": np.array([np.cos(goal), np.sin(goal)], dtype=np.float32),
            "desired_state_goal": np.array([goal], dtype=np.float32),
        }
        next_base_obs.update(goal_obs)

        return next_base_obs, reward, done, info
