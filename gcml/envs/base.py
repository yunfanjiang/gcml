from typing import Callable, Union, Any, Dict
from abc import abstractmethod

import gym
import numpy as np


class GoalReachingEnv(gym.Env):
    """
    We assume a dict observation space.
    """

    def __init__(
        self, *args, **kwargs,
    ):
        self._goal = None
        self._goal_distribution = None

    @abstractmethod
    def reset(self, task_config: Any):
        """
        Reset a goal-reaching env given a task config.
        A task config can be a value that characterize the task.
        E.g., the length of the pendulum in the pendulum task.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_goal(self, *args, **kwargs):
        """
        Sample a goal from some distribution.
        It should write `self._gaol`.
        """
        raise NotImplementedError

    @property
    def goal(self):
        """
        Return the sampled goal.
        """
        return self._goal

    @property
    def goal_distribution(self):
        """
        Return the distribution from which the goal is sampled.
        """
        return self._goal_distribution


class MetaGoalReachingEnv(object):
    """
    Base class for meta goal-reaching env
    """

    def __init__(
        self,
        base_env: GoalReachingEnv,
        metric_fn: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
        task_config_space: gym.Space,
        obs_key: str,
        achieved_goal_key: str,
        achieved_state_goal_key: str,
    ):
        super(MetaGoalReachingEnv, self).__init__()

        self._base_env = base_env
        self._metric_fn = metric_fn
        self._obs_key = obs_key
        self._achieved_goal_key = achieved_goal_key
        self._achieved_state_goal_key = achieved_state_goal_key

        # prepare observation space
        # observation space is a dict space of base_observation, task_config, goal, and state goal
        # we assume that a dict observation space of the base env contains all spaces we need
        all_spaces: gym.spaces.Dict = base_env.observation_space
        base_obs_space = all_spaces[obs_key]
        goal_space = all_spaces[achieved_goal_key]
        state_goal_space = all_spaces[achieved_state_goal_key]
        self._obs_space = gym.spaces.Dict(
            {
                "observation": base_obs_space,
                "task_config": task_config_space,
                "achieved_goal": goal_space,
                "achieved_state_goal": state_goal_space,
                "desired_goal": goal_space,
                "desired_state_goal": state_goal_space,
            }
        )

        # prepare action space
        # action space is the native action space of the base env
        self._action_space = base_env.action_space

        self._task_config = None

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def sample_task(self):
        """
        Sample a new task configuration from task distribution
        """
        self._task_config = self._sample_task()

    def _sample_task(self) -> Any:
        """
        Sample and return a task configuration
        """
        raise NotImplementedError

    def reset(
        self, sample_new_goal: bool = True, *args, **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Reset the base env with task configuration and optionally sample a new goal
        """
        if sample_new_goal:
            self._base_env.sample_goal(*args, **kwargs)
        base_obs = self._base_env.reset(task_config=self._task_config)
        obs = self._prepare_obs(base_obs)
        return obs

    def step(self, action: np.ndarray):
        action = self._preprocess_action(action)
        # we omit reward here because it doesn't matter in GCML/GCSL
        next_base_obs, _, done, info = self._base_env.step(action)
        next_obs = self._prepare_obs(next_base_obs)
        info.update(
            {
                "metric": self._metric_fn(
                    next_obs["achieved_state_goal"], next_obs["desired_state_goal"],
                )
            }
        )
        return next_obs, 0, done, info

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """
        Override if preprocess on action is required, e.g., clip
        """
        return action

    def _prepare_obs(self, base_obs) -> Dict[str, np.ndarray]:
        obs = {}
        obs.update(self._base_obs_to_obs(base_obs))
        obs.update(
            self._goal_to_desired_goal_and_desired_state_goal(self._base_env.goal)
        )
        obs.update(self._task_config_to_obs())
        return obs

    def _base_obs_to_obs(
        self, base_obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Process `base_obs` to a dict
        `{'observation': np.ndarray, 'achieved_goal': np.ndarray, 'achieved_state_goal': np.ndarray}`
        :param base_obs:
        :return:
        """
        raise NotImplementedError

    def _goal_to_desired_goal_and_desired_state_goal(
        self, goal: Any
    ) -> Dict[str, np.ndarray]:
        """
        Process `goal` to a dict `{'desired_goal': np.ndarrary, 'desired_state_goal': np.ndarray}`
        """
        raise NotImplementedError

    def _task_config_to_obs(self) -> Dict[str, np.ndarray]:
        """
        Process `self._task_config` to a dict `{'task_config': np.ndarray}`
        """
        raise NotImplementedError
