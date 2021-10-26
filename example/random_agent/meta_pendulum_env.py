from typing import Dict

import gym
import numpy as np
from tqdm import tqdm

from gcml.envs import MetaPendulumEnv


class RandomAgent(object):
    """
    A random agent.
    """

    def __init__(
        self, action_space: gym.Space,
    ):
        self._action_space = action_space

    def act(self, observation: Dict[str, np.ndarray]):
        assert "observation" in observation
        assert "achieved_goal" in observation
        assert "achieved_state_goal" in observation
        assert "desired_goal" in observation
        assert "desired_state_goal" in observation
        assert "task_config" in observation
        return self._action_space.sample()

    def learn(self, *args, **kwargs):
        pass

    def adapt(self, *args, **kwargs):
        pass


class DummyBuffer(object):
    def __init__(self):
        pass

    def store_transition(self, *args, **kwargs):
        pass

    def relabel(self, *args, **kwargs):
        return


if __name__ == "__main__":
    # we follow GCSL that we manually define the episode length
    EPISODE_LEN = 500
    # number of tasks
    N_TASKS = 2

    # instantiate meta env, a dummy buffer, and a random agent
    meta_env = MetaPendulumEnv()
    learner = RandomAgent(action_space=meta_env.action_space)
    buffer = DummyBuffer()

    for task_i in tqdm(range(N_TASKS)):
        # sample a new task
        meta_env.sample_task()
        # start collecting data
        print(f"task {task_i} start collecting trajectory for adaptation")
        obs = meta_env.reset()
        for step in range(EPISODE_LEN):
            action = learner.act(obs)
            next_obs, _, _, info = meta_env.step(action)
            meta_env.render()
            # store transition into buffer
            buffer.store_transition(next_obs, info)
            obs = next_obs
        print(f"task {task_i} finish collecting trajectory for adaptation")
        # use generated data to do inner-loop adaptation
        relabeled_data = buffer.relabel()
        learner.adapt(relabeled_data)

        # generate a new trajectory
        print(f"task {task_i} start collecting trajectory for meta learn")
        obs = meta_env.reset()
        for step in range(EPISODE_LEN):
            action = learner.act(obs)
            next_obs, _, _, info = meta_env.step(action)
            meta_env.render()
            # store transition into buffer
            buffer.store_transition(next_obs, info)
            obs = next_obs
        print(f"task {task_i} finish collecting trajectory for meta learn")

    # finally, use newly relabeled demonstrations to meta-learn
    learner.learn()
