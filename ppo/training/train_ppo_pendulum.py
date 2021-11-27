import stable_baselines3
import gym
import numpy as np

from gcml.envs.base import GoalReachingEnv
from gcml.envs.pendulum.pendulum_base import PendulumEnv
from ppo.env_utils.pendulum_wrapper import PendulumWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from gcml.envs.pendulum.metric import metric_fn


def evaluate(model, env, num_eval_episodes=100, deterministic=True):
    all_episode_distances = []

    for i in range(num_eval_episodes):
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
        distance = metric_fn([obs["achieved_state_goal"]], [env.env._goal])
        all_episode_distances.append(distance)

    mean_episode_distance = np.mean(all_episode_distances)
    std_episode_distance = np.std(all_episode_distances)
    return mean_episode_distance, std_episode_distance


if __name__ == "__main__":
    env = PendulumWrapper(PendulumEnv(goal_threshold=0.05), max_episode_steps=100)
    model = PPO(MultiInputPolicy, env, verbose=0)

    # Random Agent, before training
    mean_episode_distance, std_episode_distance = evaluate(model, env, num_eval_episodes=100, deterministic=True)
    print(f"Before training: mean_episode_distance = {mean_episode_distance:.2f} +/- {std_episode_distance:.2f}")

    # Train the agent for 10000 steps, and evaluate the trained agent
    model.learn(total_timesteps=10000)
    mean_episode_distance,  std_episode_distance = evaluate(model, env, num_eval_episodes=100, deterministic=True)
    print(f"After training: mean_episode_distance = {mean_episode_distance:.2f} +/- {std_episode_distance:.2f}")

