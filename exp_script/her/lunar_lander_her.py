import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3

from gcml.envs.sb3_goal_env import LunarLanderGoalEnv, Monitor

model_class = DQN  # works also with SAC, DDPG and TD3
EPISODE_LEN = 250
GOAL_THRESHOLD = 0.1

env = LunarLanderGoalEnv(goal_threshold=GOAL_THRESHOLD, episode_len=EPISODE_LEN)
env = Monitor(
    env, metric_fn=lambda x, y: np.linalg.norm(x - y), max_distance=2 * np.sqrt(2)
)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future"

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = EPISODE_LEN

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=2,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
    tensorboard_log="./lunar_lander",
)

# Train the model
model.learn(1000000)
