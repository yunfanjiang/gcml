import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv

from gcml.envs.sb3_goal_env import PendulumGoalEnv, Monitor
from gcml.envs.pendulum.metric import metric_fn


model_class = DDPG  # works also with SAC, DDPG and TD3
EPISODE_LEN = 50
GOAL_THRESHOLD = 0.1

env = PendulumGoalEnv(goal_threshold=GOAL_THRESHOLD, episode_len=EPISODE_LEN)
env = Monitor(env, metric_fn=metric_fn, max_distance=np.pi)

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
    tensorboard_log="./pendulum",
)

# Train the model
model.learn(1000000)
