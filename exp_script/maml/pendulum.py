import argparse

import ray
import ray.rllib.agents.maml as maml

from gcml.envs.rllib_meta_env import PendulumRLlibMetaEnv
from ray.tune import tune
from ray.tune.registry import register_env


EPISODE_LEN = 50
GOAL_THRESHOLD = 0.1


def env_creator(env_config):
    return PendulumRLlibMetaEnv(**env_config)


def train(cfg):
    ray.init()

    register_env("any_pendulum", env_creator)

    config = maml.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["rollout_fragment_length"] = EPISODE_LEN
    config["maml_optimizer_steps"] = 1
    config["lr"] = 1e-4
    config["inner_lr"] = 1e-4

    config["env"] = "any_pendulum"
    config["env_config"] = {
        "goal_threshold": GOAL_THRESHOLD,
        "episode_len": EPISODE_LEN,
    }
    tune.run("MAML", config=config, name=args.exp_name, local_dir=args.local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, required=True,
    )
    parser.add_argument(
        "--local_dir", type=str,
    )
    args = parser.parse_args()
    train(args)
