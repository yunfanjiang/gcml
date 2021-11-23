import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path
from gcml.envs.base import GoalReachingEnv
from gcml.envs.pendulum.metric import metric_fn


class PendulumEnv(GoalReachingEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_threshold):
        GoalReachingEnv.__init__(self)

        self.max_angular_speed = 8.0
        self.max_torque = 2.0
        self.action_scale = 2.0
        self.dt = 0.05
        self.g = None
        self.m = None
        self.l = None
        self.state = None
        self.last_action = None
        self.viewer = None
        self.goal_threshold = goal_threshold

        obs_high = np.array([1.0, 1.0, self.max_angular_speed], dtype=np.float32)
        goal_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "base_obs": spaces.Box(
                    low=-obs_high, high=obs_high, dtype=np.float32, shape=(3,)
                ),
                "achieved_goal": spaces.Box(
                    low=-goal_high, high=goal_high, dtype=np.float32, shape=(2,)
                ),
                "achieved_state_goal": spaces.Box(
                    low=-np.pi, high=np.pi, dtype=np.float32, shape=(1,)
                ),
            }
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_goal(self):
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
        self.state = np.array([new_theta, new_angular_speed])

        reward = 0
        if metric_fn([self.state[0]], [self._goal]) <= self.goal_threshold:
            reward = 1

        return self._get_obs_dict(), reward, False, {}

    def reset(self, g: float, m: float, l: float):
        self.g = g
        self.m = m
        self.l = l

        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_obs_dict()

    def _get_obs_dict(self):
        theta, angular_speed = self.state
        obs_dict = {
            "base_obs": np.array(
                [np.cos(theta), np.sin(theta), angular_speed], dtype=np.float32
            ),
            "achieved_goal": np.array([np.cos(theta), np.sin(theta)], dtype=np.float32),
            "achieved_state_goal": np.array([theta], dtype=np.float32),
        }
        return obs_dict

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            f_name = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(f_name, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_action is not None:
            self.imgtrans.scale = (-self.last_action / 2, np.abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
