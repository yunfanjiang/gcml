import numpy as np


def metric_fn(curr_theta, goal_theta):
    """
    Calculate the angle difference between a target angle and the current angle.
    :param curr_theta: float, [-pi, +pi]
    :param goal_theta: float, [-pi, +pi]
    :return: float, [0, +pi]
    """
    goal_theta = np.array(goal_theta)
    curr_theta = np.array(curr_theta)
    if goal_theta.ndim == 1:
        goal_theta = goal_theta[np.newaxis, ...]
    if curr_theta.ndim == 1:
        curr_theta = curr_theta[np.newaxis, ...]

    diff = goal_theta - curr_theta
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    return np.abs(diff)
