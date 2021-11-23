import numpy as np


def metric_fn(cur_coordinate: np.ndarray, goal_coordinate: np.ndarray):
    """
    Calculate Euclidean distance between current and goal coordinates.
    """
    if cur_coordinate.ndim == 1:
        cur_coordinate = cur_coordinate[np.newaxis, ...]
    if goal_coordinate.ndim == 1:
        goal_coordinate = goal_coordinate[np.newaxis, ...]
    return np.linalg.norm(cur_coordinate - goal_coordinate, axis=-1)
