import numpy as np
from scipy.optimize import linear_sum_assignment

def assign_points(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    cost = ((source[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    r, c = linear_sum_assignment(cost)
    return c
