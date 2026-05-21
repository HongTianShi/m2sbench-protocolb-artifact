import numpy as np
from .sampling import farthest_point_sample
from .affine_match import affine_match

def construct_canonical_target(dense_support: np.ndarray, n_points: int, mean, covariance, seed: int = 0):
    sampled = farthest_point_sample(dense_support, n_points, seed=seed)
    embedded = affine_match(sampled, mean, covariance)
    return embedded
