import numpy as np

EPS = 1e-8

def _cov(x):
    return np.cov(np.asarray(x, float).T, bias=True)

def affine_match(points: np.ndarray, mean, covariance, eps: float = EPS) -> np.ndarray:
    y = np.asarray(points, dtype=float)
    mu_y = y.mean(axis=0)
    cov_y = _cov(y)
    evals, evecs = np.linalg.eigh(cov_y + eps * np.eye(2))
    sqrt_inv = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, eps, None))) @ evecs.T
    tgt = np.asarray(covariance, dtype=float)
    tevals, tevecs = np.linalg.eigh(tgt + eps * np.eye(2))
    sqrt_t = tevecs @ np.diag(np.sqrt(np.clip(tevals, eps, None))) @ tevecs.T
    z = (y - mu_y) @ sqrt_inv.T @ sqrt_t.T + np.asarray(mean, dtype=float)
    return z

def final_affine_correction(points: np.ndarray, mean, covariance, eps: float = EPS) -> np.ndarray:
    return affine_match(points, mean, covariance, eps=eps)
