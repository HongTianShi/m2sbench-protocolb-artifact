from __future__ import annotations

import numpy as np


def moment_errors(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, dtype=float)
    pred_mean = pts.mean(axis=0)
    pred_cov = np.cov(pts.T, bias=True)
    return {
        "mean_linf_error": float(np.abs(pred_mean - np.asarray(mean, dtype=float)).max()),
        "cov_linf_error": float(np.abs(pred_cov - np.asarray(covariance, dtype=float)).max()),
    }


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((np.asarray(a, dtype=float)[:, None, :] - np.asarray(b, dtype=float)[None, :, :]) ** 2).sum(axis=2))


def normalized_chamfer(prediction: np.ndarray, target: np.ndarray, covariance: np.ndarray) -> float:
    distances = pairwise_distances(prediction, target)
    norm = float(np.sqrt(np.trace(np.asarray(covariance, dtype=float))) + 1e-8)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / norm)


def coverage(prediction: np.ndarray, target: np.ndarray) -> float:
    distances = pairwise_distances(prediction, target)
    target_distances = pairwise_distances(target, target)
    np.fill_diagonal(target_distances, np.inf)
    radius = float(np.quantile(target_distances.min(axis=1), 0.5))
    return float((distances.min(axis=0) <= radius).mean())


def raster_iou(a: np.ndarray, b: np.ndarray, resolution: int = 64) -> float:
    ra = _raster(a, resolution)
    rb = _raster(b, resolution)
    union = np.logical_or(ra, rb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(ra, rb).sum() / union)


def _raster(points: np.ndarray, resolution: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scaled = ((pts - lo) / span * (resolution - 1)).astype(int)
    scaled = np.clip(scaled, 0, resolution - 1)
    grid = np.zeros((resolution, resolution), dtype=bool)
    grid[scaled[:, 1], scaled[:, 0]] = True
    return grid
