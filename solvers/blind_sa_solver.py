from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _seed(cell_id: str) -> int:
    return int(hashlib.sha256(f"blind-sa:{cell_id}".encode("utf-8")).hexdigest()[:8], 16)


def _moment_loss(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    pred_mean = points.mean(axis=0)
    pred_cov = np.cov(points.T, bias=True)
    return float(np.abs(pred_mean - mean).max() + np.abs(pred_cov - covariance).max())


def solve(cell: dict[str, Any]) -> dict[str, Any]:
    support = np.asarray(cell["granted_views"]["support_points"], dtype=float)
    n_points = int(cell["n_points"])
    mean = np.asarray(cell["summary"]["mean"], dtype=float)
    covariance = np.asarray(cell["summary"]["covariance"], dtype=float)
    rng = np.random.default_rng(_seed(str(cell["cell_id"])))
    best = None
    best_loss = float("inf")
    for _ in range(80):
        idx = rng.choice(len(support), size=n_points, replace=len(support) < n_points)
        candidate = support[idx]
        loss = _moment_loss(candidate, mean, covariance)
        if loss < best_loss:
            best = candidate
            best_loss = loss
    ranked_views = ["occupancy", "raster", "moments", "point"] if best_loss > 0.15 else ["moments", "occupancy", "raster", "point"]
    return {
        "cell_id": cell["cell_id"],
        "method": "blind_sa_adapter_demo",
        "candidate": np.round(best, 6).tolist(),
        "ranked_views": ranked_views,
        "route": ranked_views[0],
    }
