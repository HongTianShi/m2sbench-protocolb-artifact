from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _seed(cell_id: str) -> int:
    return int(hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:8], 16)


def _candidate(cell: dict[str, Any]) -> list[list[float]]:
    summary = cell["summary"]
    n_points = int(cell.get("n_points", summary.get("n_points", 64)))
    mean = np.asarray(summary["mean"], dtype=float)
    covariance = np.asarray(summary["covariance"], dtype=float)
    rng = np.random.default_rng(_seed(str(cell["cell_id"])))
    raw = rng.normal(size=(n_points, len(mean)))
    raw -= raw.mean(axis=0, keepdims=True)
    src_cov = np.cov(raw.T, bias=True) + np.eye(len(mean)) * 1e-8
    src_vals, src_vecs = np.linalg.eigh(src_cov)
    tgt_vals, tgt_vecs = np.linalg.eigh(covariance + np.eye(len(mean)) * 1e-8)
    points = raw @ (src_vecs @ np.diag(1.0 / np.sqrt(np.maximum(src_vals, 1e-8))) @ src_vecs.T).T
    points = points @ (tgt_vecs @ np.diag(np.sqrt(np.maximum(tgt_vals, 1e-8))) @ tgt_vecs.T).T + mean
    return np.round(points, 6).tolist()


def solve(cell: dict[str, Any]) -> dict[str, Any]:
    """
    Input: method-visible Protocol B cell only.
    Hidden target and evaluator-only views are never exposed.
    """
    ranked_views = ["moments", "occupancy", "raster", "point"]
    return {
        "cell_id": cell["cell_id"],
        "method": "adapter_template",
        "candidate": _candidate(cell),
        "ranked_views": ranked_views,
        "route": ranked_views[0],
    }
