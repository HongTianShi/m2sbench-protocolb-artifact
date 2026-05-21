from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _seed_from_cell(cell_id: str) -> int:
    return int(hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:8], 16)


def _moment_matched_points(cell: dict[str, Any]) -> list[list[float]]:
    summary = cell["summary"]
    n_points = int(cell.get("n_points", summary.get("n_points", 64)))
    mean = np.asarray(summary["mean"], dtype=float)
    covariance = np.asarray(summary["covariance"], dtype=float)
    rng = np.random.default_rng(_seed_from_cell(str(cell["cell_id"])))
    raw = rng.normal(size=(n_points, len(mean)))
    raw -= raw.mean(axis=0, keepdims=True)
    src_cov = np.cov(raw.T, bias=True) + np.eye(len(mean)) * 1e-8
    src_vals, src_vecs = np.linalg.eigh(src_cov)
    tgt_vals, tgt_vecs = np.linalg.eigh(covariance + np.eye(len(mean)) * 1e-8)
    whiten = src_vecs @ np.diag(1.0 / np.sqrt(np.maximum(src_vals, 1e-8))) @ src_vecs.T
    color = tgt_vecs @ np.diag(np.sqrt(np.maximum(tgt_vals, 1e-8))) @ tgt_vecs.T
    points = raw @ whiten.T @ color.T + mean
    return np.round(points, 6).tolist()


def _rank_views(cell: dict[str, Any]) -> list[str]:
    covariance = np.asarray(cell["summary"]["covariance"], dtype=float)
    eigvals = np.linalg.eigvalsh(covariance)
    condition = float(eigvals.max() / max(eigvals.min(), 1e-8))
    if condition >= 5.0:
        return ["point", "raster", "occupancy", "moments"]
    if condition >= 2.0:
        return ["raster", "occupancy", "point", "moments"]
    return ["moments", "occupancy", "raster", "point"]


def solve(cell: dict[str, Any]) -> dict[str, Any]:
    """
    Input: method-visible summary S and granted views only.
    Output: candidate structures, ranked views, or routing decisions.
    Hidden targets are never exposed to this function.
    """
    return {
        "cell_id": cell["cell_id"],
        "method": "template_moment_sampler",
        "candidates": [{"points": _moment_matched_points(cell)}],
        "ranked_views": _rank_views(cell),
        "route": "direct",
    }
