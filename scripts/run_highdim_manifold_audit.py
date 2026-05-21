"""High-dimensional manifold/non-Gaussian matched-summary audit.

This audit addresses a specific boundary question: if the ambient dimension is
large, does matching (mu, Sigma, N) collapse the task into a Gaussian
approximation problem?  The script separates two cases.

1. Full-rank Gaussian controls, where the benchmark should mostly degenerate:
   moment-matched Gaussian replicas have little stable hidden structure.
2. Low-rank manifold or mixture structures embedded in high ambient dimension,
   where the full covariance can be identical while local topology,
   multimodality, or support geometry remains hidden.

All manifold candidates are centered and whitened in the intrinsic coordinates
before being embedded by the same orthonormal basis.  Therefore every candidate
in a cell has the same ambient mean and the same rank-r covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "summaries" / "dimension_stress_audit"

KINDS = ("ring", "two_cluster", "four_cluster", "curve")
VIEW_COSTS = {
    "summary": 0.00,
    "local_graph_sketch": 0.28,
    "higher_order_sketch": 0.42,
    "point": 0.58,
}


@dataclass(frozen=True)
class ManifoldCell:
    ambient_dim: int
    intrinsic_dim: int
    n_points: int
    seed: int
    target_kind: str
    target: np.ndarray
    candidates: list[tuple[str, np.ndarray]]


def _orthonormal_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return q[:, :r]


def _whiten_latent(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z = z - z.mean(axis=0, keepdims=True)
    cov = (z.T @ z) / len(z)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    out = z @ (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T)
    return out - out.mean(axis=0, keepdims=True)


def _latent(kind: str, r: int, n: int, rng: np.random.Generator) -> np.ndarray:
    z = 0.10 * rng.normal(size=(n, r))
    if kind == "ring":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rng.shuffle(t)
        z[:, 0] += 2.0 * np.cos(t)
        if r > 1:
            z[:, 1] += 2.0 * np.sin(t)
        for j in range(2, r):
            z[:, j] += 0.30 * np.cos((j + 1) * t)
    elif kind == "two_cluster":
        labels = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
        rng.shuffle(labels)
        z[:, 0] += np.where(labels == 0, -2.0, 2.0)
        if r > 1:
            z[:, 1] += 0.40 * rng.normal(size=n)
    elif kind == "four_cluster":
        labels = np.arange(n) % 4
        rng.shuffle(labels)
        centers = np.array([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])
        z[:, 0] += centers[labels, 0]
        if r > 1:
            z[:, 1] += centers[labels, 1]
    elif kind == "curve":
        t = np.linspace(-2.0, 2.0, n)
        rng.shuffle(t)
        z[:, 0] += t
        if r > 1:
            z[:, 1] += t**2 - np.mean(t**2)
        if r > 2:
            z[:, 2] += np.sin(3 * t)
    else:
        raise ValueError(kind)
    return _whiten_latent(z)


def _embed(kind: str, basis: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = _latent(kind, basis.shape[1], n, rng)
    return z @ basis.T


def _fullrank_gaussian(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    return x @ (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T)


def _moment_error(x: np.ndarray, target_cov: np.ndarray) -> tuple[float, float]:
    mean_err = float(np.max(np.abs(x.mean(axis=0))))
    cov = (x.T @ x) / len(x)
    cov_err = float(np.max(np.abs(cov - target_cov)))
    return mean_err, cov_err


def _subsample(x: np.ndarray, max_points: int = 128) -> np.ndarray:
    if len(x) <= max_points:
        return x
    idx = np.linspace(0, len(x) - 1, max_points).round().astype(int)
    return x[idx]


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    aa = _subsample(a)
    bb = _subsample(b)
    dist = cdist(aa, bb, metric="sqeuclidean")
    cd = float(dist.min(axis=1).mean() + dist.min(axis=0).mean())
    scale = float(np.median(cdist(aa, aa, metric="sqeuclidean")))
    return cd / max(scale, 1e-9)


def _mmd(a: np.ndarray, b: np.ndarray) -> float:
    aa = _subsample(a)
    bb = _subsample(b)
    z = np.vstack([aa, bb])
    sq = cdist(z, z, metric="sqeuclidean")
    sigma2 = float(np.median(sq[sq > 0]))
    sigma2 = max(sigma2, 1e-6)
    kxx = np.exp(-cdist(aa, aa, metric="sqeuclidean") / (2 * sigma2)).mean()
    kyy = np.exp(-cdist(bb, bb, metric="sqeuclidean") / (2 * sigma2)).mean()
    kxy = np.exp(-cdist(aa, bb, metric="sqeuclidean") / (2 * sigma2)).mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def _local_graph_features(x: np.ndarray) -> np.ndarray:
    xx = _subsample(x, max_points=160)
    dist = cdist(xx, xx, metric="sqeuclidean")
    np.fill_diagonal(dist, np.inf)
    knn = np.sort(dist, axis=1)[:, :8]
    eps = float(np.quantile(knn[:, 5], 0.65))
    graph = dist <= eps
    seen = np.zeros(len(xx), dtype=bool)
    comps = 0
    for i in range(len(xx)):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(graph[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
    pair = dist[np.isfinite(dist)]
    return np.concatenate(
        [
            np.quantile(knn, [0.10, 0.25, 0.50, 0.75, 0.90]),
            np.quantile(pair, [0.10, 0.25, 0.50, 0.75, 0.90]),
            np.array([comps / len(xx), graph.mean()], dtype=float),
        ]
    )


def _higher_order_features(x: np.ndarray) -> np.ndarray:
    xx = _subsample(x, max_points=160)
    norm = np.linalg.norm(xx, axis=1)
    dist = cdist(xx, xx, metric="sqeuclidean")
    pair = dist[np.triu_indices_from(dist, k=1)]
    centered = xx - xx.mean(axis=0, keepdims=True)
    s = centered.std(axis=0) + 1e-9
    z = centered / s
    skew = np.mean(z**3, axis=0)
    kurt = np.mean(z**4, axis=0)
    return np.concatenate(
        [
            np.quantile(norm, [0.10, 0.25, 0.50, 0.75, 0.90]),
            np.quantile(pair, [0.10, 0.25, 0.50, 0.75, 0.90]),
            np.quantile(np.abs(skew), [0.25, 0.50, 0.75]),
            np.quantile(kurt, [0.25, 0.50, 0.75]),
        ]
    )


def _choose(cell: ManifoldCell, view: str) -> str:
    if view == "summary":
        return KINDS[0]
    if view == "local_graph_sketch":
        target_features = _local_graph_features(cell.target)
        scores = [
            float(np.linalg.norm(_local_graph_features(candidate) - target_features))
            for _, candidate in cell.candidates
        ]
    elif view == "higher_order_sketch":
        target_features = _higher_order_features(cell.target)
        scores = [
            float(np.linalg.norm(_higher_order_features(candidate) - target_features))
            for _, candidate in cell.candidates
        ]
    elif view == "point":
        scores = [_chamfer(cell.target, candidate) for _, candidate in cell.candidates]
    else:
        raise ValueError(view)
    return cell.candidates[int(np.argmin(scores))][0]


def _make_cell(d: int, r: int, n: int, seed: int, target_kind: str) -> ManifoldCell:
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(d, r, rng)
    target = _embed(target_kind, basis, n, seed + 1)
    candidates = [(kind, _embed(kind, basis, n, seed + 101 * (idx + 1))) for idx, kind in enumerate(KINDS)]
    return ManifoldCell(d, r, n, seed, target_kind, target, candidates)


def _run_manifold() -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    summary_rows = []
    for d in (50, 128):
        for r in (2, 8, 16):
            for n in (96, 192):
                rows = []
                for seed_idx in range(10):
                    for kind_idx, kind in enumerate(KINDS):
                        seed = 20260516 + d * 10000 + r * 1000 + n + seed_idx * 17 + kind_idx
                        cell = _make_cell(d, r, n, seed, kind)
                        target_cov = (cell.target.T @ cell.target) / len(cell.target)
                        mean_err, cov_err = _moment_error(cell.target, target_cov)
                        gauss = _embed("two_cluster", _orthonormal_basis(d, r, np.random.default_rng(seed + 999)), n, seed + 77)
                        # Re-embed an intrinsic Gaussian in the same basis for a fair moment-matched Gaussian proxy.
                        basis = np.linalg.eigh(target_cov)[1][:, -r:]
                        latent_gauss = _whiten_latent(np.random.default_rng(seed + 555).normal(size=(n, r)))
                        gauss = latent_gauss @ basis.T
                        view_payload = {}
                        best_view = None
                        best_utility = -1e9
                        for view, cost in VIEW_COSTS.items():
                            pred = _choose(cell, view)
                            correct = float(pred == kind)
                            utility = correct - cost
                            view_payload[f"{view}_correct"] = correct
                            view_payload[f"{view}_utility"] = utility
                            if utility > best_utility:
                                best_utility = utility
                                best_view = view
                        row = {
                            "case": "lowrank_manifold",
                            "ambient_dim": d,
                            "intrinsic_dim": r,
                            "n_points": n,
                            "seed": seed,
                            "target_kind": kind,
                            "mean_err": mean_err,
                            "cov_err": cov_err,
                            "gaussian_proxy_cd": _chamfer(cell.target, gauss),
                            "gaussian_proxy_mmd": _mmd(cell.target, gauss),
                            "best_view": best_view,
                            "best_utility": best_utility,
                        } | view_payload
                        rows.append(row)
                        cell_rows.append(row)
                df = pd.DataFrame(rows)
                summary_rows.append(
                    {
                        "case": "lowrank_manifold",
                        "ambient_dim": d,
                        "intrinsic_dim": r,
                        "n_points": n,
                        "cells": len(df),
                        "max_mean_err": df["mean_err"].max(),
                        "max_cov_err": df["cov_err"].max(),
                        "gaussian_proxy_cd": df["gaussian_proxy_cd"].mean(),
                        "gaussian_proxy_mmd": df["gaussian_proxy_mmd"].mean(),
                        "summary_correct": df["summary_correct"].mean(),
                        "local_graph_correct": df["local_graph_sketch_correct"].mean(),
                        "higher_order_correct": df["higher_order_sketch_correct"].mean(),
                        "point_correct": df["point_correct"].mean(),
                        "non_richest_best": float((df["best_view"] != "point").mean()),
                        "local_graph_best": float((df["best_view"] == "local_graph_sketch").mean()),
                        "summary_best": float((df["best_view"] == "summary").mean()),
                    }
                )
    return pd.DataFrame(cell_rows), pd.DataFrame(summary_rows)


def _run_gaussian_control() -> pd.DataFrame:
    rows = []
    for d in (50, 128):
        for n in (192, 384):
            cds = []
            mmds = []
            mean_errs = []
            cov_errs = []
            for seed_idx in range(32):
                seed = 20260516 + d * 1000 + n + seed_idx
                target = _fullrank_gaussian(d, n, seed)
                replica = _fullrank_gaussian(d, n, seed + 10_000)
                target_cov = np.eye(d)
                mean_err, cov_err = _moment_error(target, target_cov)
                mean_errs.append(mean_err)
                cov_errs.append(cov_err)
                cds.append(_chamfer(target, replica))
                mmds.append(_mmd(target, replica))
            rows.append(
                {
                    "case": "fullrank_gaussian_control",
                    "ambient_dim": d,
                    "intrinsic_dim": d,
                    "n_points": n,
                    "cells": 32,
                    "max_mean_err": max(mean_errs),
                    "max_cov_err": max(cov_errs),
                    "gaussian_replica_cd": float(np.mean(cds)),
                    "gaussian_replica_mmd": float(np.mean(mmds)),
                    "interpretation": "degenerate control: no stable non-Gaussian hidden structure is injected",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells, manifold = _run_manifold()
    control = _run_gaussian_control()
    cells.to_csv(OUT_DIR / "highdim_manifold_by_cell.csv", index=False)
    manifold.to_csv(OUT_DIR / "highdim_manifold_summary.csv", index=False)
    control.to_csv(OUT_DIR / "highdim_gaussian_control.csv", index=False)

    d128 = manifold[(manifold["ambient_dim"] == 128) & (manifold["n_points"] == 192)]
    report = {
        "lowrank_d128_summary_correct": float(d128["summary_correct"].mean()),
        "lowrank_d128_local_graph_correct": float(d128["local_graph_correct"].mean()),
        "lowrank_d128_non_richest_best": float(d128["non_richest_best"].mean()),
        "lowrank_d128_gaussian_proxy_cd": float(d128["gaussian_proxy_cd"].mean()),
        "fullrank_control_d128_cd": float(control[control["ambient_dim"] == 128]["gaussian_replica_cd"].mean()),
        "fullrank_control_d128_mmd": float(control[control["ambient_dim"] == 128]["gaussian_replica_mmd"].mean()),
    }
    pd.DataFrame([report]).to_csv(OUT_DIR / "highdim_manifold_paper_summary.csv", index=False)
    print("Wrote high-dimensional manifold audit to", OUT_DIR)
    for key, value in report.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
