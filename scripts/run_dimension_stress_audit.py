"""High-dimensional matched-summary stress audit.

The main synthetic cells are intentionally visual, but the benchmark contract is
not tied to two dimensions.  This script creates matched (mu, Sigma, N) point
sets in d = 2, 10, 50, 128, evaluates exact moment feasibility, hidden-structure
recovery, and a simple cost-aware structural-view task.

The construction is deliberately conservative: every candidate is independently
whitened to the same released summary, so first and second moments are not
available to identify the hidden structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "summaries" / "dimension_stress_audit"
FIG_DIR = ROOT / "figures"


KINDS = ("two_cluster", "four_cluster", "ring", "sparse_blocks")
VIEW_COSTS = {
    "summary": 0.00,
    "coarse_sketch": 0.28,
    "higher_order": 0.42,
    "point": 0.58,
}


@dataclass(frozen=True)
class Cell:
    dim: int
    seed: int
    target_kind: str
    target: np.ndarray
    candidates: list[tuple[str, np.ndarray]]


def _orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def _whiten_to_summary(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    y = x @ (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T)
    # One final centering removes numerical drift after the eigensolve.
    return y - y.mean(axis=0, keepdims=True)


def make_raw(kind: str, d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    x = 0.18 * rng.normal(size=(n, d))
    if kind == "two_cluster":
        labels = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
        rng.shuffle(labels)
        x[:, 0] += np.where(labels == 0, -2.0, 2.0)
        if d > 1:
            x[:, 1] += 0.35 * rng.normal(size=n)
    elif kind == "four_cluster":
        labels = np.arange(n) % 4
        rng.shuffle(labels)
        centers = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float) * 1.7
        if d == 1:
            x[:, 0] += centers[labels, 0]
        else:
            x[:, :2] += centers[labels]
    elif kind == "ring":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rng.shuffle(t)
        if d == 1:
            x[:, 0] += 2.0 * np.cos(t)
        else:
            x[:, 0] += 2.0 * np.cos(t)
            x[:, 1] += 2.0 * np.sin(t)
        if d > 3:
            x[:, 2] += 0.6 * np.cos(3 * t)
            x[:, 3] += 0.6 * np.sin(3 * t)
    elif kind == "sparse_blocks":
        blocks = min(8, d)
        labels = np.arange(n) % blocks
        rng.shuffle(labels)
        for i in range(n):
            x[i, labels[i]] += 2.2 * (1 if labels[i] % 2 == 0 else -1)
        if d > blocks:
            x[:, blocks:] += 0.12 * rng.standard_t(df=3, size=(n, d - blocks))
    else:
        raise ValueError(f"unknown kind: {kind}")

    # Randomly rotate so the dimension audit is not tied to named coordinate axes.
    return x @ _orthogonal_matrix(d, rng)


def make_structure(kind: str, d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _whiten_to_summary(make_raw(kind, d, n, rng))


def moment_errors(x: np.ndarray) -> tuple[float, float]:
    mean_err = float(np.max(np.abs(x.mean(axis=0))))
    cov = (x - x.mean(axis=0)).T @ (x - x.mean(axis=0)) / len(x)
    cov_err = float(np.max(np.abs(cov - np.eye(x.shape[1]))))
    return mean_err, cov_err


def _subsample(x: np.ndarray, max_points: int = 96) -> np.ndarray:
    if len(x) <= max_points:
        return x
    idx = np.linspace(0, len(x) - 1, max_points).round().astype(int)
    return x[idx]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    aa = _subsample(a)
    bb = _subsample(b)
    d = cdist(aa, bb, metric="sqeuclidean")
    cd = float(d.min(axis=1).mean() + d.min(axis=0).mean())
    scale = float(np.median(cdist(aa, aa, metric="sqeuclidean")))
    return cd / max(scale, 1e-9)


def rbf_mmd(a: np.ndarray, b: np.ndarray) -> float:
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


def gaussian_projection(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _whiten_to_summary(rng.normal(size=(n, d)))


def coarse_feature(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + dim * 17 + seed)
    k = min(2, dim)
    proj = rng.normal(size=(dim, k))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12
    z = x @ proj
    bins = np.array([-np.inf, -1.25, 0.0, 1.25, np.inf])
    feats = []
    for j in range(k):
        hist, _ = np.histogram(z[:, j], bins=bins, density=False)
        feats.extend((hist / max(len(z), 1)).tolist())
    return np.asarray(feats, dtype=float)


def higher_order_feature(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=0)
    s = x.std(axis=0) + 1e-9
    z = (x - m) / s
    skew = np.mean(z**3, axis=0)
    kurt = np.mean(z**4, axis=0)
    pts = _subsample(x, max_points=80)
    pair = cdist(pts, pts, metric="sqeuclidean")
    pair = pair[np.triu_indices_from(pair, k=1)]
    norm = np.linalg.norm(x, axis=1)
    # Keep feature length bounded in high dimensions.
    parts = [
        np.quantile(np.abs(skew), [0.25, 0.5, 0.75]),
        np.quantile(kurt, [0.25, 0.5, 0.75]),
        np.quantile(norm, [0.25, 0.5, 0.75]),
        np.quantile(pair, [0.25, 0.5, 0.75]),
    ]
    return np.concatenate(parts).astype(float)


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return chamfer(a, b)


def choose_candidate(cell: Cell, view: str) -> str:
    target = cell.target
    if view == "summary":
        return "two_cluster"
    if view == "coarse_sketch":
        ft = coarse_feature(target, cell.dim, cell.seed)
        scores = [
            float(np.linalg.norm(coarse_feature(cand, cell.dim, cell.seed) - ft))
            for _, cand in cell.candidates
        ]
    elif view == "higher_order":
        ft = higher_order_feature(target)
        scores = [
            float(np.linalg.norm(higher_order_feature(cand) - ft))
            for _, cand in cell.candidates
        ]
    elif view == "point":
        scores = [point_distance(target, cand) for _, cand in cell.candidates]
    else:
        raise ValueError(view)
    return cell.candidates[int(np.argmin(scores))][0]


def make_cell(dim: int, seed: int, target_kind: str, n: int) -> Cell:
    target = make_structure(target_kind, dim, n, seed)
    candidates: list[tuple[str, np.ndarray]] = [(target_kind, target)]
    alt_order = [k for k in KINDS if k != target_kind]
    # Fixed but nontrivial ordering: summary-only tie breaking is not privileged.
    shift = seed % len(alt_order)
    alt_order = alt_order[shift:] + alt_order[:shift]
    for idx, kind in enumerate(alt_order):
        candidates.append((kind, make_structure(kind, dim, n, seed + 1009 * (idx + 1))))
    if seed % 2:
        candidates = candidates[1:] + candidates[:1]
    return Cell(dim=dim, seed=seed, target_kind=target_kind, target=target, candidates=candidates)


def run_audit(dims: tuple[int, ...] = (2, 10, 50, 128), seeds_per_kind: int = 16, n: int = 160) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    cell_rows = []
    for dim in dims:
        for kind_idx, kind in enumerate(KINDS):
            for seed_idx in range(seeds_per_kind):
                seed = 20260517 + dim * 1000 + kind_idx * 101 + seed_idx
                cell = make_cell(dim, seed, kind, n)
                mean_err, cov_err = moment_errors(cell.target)
                gauss = gaussian_projection(dim, n, seed + 77_777)
                g_mean_err, g_cov_err = moment_errors(gauss)
                g_cd = chamfer(cell.target, gauss)
                g_mmd = rbf_mmd(cell.target, gauss)

                view_payload = {}
                best_view = None
                best_utility = -1e9
                for view, cost in VIEW_COSTS.items():
                    pred_kind = choose_candidate(cell, view)
                    correct = float(pred_kind == kind)
                    utility = correct - cost
                    view_payload[f"{view}_correct"] = correct
                    view_payload[f"{view}_utility"] = utility
                    if utility > best_utility:
                        best_view = view
                        best_utility = utility

                point_gain = view_payload["point_utility"] - view_payload["summary_utility"]
                rich_best = best_view == "point"
                nonmonotone = best_view != "point"
                rows.append(
                    {
                        "dim": dim,
                        "kind": kind,
                        "seed": seed,
                        "target_mean_linf": mean_err,
                        "target_cov_linf": cov_err,
                        "gaussian_mean_linf": g_mean_err,
                        "gaussian_cov_linf": g_cov_err,
                        "gaussian_norm_cd": g_cd,
                        "gaussian_mmd": g_mmd,
                        "best_view": best_view,
                        "best_utility": best_utility,
                        "point_gain_vs_summary": point_gain,
                        "richest_is_best": float(rich_best),
                        "nonmonotone_best_view": float(nonmonotone),
                        **view_payload,
                    }
                )
                cell_rows.append({"dim": dim, "kind": kind, "seed": seed, "n_points": n})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "dimension_stress_by_cell.csv", index=False)

    summary = (
        detail.groupby("dim", as_index=False)
        .agg(
            cells=("seed", "size"),
            max_target_mean_linf=("target_mean_linf", "max"),
            max_target_cov_linf=("target_cov_linf", "max"),
            max_gaussian_cov_linf=("gaussian_cov_linf", "max"),
            gaussian_norm_cd=("gaussian_norm_cd", "mean"),
            gaussian_mmd=("gaussian_mmd", "mean"),
            summary_correct=("summary_correct", "mean"),
            coarse_correct=("coarse_sketch_correct", "mean"),
            higher_order_correct=("higher_order_correct", "mean"),
            point_correct=("point_correct", "mean"),
            best_utility=("best_utility", "mean"),
            point_gain_vs_summary=("point_gain_vs_summary", "mean"),
            nonmonotone_best_view=("nonmonotone_best_view", "mean"),
            richest_is_best=("richest_is_best", "mean"),
        )
    )
    win = (
        detail.pivot_table(index="dim", columns="best_view", values="seed", aggfunc="count", fill_value=0)
        .div(detail.groupby("dim")["seed"].count(), axis=0)
        .reset_index()
    )
    summary = summary.merge(win, on="dim", how="left")
    summary.to_csv(OUT_DIR / "dimension_stress_summary.csv", index=False)

    pd.DataFrame(cell_rows).to_csv(OUT_DIR / "dimension_stress_manifest.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=180)
        axes[0].plot(summary["dim"], summary["gaussian_norm_cd"], marker="o", label="Gaussian + projection")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("ambient dimension")
        axes[0].set_ylabel("Norm. CD")
        axes[0].set_title("Feasible but structurally far")
        axes[0].grid(True, color="#e6e6e6", linewidth=0.6)
        axes[1].plot(summary["dim"], summary["nonmonotone_best_view"], marker="o", color="#2a9d8f")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("ambient dimension")
        axes[1].set_ylabel("non-richest best-view share")
        axes[1].set_title("Cost-aware view reversal persists")
        axes[1].grid(True, color="#e6e6e6", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "dimension_stress_audit.png")
        fig.savefig(FIG_DIR / "dimension_stress_audit.pdf")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is optional.
        print(f"plot skipped: {exc}")

    with (OUT_DIR / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Dimension Stress Audit\n\n"
            "Auxiliary matched-summary stress test over d = 2, 10, 50, 128. "
            "Each target and candidate is whitened to the same released summary "
            "(zero mean, identity covariance, fixed N), then evaluated for exact "
            "feasibility, hidden-structure distance, and cost-aware view selection.\n"
        )

    print(summary.to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    run_audit()
