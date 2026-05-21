from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_conditional_flow_projection_baseline import (  # noqa: E402
    FlowProjectionConfig,
    _budget_errors,
    _gaussian_feasible,
    _load_eval_subset,
    _make_case_from_row,
    _shape_proxy_score,
    exact_moment_project,
)
from m2sbench.research.metrics import recovery_metrics  # noqa: E402
from m2sbench.research.utils import REPORTS_DIR, SUMMARIES_DIR, ensure_dir  # noqa: E402


@dataclass(slots=True)
class GmmProjectionConfig(FlowProjectionConfig):
    components: int = 3
    restarts_per_alpha: int = 16
    alpha_grid: tuple[float, ...] = (0.15, 0.35, 0.55)
    model_seed: int = 20260523


def _population_cov(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(points), 1)


def _affine_match(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    src_cov = _population_cov(centered)
    src_vals, src_vecs = np.linalg.eigh(src_cov)
    tgt_vals, tgt_vecs = np.linalg.eigh(cov)
    whiten = src_vecs @ np.diag(1.0 / np.sqrt(np.maximum(src_vals, 1e-12))) @ src_vecs.T
    color = tgt_vecs @ np.diag(np.sqrt(np.maximum(tgt_vals, 0.0))) @ tgt_vecs.T
    out = centered @ whiten.T @ color.T
    return out - out.mean(axis=0, keepdims=True) + mean


def _component_centers(mean: np.ndarray, cov: np.ndarray, components: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(components, len(mean)))
    return _affine_match(raw, mean, cov)


def _sample_gmm_prior(case: Any, config: GmmProjectionConfig, seed: int, alpha: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.asarray(case.summary_stats["mean"], dtype=float)
    cov = np.asarray(case.summary_stats["covariance"], dtype=float)
    center_cov = (1.0 - alpha) * cov
    within_cov = alpha * cov
    centers = _component_centers(mean, center_cov, config.components, seed + 17)
    assignments = np.arange(case.n_points) % config.components
    rng.shuffle(assignments)
    raw = np.empty((case.n_points, len(mean)), dtype=float)
    jitter = 1e-8 * np.eye(len(mean))
    for comp in range(config.components):
        idx = np.where(assignments == comp)[0]
        if len(idx) == 0:
            continue
        raw[idx] = rng.multivariate_normal(centers[comp], within_cov + jitter, size=len(idx))
    return exact_moment_project(raw, mean, cov, seed=seed + 101)


def _best_gmm_candidate(case: Any, config: GmmProjectionConfig, cell_index: int) -> tuple[np.ndarray, float, float]:
    best: np.ndarray | None = None
    best_score = math.inf
    best_alpha = math.nan
    trial = 0
    for alpha in config.alpha_grid:
        for restart in range(config.restarts_per_alpha):
            seed = config.model_seed + cell_index * 10007 + trial * 503 + restart
            try:
                candidate = _sample_gmm_prior(case, config, seed, alpha)
            except np.linalg.LinAlgError:
                candidate = _gaussian_feasible(case, seed)
            score = _shape_proxy_score(candidate)
            if score < best_score:
                best = candidate
                best_score = float(score)
                best_alpha = float(alpha)
            trial += 1
    if best is None:
        best = _gaussian_feasible(case, config.model_seed + cell_index)
        best_score = float(_shape_proxy_score(best))
    return best, best_score, best_alpha


def run(config: GmmProjectionConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(out_dir)
    eval_subset = _load_eval_subset(config, out_dir)
    legacy_cells_path = out_dir / "conditional_flow_projection_cells.csv"
    if legacy_cells_path.exists():
        legacy_cells_path.unlink()
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_subset.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        started = time.perf_counter()
        pred, proxy_score, alpha = _best_gmm_candidate(case, config, idx)
        elapsed = time.perf_counter() - started
        structural = recovery_metrics(pred, case.canonical_target, case.summary_stats)
        record: dict[str, Any] = {
            "probe_cell_id": row.probe_cell_id,
            "budget": row.budget,
            "family": row.family,
            "seed": int(row.seed),
            "phase_name": row.phase_name,
            "method": "3-component GMM prior + exact projection",
            "valid_output": 1,
            "runtime_s": elapsed,
            "proxy_score": proxy_score,
            "selected_alpha": alpha,
            "norm_cd": float(structural["norm_cd"]),
            "iou": float(structural["iou"]),
            "coverage": float(structural["coverage"]),
        }
        record.update(_budget_errors(pred, case.summary_stats, config.epsilon_mean, config.epsilon_cov))
        rows.append(record)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "gmm_projection_results.csv", index=False)
    eval_subset.to_csv(out_dir / "gmm_projection_cells.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "row": "GMM + proj.",
                "components": config.components,
                "restarts": config.restarts_per_alpha * len(config.alpha_grid),
                "cells": int(len(results)),
                "valid": float(results["valid_output"].mean()),
                "violation": float(results["budget_violation"].mean()),
                "med_v": float(results["normalized_violation_severity"].median()),
                "cov": float(results["cov_fro_error"].median()),
                "norm_cd": float(results["norm_cd"].mean()),
                "iou": float(results["iou"].mean()),
                "coverage": float(results["coverage"].mean()),
                "runtime_s": float(results["runtime_s"].mean()),
                "scope": (
                    f"equal-weight {config.components}-component GMM prior; alpha grid {list(config.alpha_grid)}; "
                    "target-blind proxy selection; exact moment projection"
                ),
            }
        ]
    )
    summary.to_csv(out_dir / "gmm_projection_summary.csv", index=False)
    by_phase = (
        results.groupby("phase_name", dropna=False)
        .agg(
            cells=("probe_cell_id", "size"),
            violation=("budget_violation", "mean"),
            norm_cd=("norm_cd", "mean"),
            iou=("iou", "mean"),
            coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    by_phase.to_csv(out_dir / "gmm_projection_by_phase.csv", index=False)
    comparison_rows: list[dict[str, Any]] = [
        {
            "model_class": "GMM prior + exact projection",
            "status": "Classical mixture reconstruction baseline",
            "valid": float(summary.iloc[0]["valid"]),
            "violation": float(summary.iloc[0]["violation"]),
            "norm_cd": float(summary.iloc[0]["norm_cd"]),
            "iou": float(summary.iloc[0]["iou"]),
        }
    ]
    for path, label in [
        (SUMMARIES_DIR / "gaussian_projection_baseline_summary.csv", "Gaussian + projection"),
        (SUMMARIES_DIR / "conditional_flow_projection_summary.csv", "Conditional flow + projection"),
        (SUMMARIES_DIR / "transformer_flow_probe" / "transformer_flow_projection_summary.csv", "Transformer flow + projection"),
    ]:
        if path.exists():
            baseline = pd.read_csv(path).iloc[0]
            comparison_rows.append(
                {
                    "model_class": label,
                    "status": "Existing shared-probe baseline",
                    "valid": float(baseline["valid"]),
                    "violation": float(baseline["violation"]),
                    "norm_cd": float(baseline["norm_cd"]),
                    "iou": float(baseline["iou"]),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "gmm_projection_comparison.csv", index=False)
    manifest = {
        "config": asdict(config),
        "train_cells": 0,
        "eval_cells": int(len(results)),
        "visibility": "summary-only classical GMM prior; hidden target evaluator-only; exact moment projection enforces feasibility",
        "outputs": {
            "cells": "summaries/gmm_projection_probe/gmm_projection_cells.csv",
            "results": "summaries/gmm_projection_probe/gmm_projection_results.csv",
            "summary": "summaries/gmm_projection_probe/gmm_projection_summary.csv",
            "by_phase": "summaries/gmm_projection_probe/gmm_projection_by_phase.csv",
            "comparison": "summaries/gmm_projection_probe/gmm_projection_comparison.csv",
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    (REPORTS_DIR / "gmm_projection_baseline_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a classical GMM-prior plus exact moment projection baseline.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--cells-per-phase", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260426)
    parser.add_argument("--model-seed", type=int, default=20260523)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--restarts-per-alpha", type=int, default=16)
    parser.add_argument("--alpha-grid", default="0.15,0.35,0.55")
    parser.add_argument("--out-dir", type=Path, default=SUMMARIES_DIR / "gmm_projection_probe")
    args = parser.parse_args()
    alpha_grid = tuple(float(x.strip()) for x in args.alpha_grid.split(",") if x.strip())
    config = GmmProjectionConfig(
        n_points=args.n_points,
        cells_per_phase=args.cells_per_phase,
        sample_seed=args.sample_seed,
        model_seed=args.model_seed,
        components=args.components,
        restarts_per_alpha=args.restarts_per_alpha,
        alpha_grid=alpha_grid,
    )
    ensure_dir(SUMMARIES_DIR)
    ensure_dir(REPORTS_DIR)
    summary, comparison = run(config, args.out_dir)
    print(summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
