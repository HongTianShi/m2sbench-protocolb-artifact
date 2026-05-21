from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_conditional_flow_projection_baseline import (  # noqa: E402
    _budget_errors,
    _build_training_frame,
    _gaussian_feasible,
    _load_eval_subset,
    _make_case_from_row,
    _make_xy,
    _shape_proxy_score,
    _summary_cond,
    exact_moment_project,
)
from run_transformer_flow_projection_audit import (  # noqa: E402
    ConditionalTransformerVectorField,
    TransformerFlowConfig,
    _device,
    train_transformer_flow,
)
from m2sbench.research.metrics import recovery_metrics  # noqa: E402
from m2sbench.research.utils import REPORTS_DIR, SUMMARIES_DIR, ensure_dir  # noqa: E402


HISTORY_COLUMNS = [
    "phase_x",
    "phase_y",
    "access_value_gain",
    "early_warning_risk",
    "density_entropy",
    "collapse_score",
    "downstream_fragility",
    "need_escalation",
    "hypothesis_diversity",
    "posterior_concentration_proxy",
    "blind_success",
]


def _population_cov(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(points), 1)


def _local_density_variance(points: np.ndarray, k: int = 5) -> tuple[float, float]:
    pts = np.asarray(points, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    kth = np.partition(dist, kth=min(k, len(points) - 1), axis=1)[:, min(k, len(points) - 1)]
    finite = kth[np.isfinite(kth)]
    if finite.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(finite))
    var = float(np.var(finite) / max(mean * mean, 1e-8))
    return mean, var


def _summary_log_proxy(case: Any, seed: int) -> np.ndarray:
    cov = np.asarray(case.summary_stats["covariance"], dtype=float)
    eig = np.linalg.eigvalsh(cov)
    spectral_ratio = math.log1p(float(np.max(eig) / max(float(np.min(eig)), 1e-8)))
    log_n = math.log1p(float(case.n_points))
    candidates = [_gaussian_feasible(case, seed + 193 * i) for i in range(4)]
    density_means: list[float] = []
    density_vars: list[float] = []
    cov_spreads: list[float] = []
    for candidate in candidates:
        mean_k, var_k = _local_density_variance(candidate)
        density_means.append(mean_k)
        density_vars.append(var_k)
        c = _population_cov(candidate)
        vals = np.linalg.eigvalsh(c)
        cov_spreads.append(float(np.max(vals) - np.min(vals)))
    return np.array(
        [
            spectral_ratio,
            log_n,
            float(np.mean(density_means)),
            float(np.mean(density_vars)),
            float(np.std(density_vars)),
            float(np.mean(cov_spreads)),
        ],
        dtype=np.float32,
    )


def _case_base_and_proxy(rows: pd.DataFrame, config: TransformerFlowConfig, seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
    base_rows: list[np.ndarray] = []
    proxy_rows: list[np.ndarray] = []
    for idx, row in enumerate(rows.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        base_rows.append(_summary_cond(case))
        proxy_rows.append(_summary_log_proxy(case, config.model_seed + seed_offset + idx * 997))
    return np.vstack(base_rows).astype(np.float32), np.vstack(proxy_rows).astype(np.float32)


def _standardize(train: np.ndarray, eval_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train - mean) / std).astype(np.float32), ((eval_ - mean) / std).astype(np.float32), mean, std


def _history_targets(train_rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [col for col in HISTORY_COLUMNS if col in train_rows.columns]
    if not columns:
        return np.zeros((len(train_rows), 0), dtype=np.float32), []
    frame = train_rows[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    return frame.to_numpy(dtype=np.float32), columns


def _knn_history_context(
    train_base: np.ndarray,
    eval_base: np.ndarray,
    train_targets: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_targets.size == 0:
        return np.zeros((len(train_base), 0), dtype=np.float32), np.zeros((len(eval_base), 0), dtype=np.float32)
    base_train_z, base_eval_z, _, _ = _standardize(train_base, eval_base)
    k = max(2, min(k, len(train_base) - 1))

    def weighted_context(query: np.ndarray, exclude: int | None) -> np.ndarray:
        diff = base_train_z - query[None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        if exclude is not None:
            dist[exclude] = np.inf
        idx = np.argpartition(dist, kth=k - 1)[:k]
        finite = np.isfinite(dist[idx])
        idx = idx[finite]
        if idx.size == 0:
            vals = train_targets.mean(axis=0)
            stds = train_targets.std(axis=0)
        else:
            local_dist = dist[idx]
            scale = float(np.median(local_dist[np.isfinite(local_dist)])) if np.any(np.isfinite(local_dist)) else 1.0
            weights = np.exp(-local_dist / max(scale, 1e-6))
            weights = weights / max(float(weights.sum()), 1e-8)
            vals = weights @ train_targets[idx]
            centered = train_targets[idx] - vals[None, :]
            stds = np.sqrt(weights @ (centered * centered))
        return np.concatenate([vals, stds]).astype(np.float32)

    train_context = np.vstack([weighted_context(base_train_z[i], i) for i in range(len(base_train_z))])
    eval_context = np.vstack([weighted_context(base_eval_z[i], None) for i in range(len(base_eval_z))])
    return train_context.astype(np.float32), eval_context.astype(np.float32)


def _log_proxy_xy(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    config: TransformerFlowConfig,
    history_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base_train, x0, x1 = _make_xy(train_rows, config)
    base_train2, train_summary_proxy = _case_base_and_proxy(train_rows, config, seed_offset=11)
    base_eval, eval_summary_proxy = _case_base_and_proxy(eval_rows, config, seed_offset=29)
    # Use the _make_xy base features for exact alignment with the trained zero-prior row.
    if base_train.shape == base_train2.shape:
        base_train_for_history = base_train
    else:
        base_train_for_history = base_train2
    train_targets, history_cols = _history_targets(train_rows)
    train_history, eval_history = _knn_history_context(base_train_for_history, base_eval, train_targets, history_k)
    train_proxy = np.hstack([train_summary_proxy, train_history]).astype(np.float32)
    eval_proxy = np.hstack([eval_summary_proxy, eval_history]).astype(np.float32)
    train_proxy_z, eval_proxy_z, _, _ = _standardize(train_proxy, eval_proxy)
    labels = [
        "spectral_ratio",
        "log_count",
        "knn_density_mean",
        "knn_density_var",
        "knn_density_var_sd",
        "cov_spread",
    ] + [f"hist_mean_{col}" for col in history_cols] + [f"hist_sd_{col}" for col in history_cols]
    cond = np.hstack([base_train, train_proxy_z]).astype(np.float32)
    return cond, x0, x1, eval_proxy_z.astype(np.float32), labels


@torch.no_grad()
def _generate_log_proxy(
    model: ConditionalTransformerVectorField,
    case: Any,
    context: np.ndarray,
    config: TransformerFlowConfig,
    sample_seed: int,
) -> tuple[np.ndarray, float]:
    dev = next(model.parameters()).device
    base = np.asarray(_summary_cond(case), dtype=np.float32)
    cond_np = np.concatenate([base, context.astype(np.float32)])
    cond = torch.tensor(cond_np, dtype=torch.float32, device=dev).repeat(config.samples, 1)
    starts = [_gaussian_feasible(case, sample_seed + k).reshape(-1).astype(np.float32) for k in range(config.samples)]
    x = torch.tensor(np.vstack(starts), dtype=torch.float32, device=dev)
    dt = 1.0 / max(config.ode_steps, 1)
    model.eval()
    for step in range(config.ode_steps):
        t = torch.full((config.samples, 1), (step + 0.5) * dt, dtype=torch.float32, device=dev)
        x = x + dt * model(x, cond, t)
    raw = x.detach().cpu().numpy().reshape(config.samples, config.n_points, 2)
    candidates = [
        exact_moment_project(sample, case.summary_stats["mean"], case.summary_stats["covariance"], seed=sample_seed + idx)
        for idx, sample in enumerate(raw)
    ]
    scores = [_shape_proxy_score(candidate) for candidate in candidates]
    best_idx = int(np.argmin(scores))
    return candidates[best_idx], float(scores[best_idx])


def _evaluate_log_proxy(
    model: ConditionalTransformerVectorField,
    eval_subset: pd.DataFrame,
    eval_context: np.ndarray,
    config: TransformerFlowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_subset.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        started = time.perf_counter()
        pred, proxy_score = _generate_log_proxy(model, case, eval_context[idx], config, config.model_seed + idx * 7919)
        elapsed = time.perf_counter() - started
        structural = recovery_metrics(pred, case.canonical_target, case.summary_stats)
        record: dict[str, Any] = {
            "probe_cell_id": row.probe_cell_id,
            "budget": row.budget,
            "family": row.family,
            "seed": int(row.seed),
            "phase_name": row.phase_name,
            "method": "Log-proxy Transformer flow + exact projection",
            "valid_output": 1,
            "runtime_s": elapsed,
            "proxy_score": proxy_score,
            "norm_cd": float(structural["norm_cd"]),
            "iou": float(structural["iou"]),
            "coverage": float(structural["coverage"]),
        }
        record.update(_budget_errors(pred, case.summary_stats, config.epsilon_mean, config.epsilon_cov))
        rows.append(record)
    return pd.DataFrame(rows)


def _summary(results: pd.DataFrame, train_rows: pd.DataFrame, eval_subset: pd.DataFrame, diagnostics: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row": "Log-proxy context + transf. flow + proj.",
                "train_cells": int(len(train_rows)),
                "cells": int(len(eval_subset)),
                "valid": float(results["valid_output"].mean()),
                "violation": float(results["budget_violation"].mean()),
                "med_v": float(results["normalized_violation_severity"].median()),
                "cov": float(results["cov_fro_error"].median()),
                "norm_cd": float(results["norm_cd"].mean()),
                "iou": float(results["iou"].mean()),
                "coverage": float(results["coverage"].mean()),
                "runtime_s": float(results["runtime_s"].mean()),
                "train_seconds": float(diagnostics["train_seconds"]),
                "context_dim": int(len(labels)),
                "scope": (
                    f"{len(train_rows)} held-in phase-map cells train; {len(eval_subset)} FM-probe cells eval; "
                    "summary plus continuous log-derived proxy context; exact moment projection"
                ),
            }
        ]
    )


def _comparison(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row = summary.iloc[0]
    rows.append(
        {
            "model_class": row["row"],
            "status": "Continuous log-derived context budget pilot",
            "norm_cd": float(row["norm_cd"]),
            "iou": float(row["iou"]),
            "violation": float(row["violation"]),
        }
    )
    for path, label, status in [
        (SUMMARIES_DIR / "contextual_prior_flow_probe" / "contextual_prior_flow_summary.csv", "One-hot context + transf. flow + proj.", "Existing one-hot context pilot"),
        (SUMMARIES_DIR / "transformer_flow_probe" / "transformer_flow_projection_summary.csv", "Transformer flow + proj.", "Existing zero-prior comparison"),
        (SUMMARIES_DIR / "conditional_flow_projection_summary.csv", "Conditional flow + proj.", "Existing trained flow"),
        (SUMMARIES_DIR / "gaussian_projection_baseline_summary.csv", "Gaussian + projection", "Existing control"),
    ]:
        if path.exists():
            base = pd.read_csv(path).iloc[0]
            rows.append(
                {
                    "model_class": label,
                    "status": status,
                    "norm_cd": float(base["norm_cd"]),
                    "iou": float(base["iou"]),
                    "violation": float(base["violation"]),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "log_proxy_context_flow_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a log-derived contextual-prior Transformer-flow audit.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--cells-per-phase", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260426)
    parser.add_argument("--model-seed", type=int, default=20260525)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--noise-jitter", type=float, default=0.02)
    parser.add_argument("--history-k", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=SUMMARIES_DIR / "log_proxy_context_flow_probe")
    args = parser.parse_args()

    config = TransformerFlowConfig(
        n_points=args.n_points,
        cells_per_phase=args.cells_per_phase,
        sample_seed=args.sample_seed,
        model_seed=args.model_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        lr=args.lr,
        ode_steps=args.ode_steps,
        samples=args.samples,
        noise_jitter=args.noise_jitter,
        device=args.device,
    )
    ensure_dir(SUMMARIES_DIR)
    ensure_dir(REPORTS_DIR)
    out_dir = ensure_dir(args.out_dir)
    eval_subset = _load_eval_subset(config, out_dir)
    legacy_cells_path = out_dir / "conditional_flow_projection_cells.csv"
    if legacy_cells_path.exists():
        legacy_cells_path.unlink()
    train_cfg = TransformerFlowConfig(
        n_points=config.n_points,
        cells_per_phase=config.cells_per_phase,
        sample_seed=config.sample_seed,
        model_seed=config.model_seed,
    )
    train_rows = _build_training_frame(eval_subset, train_cfg)
    cond, x0, x1, eval_context, labels = _log_proxy_xy(train_rows, eval_subset, config, args.history_k)
    _ = _device(config.device)
    model, diagnostics = train_transformer_flow(cond, x0, x1, config)
    results = _evaluate_log_proxy(model, eval_subset, eval_context, config)
    results.to_csv(out_dir / "log_proxy_context_flow_results.csv", index=False)
    eval_subset.to_csv(out_dir / "log_proxy_context_flow_cells.csv", index=False)
    summary = _summary(results, train_rows, eval_subset, diagnostics, labels)
    summary.to_csv(out_dir / "log_proxy_context_flow_summary.csv", index=False)
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
    by_phase.to_csv(out_dir / "log_proxy_context_flow_by_phase.csv", index=False)
    pd.DataFrame({"context_feature": labels}).to_csv(out_dir / "log_proxy_context_features.csv", index=False)
    comparison = _comparison(summary, out_dir)
    manifest = {
        "config": asdict(config),
        "history_k": int(args.history_k),
        "diagnostics": diagnostics,
        "context_budget": (
            "continuous log-derived proxy features: summary-computable spectral/count/density proxies plus "
            "nearest-neighbor historical aggregates from held-in cells, simulating query-log/cache priors"
        ),
        "target_leakage_guard": (
            "eval-cell hidden targets are not used to construct context; historical aggregates use only held-in cells, "
            "and summary-density proxies are computed from released moments"
        ),
        "train_cells": int(len(train_rows)),
        "eval_cells": int(len(eval_subset)),
        "context_features": labels,
        "outputs": {
            "cells": "summaries/log_proxy_context_flow_probe/log_proxy_context_flow_cells.csv",
            "results": "summaries/log_proxy_context_flow_probe/log_proxy_context_flow_results.csv",
            "summary": "summaries/log_proxy_context_flow_probe/log_proxy_context_flow_summary.csv",
            "by_phase": "summaries/log_proxy_context_flow_probe/log_proxy_context_flow_by_phase.csv",
            "comparison": "summaries/log_proxy_context_flow_probe/log_proxy_context_flow_comparison.csv",
            "features": "summaries/log_proxy_context_flow_probe/log_proxy_context_features.csv",
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    (REPORTS_DIR / "log_proxy_context_flow_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
