from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from run_conditional_flow_projection_baseline import (
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
from run_transformer_flow_projection_audit import (
    ConditionalTransformerVectorField,
    TransformerFlowConfig,
    _device,
    train_transformer_flow,
)
from m2sbench.research.metrics import recovery_metrics
from m2sbench.research.utils import REPORTS_DIR, SUMMARIES_DIR, ensure_dir


def _one_hot(values: pd.Series, labels: list[str]) -> np.ndarray:
    index = {label: pos for pos, label in enumerate(labels)}
    out = np.zeros((len(values), len(labels)), dtype=np.float32)
    for row, value in enumerate(values.astype(str)):
        out[row, index[value]] = 1.0
    return out


def _contextual_xy(train_rows: pd.DataFrame, eval_rows: pd.DataFrame, config: TransformerFlowConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base_train, x0, x1 = _make_xy(train_rows, config)
    labels = sorted(set(train_rows["phase_name"].astype(str)) | set(eval_rows["phase_name"].astype(str)))
    train_context = _one_hot(train_rows["phase_name"], labels)
    eval_context = _one_hot(eval_rows["phase_name"], labels)
    return np.hstack([base_train, train_context]).astype(np.float32), x0, x1, eval_context, labels


@torch.no_grad()
def _generate_contextual(
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
    starts = [
        _gaussian_feasible(case, sample_seed + k).reshape(-1).astype(np.float32)
        for k in range(config.samples)
    ]
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


def _evaluate_contextual(
    model: ConditionalTransformerVectorField,
    eval_subset: pd.DataFrame,
    eval_context: np.ndarray,
    config: TransformerFlowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_subset.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        started = time.perf_counter()
        pred, proxy_score = _generate_contextual(model, case, eval_context[idx], config, config.model_seed + idx * 7919)
        elapsed = time.perf_counter() - started
        structural = recovery_metrics(pred, case.canonical_target, case.summary_stats)
        record: dict[str, Any] = {
            "probe_cell_id": row.probe_cell_id,
            "budget": row.budget,
            "family": row.family,
            "seed": int(row.seed),
            "phase_name": row.phase_name,
            "method": "Contextual-prior Transformer flow + exact projection",
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


def _summary_rows(results: pd.DataFrame, train_rows: pd.DataFrame, eval_subset: pd.DataFrame, diagnostics: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row": "Context prior + transf. flow + proj.",
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
                "context_labels": "; ".join(labels),
                "scope": (
                    f"{len(train_rows)} held-in phase-map cells train; {len(eval_subset)} FM-probe cells eval; "
                    "summary plus coarse diagnostic-regime context; exact moment projection"
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
            "status": "Contextual budget pilot",
            "norm_cd": row["norm_cd"],
            "iou": row["iou"],
            "violation": row["violation"],
        }
    )
    for path, label in [
        (SUMMARIES_DIR / "transformer_flow_probe" / "transformer_flow_projection_summary.csv", "Transformer flow + proj."),
        (SUMMARIES_DIR / "conditional_flow_projection_summary.csv", "Conditional flow + proj."),
        (SUMMARIES_DIR / "gaussian_projection_baseline_summary.csv", "Gaussian + projection"),
    ]:
        if path.exists():
            base = pd.read_csv(path).iloc[0]
            rows.append(
                {
                    "model_class": label,
                    "status": "Existing zero-prior/projection comparison",
                    "norm_cd": float(base["norm_cd"]),
                    "iou": float(base["iou"]),
                    "violation": float(base["violation"]),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "contextual_prior_flow_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a contextual-prior Transformer-flow audit for side-information budgets.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--cells-per-phase", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260426)
    parser.add_argument("--model-seed", type=int, default=20260522)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=SUMMARIES_DIR / "contextual_prior_flow_probe")
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
    cond, x0, x1, eval_context, labels = _contextual_xy(train_rows, eval_subset, config)
    _ = _device(config.device)
    model, diagnostics = train_transformer_flow(cond, x0, x1, config)
    results = _evaluate_contextual(model, eval_subset, eval_context, config)
    results.to_csv(out_dir / "contextual_prior_flow_results.csv", index=False)
    eval_subset.to_csv(out_dir / "contextual_prior_flow_cells.csv", index=False)
    summary = _summary_rows(results, train_rows, eval_subset, diagnostics, labels)
    summary.to_csv(out_dir / "contextual_prior_flow_summary.csv", index=False)
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
    by_phase.to_csv(out_dir / "contextual_prior_flow_by_phase.csv", index=False)
    comparison = _comparison(summary, out_dir)
    manifest = {
        "config": asdict(config),
        "diagnostics": diagnostics,
        "context_budget": "one-hot coarse diagnostic-regime label, used as a stand-in for historical routing/cache priors",
        "train_cells": int(len(train_rows)),
        "eval_cells": int(len(eval_subset)),
        "visibility": (
            "target remains evaluator-only; this pilot adds a method-visible context budget to the released summary, "
            "simulating side information such as historical routing logs"
        ),
        "labels": labels,
        "outputs": {
            "cells": "summaries/contextual_prior_flow_probe/contextual_prior_flow_cells.csv",
            "results": "summaries/contextual_prior_flow_probe/contextual_prior_flow_results.csv",
            "summary": "summaries/contextual_prior_flow_probe/contextual_prior_flow_summary.csv",
            "by_phase": "summaries/contextual_prior_flow_probe/contextual_prior_flow_by_phase.csv",
            "comparison": "summaries/contextual_prior_flow_probe/contextual_prior_flow_comparison.csv",
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    (REPORTS_DIR / "contextual_prior_flow_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
