from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_conditional_flow_projection_baseline import (
    FlowProjectionConfig,
    TimeEmbedding,
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
from m2sbench.research.metrics import recovery_metrics
from m2sbench.research.utils import REPORTS_DIR, SUMMARIES_DIR, ensure_dir


@dataclass(slots=True)
class TransformerFlowConfig(FlowProjectionConfig):
    epochs: int = 240
    batch_size: int = 64
    hidden_dim: int = 128
    depth: int = 2
    heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.02
    lr: float = 7e-4
    weight_decay: float = 2e-5
    ode_steps: int = 32
    samples: int = 8
    model_seed: int = 20260520


class ConditionalTransformerVectorField(nn.Module):
    def __init__(
        self,
        n_points: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.time = TimeEmbedding(32)
        self.point_in = nn.Linear(2, hidden_dim)
        self.cond_in = nn.Sequential(
            nn.Linear(cond_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos = nn.Parameter(torch.zeros(1, n_points, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pts = x.reshape(bsz, self.n_points, 2)
        context = self.cond_in(torch.cat([cond, self.time(t)], dim=-1))
        tokens = self.point_in(pts) + self.pos + context[:, None, :]
        encoded = self.encoder(tokens)
        return self.out(encoded).reshape(bsz, self.n_points * 2)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_transformer_flow(
    cond: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    config: TransformerFlowConfig,
) -> tuple[ConditionalTransformerVectorField, dict[str, Any]]:
    torch.manual_seed(config.model_seed)
    np.random.seed(config.model_seed)
    dev = _device(config.device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = ConditionalTransformerVectorField(
        n_points=config.n_points,
        cond_dim=cond.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        heads=config.heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    ).to(dev)
    dataset = TensorDataset(
        torch.tensor(cond, dtype=torch.float32),
        torch.tensor(x0, dtype=torch.float32),
        torch.tensor(x1, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(config.epochs):
        batch_losses: list[float] = []
        for cond_b, x0_b, x1_b in loader:
            cond_b = cond_b.to(dev)
            x0_b = x0_b.to(dev)
            x1_b = x1_b.to(dev)
            if config.noise_jitter > 0:
                x0_b = x0_b + config.noise_jitter * torch.randn_like(x0_b)
            t = torch.rand(len(cond_b), 1, device=dev)
            xt = (1.0 - t) * x0_b + t * x1_b
            target_v = x1_b - x0_b
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_v = model(xt, cond_b, t)
                loss = torch.mean((pred_v - target_v) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            batch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(batch_losses)))
    elapsed = time.perf_counter() - started
    return model, {
        "device": str(dev),
        "epochs": config.epochs,
        "train_seconds": elapsed,
        "final_loss": losses[-1] if losses else math.nan,
        "loss_trace": losses[:: max(1, len(losses) // 20)] + (losses[-1:] if losses else []),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def generate_transformer_flow(
    model: ConditionalTransformerVectorField,
    case: Any,
    config: TransformerFlowConfig,
    sample_seed: int,
) -> tuple[np.ndarray, float]:
    dev = next(model.parameters()).device
    cond = torch.tensor(_summary_cond(case), dtype=torch.float32, device=dev).repeat(config.samples, 1)
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


def evaluate_transformer_flow(
    model: ConditionalTransformerVectorField,
    eval_subset: pd.DataFrame,
    config: TransformerFlowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_subset.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        started = time.perf_counter()
        pred, proxy_score = generate_transformer_flow(model, case, config, config.model_seed + idx * 7919)
        elapsed = time.perf_counter() - started
        structural = recovery_metrics(pred, case.canonical_target, case.summary_stats)
        record: dict[str, Any] = {
            "probe_cell_id": row.probe_cell_id,
            "budget": row.budget,
            "family": row.family,
            "seed": int(row.seed),
            "phase_name": row.phase_name,
            "method": "Transformer rectified flow + exact projection",
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


def _baseline_comparison(summary: pd.DataFrame, eval_subset: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row = summary.iloc[0]
    rows.append(
        {
            "model_class": "Transformer rectified flow + exact projection",
            "status": "Sequence-aware constraint-aware audit",
            "valid": row["valid"],
            "violation": row["violation"],
            "norm_cd": row["norm_cd"],
            "iou": row["iou"],
            "boundary": "Stronger summary-only rectified-flow vector field followed by exact projection.",
        }
    )
    for path, label, status in [
        (SUMMARIES_DIR / "conditional_flow_projection_summary.csv", "Conditional flow + exact projection", "Existing trained flow"),
        (SUMMARIES_DIR / "neural_projection_baseline_summary.csv", "Neural proposal + exact projection", "Existing lightweight baseline"),
        (SUMMARIES_DIR / "gaussian_projection_baseline_summary.csv", "Gaussian proposal + projection", "Existing control"),
    ]:
        if path.exists():
            baseline = pd.read_csv(path).iloc[0]
            rows.append(
                {
                    "model_class": label,
                    "status": status,
                    "valid": baseline["valid"],
                    "violation": baseline["violation"],
                    "norm_cd": baseline["norm_cd"],
                    "iou": baseline["iou"],
                    "boundary": "Existing Table 3 projection/proposal comparison.",
                }
            )
    stress_path = SUMMARIES_DIR / "track1_blind_stress_runs.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        stress = stress[(stress["suite"] == "challenge") & (stress["protocol"] == "protocol_b")]
        keys = eval_subset[["budget", "family", "seed"]].drop_duplicates()
        baseline = stress.merge(keys, on=["budget", "family", "seed"], how="inner")
        for method_key, label in [("evolutionary-blind", "Evolutionary blind"), ("blind-sa", "Blind-SA")]:
            group = baseline[baseline["method"] == method_key]
            rows.append(
                {
                    "model_class": label,
                    "status": "Search baseline",
                    "valid": 1.0,
                    "violation": 0.0,
                    "norm_cd": float(group["norm_cd"].mean()) if len(group) else math.nan,
                    "iou": float(group["iou"].mean()) if len(group) else math.nan,
                    "boundary": "Search-based Protocol B recovery remains stronger on this subset.",
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "transformer_flow_projection_comparison.csv", index=False)
    return comparison


def _write_summary(
    results: pd.DataFrame,
    train_rows: pd.DataFrame,
    eval_subset: pd.DataFrame,
    diagnostics: dict[str, Any],
    out_dir: Path,
    config: TransformerFlowConfig,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "row": "Transformer flow + proj.",
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
                "scope": (
                    f"{len(train_rows)} held-in phase-map cells train; {len(eval_subset)} FM-probe cells eval; "
                    "summary-only transformer rectified-flow vector field; exact moment projection"
                ),
            }
        ]
    )
    summary.to_csv(out_dir / "transformer_flow_projection_summary.csv", index=False)
    by_phase = (
        results.groupby("phase_name", dropna=False)
        .agg(
            cells=("probe_cell_id", "size"),
            violation=("budget_violation", "mean"),
            med_v=("normalized_violation_severity", "median"),
            norm_cd=("norm_cd", "mean"),
            iou=("iou", "mean"),
            coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    by_phase.to_csv(out_dir / "transformer_flow_projection_by_phase.csv", index=False)
    comparison = _baseline_comparison(summary, eval_subset, out_dir)
    manifest = {
        "config": asdict(config),
        "diagnostics": diagnostics,
        "train_cells": int(len(train_rows)),
        "eval_cells": int(len(eval_subset)),
        "visibility": (
            "test-time summary-only: held-out cells expose only released moments and N to the transformer-flow proposal; "
            "held-in training targets are supervised labels; held-out hidden targets are used only for scoring"
        ),
        "outputs": {
            "cells": str(out_dir / "transformer_flow_projection_cells.csv"),
            "results": str(out_dir / "transformer_flow_projection_results.csv"),
            "summary": str(out_dir / "transformer_flow_projection_summary.csv"),
            "by_phase": str(out_dir / "transformer_flow_projection_by_phase.csv"),
            "comparison": str(out_dir / "transformer_flow_projection_comparison.csv"),
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    (REPORTS_DIR / "transformer_flow_projection_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a transformer rectified-flow proposal plus exact moment projection audit.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--cells-per-phase", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260426)
    parser.add_argument("--model-seed", type=int, default=20260520)
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
    parser.add_argument("--out-dir", type=Path, default=SUMMARIES_DIR / "transformer_flow_probe")
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
    train_cfg = FlowProjectionConfig(
        n_points=config.n_points,
        cells_per_phase=config.cells_per_phase,
        sample_seed=config.sample_seed,
        model_seed=config.model_seed,
    )
    train_rows = _build_training_frame(eval_subset, train_cfg)
    cond, x0, x1 = _make_xy(train_rows, config)
    model, diagnostics = train_transformer_flow(cond, x0, x1, config)
    results = evaluate_transformer_flow(model, eval_subset, config)
    results.to_csv(out_dir / "transformer_flow_projection_results.csv", index=False)
    eval_subset.to_csv(out_dir / "transformer_flow_projection_cells.csv", index=False)
    summary = _write_summary(results, train_rows, eval_subset, diagnostics, out_dir, config)
    print(summary.to_string(index=False))
    print()
    comparison = pd.read_csv(out_dir / "transformer_flow_projection_comparison.csv")
    print(comparison[["model_class", "status", "norm_cd", "iou", "violation"]].to_string(index=False))


if __name__ == "__main__":
    main()
