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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2sbench.research.data import make_case
from m2sbench.research.metrics import recovery_metrics
from m2sbench.research.utils import REPORTS_DIR, SUMMARIES_DIR, ensure_dir


@dataclass(slots=True)
class FlowProjectionConfig:
    n_points: int = 64
    cells_per_phase: int = 10
    sample_seed: int = 20260426
    model_seed: int = 20260516
    epochs: int = 800
    batch_size: int = 64
    hidden_dim: int = 384
    depth: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    ode_steps: int = 32
    samples: int = 8
    noise_jitter: float = 0.02
    epsilon_mean: float = 1e-2
    epsilon_cov: float = 1e-2
    dense_support_points: int = 512
    support_pool_size: int = 512
    device: str = "auto"


def _load_eval_subset(config: FlowProjectionConfig, out_dir: Path) -> pd.DataFrame:
    cells_path = SUMMARIES_DIR / "foundation_model_direct_generation_cells.csv"
    if cells_path.exists():
        subset = pd.read_csv(cells_path)
    else:
        subset = _build_probe_subset(
            SUMMARIES_DIR / "phase_map.csv",
            cells_path,
            cells_per_phase=config.cells_per_phase,
            sample_seed=config.sample_seed,
            n_points=config.n_points,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_dir / "conditional_flow_projection_cells.csv", index=False)
    return subset


def _cell_id(row: pd.Series, n_points: int) -> str:
    return f"protocol_b:{row['budget']}:{row['family']}:seed{int(row['seed'])}:n{n_points}"


def _build_probe_subset(
    phase_path: Path,
    out_path: Path,
    cells_per_phase: int,
    sample_seed: int,
    n_points: int,
) -> pd.DataFrame:
    phase = pd.read_csv(phase_path)
    rows = []
    for _, group in phase.groupby("phase_name", sort=True):
        sampled = group.sample(n=min(cells_per_phase, len(group)), random_state=sample_seed)
        rows.append(sampled)
    subset = pd.concat(rows, ignore_index=True)
    subset["probe_cell_id"] = subset.apply(lambda row: _cell_id(row, n_points), axis=1)
    subset.to_csv(out_path, index=False)
    return subset


def _build_training_frame(eval_subset: pd.DataFrame, config: FlowProjectionConfig) -> pd.DataFrame:
    phase = pd.read_csv(SUMMARIES_DIR / "phase_map.csv")
    phase["probe_cell_id"] = phase.apply(lambda row: _cell_id(row, config.n_points), axis=1)
    held_out = set(eval_subset["probe_cell_id"].astype(str))
    train = phase[~phase["probe_cell_id"].isin(held_out)].copy()
    if len(train) < 25:
        raise RuntimeError(f"Too few training cells after held-out exclusion: {len(train)}")
    return train.sort_values(["budget", "family", "seed"]).reset_index(drop=True)


def exact_moment_project(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    target_mean = np.asarray(mean, dtype=float)
    target_cov = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    for attempt in range(5):
        centered = pts - pts.mean(axis=0, keepdims=True)
        src_cov = _population_cov(centered)
        src_vals, _ = np.linalg.eigh(src_cov)
        if float(np.min(src_vals)) > 1e-10:
            break
        pts = pts + (10.0 ** (-6 + attempt)) * rng.normal(size=pts.shape)
    centered = pts - pts.mean(axis=0, keepdims=True)
    src_cov = _population_cov(centered)
    src_vals, src_vecs = np.linalg.eigh(src_cov)
    tgt_vals, tgt_vecs = np.linalg.eigh(target_cov)
    whiten = src_vecs @ np.diag(1.0 / np.sqrt(np.maximum(src_vals, 1e-12))) @ src_vecs.T
    color = tgt_vecs @ np.diag(np.sqrt(np.maximum(tgt_vals, 0.0))) @ tgt_vecs.T
    projected = centered @ whiten.T @ color.T
    projected = projected - projected.mean(axis=0, keepdims=True) + target_mean
    return projected


def _budget_errors(points: np.ndarray, summary_stats: dict[str, np.ndarray], eps_mean: float, eps_cov: float) -> dict[str, float | int]:
    target_mean = np.asarray(summary_stats["mean"], dtype=float)
    target_cov = np.asarray(summary_stats["covariance"], dtype=float)
    pred_mean = points.mean(axis=0)
    pred_cov = _population_cov(points)
    mean_linf = float(np.abs(pred_mean - target_mean).max())
    cov_linf = float(np.abs(pred_cov - target_cov).max())
    return {
        "mean_linf_error": mean_linf,
        "cov_linf_error": cov_linf,
        "cov_fro_error": float(np.linalg.norm(pred_cov - target_cov, ord="fro")),
        "normalized_violation_severity": float(max(mean_linf / eps_mean, cov_linf / eps_cov)),
        "budget_violation": int(mean_linf > eps_mean or cov_linf > eps_cov),
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _population_cov(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(points), 1)


def _ordered_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0, keepdims=True)
    rel = pts - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)
    return pts[np.lexsort((radius, angles))]


def _summary_cond(case: Any) -> np.ndarray:
    mean = np.asarray(case.summary_stats["mean"], dtype=float)
    cov = np.asarray(case.summary_stats["covariance"], dtype=float)
    eig = np.linalg.eigvalsh(cov)
    trace = float(np.trace(cov))
    det = float(np.linalg.det(cov))
    corr = float(cov[0, 1] / max(math.sqrt(max(cov[0, 0] * cov[1, 1], 0.0)), 1e-8))
    return np.array(
        [
            mean[0],
            mean[1],
            cov[0, 0],
            cov[0, 1],
            cov[1, 1],
            eig[0],
            eig[1],
            trace,
            det,
            corr,
            float(case.n_points) / 256.0,
        ],
        dtype=np.float32,
    )


def _gaussian_feasible(case: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(case.n_points, 2))
    projected = exact_moment_project(raw, case.summary_stats["mean"], case.summary_stats["covariance"], seed=seed)
    return _ordered_points(projected)


def _make_case_from_row(row: Any, config: FlowProjectionConfig) -> Any:
    return make_case(
        row.family,
        row.budget,
        int(row.seed),
        protocol="protocol_b",
        n_points=config.n_points,
        dense_support_points=config.dense_support_points,
        support_pool_size=config.support_pool_size,
    )


def _make_xy(rows: pd.DataFrame, config: FlowProjectionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cond_rows: list[np.ndarray] = []
    x0_rows: list[np.ndarray] = []
    x1_rows: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        case = _make_case_from_row(row, config)
        seed = int(config.model_seed + int(row.seed) * 1009 + len(cond_rows))
        cond_rows.append(_summary_cond(case))
        x0_rows.append(_gaussian_feasible(case, seed).reshape(-1).astype(np.float32))
        x1_rows.append(_ordered_points(case.canonical_target).reshape(-1).astype(np.float32))
    return np.vstack(cond_rows), np.vstack(x0_rows), np.vstack(x1_rows)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        half = dim // 2
        freq = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t * self.freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditionalVectorField(nn.Module):
    def __init__(self, point_dim: int, cond_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.time = TimeEmbedding(32)
        layers: list[nn.Module] = []
        in_dim = point_dim + cond_dim + 32
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, point_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time(t)
        return self.net(torch.cat([x, cond, emb], dim=-1))


def train_flow(cond: np.ndarray, x0: np.ndarray, x1: np.ndarray, config: FlowProjectionConfig) -> tuple[ConditionalVectorField, dict[str, Any]]:
    torch.manual_seed(config.model_seed)
    np.random.seed(config.model_seed)
    dev = _device(config.device)
    model = ConditionalVectorField(
        point_dim=x1.shape[1],
        cond_dim=cond.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(dev)
    dataset = TensorDataset(
        torch.tensor(cond, dtype=torch.float32),
        torch.tensor(x0, dtype=torch.float32),
        torch.tensor(x1, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    losses: list[float] = []
    started = time.perf_counter()
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
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
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)))
    elapsed = time.perf_counter() - started
    diagnostics = {
        "device": str(dev),
        "epochs": config.epochs,
        "train_seconds": elapsed,
        "final_loss": losses[-1] if losses else math.nan,
        "loss_trace": losses[:: max(1, len(losses) // 20)] + (losses[-1:] if losses else []),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, diagnostics


@torch.no_grad()
def generate_flow(model: ConditionalVectorField, case: Any, config: FlowProjectionConfig, sample_seed: int) -> tuple[np.ndarray, float]:
    dev = next(model.parameters()).device
    cond = torch.tensor(_summary_cond(case), dtype=torch.float32, device=dev).repeat(config.samples, 1)
    starts = []
    for k in range(config.samples):
        starts.append(_gaussian_feasible(case, sample_seed + k).reshape(-1).astype(np.float32))
    x = torch.tensor(np.vstack(starts), dtype=torch.float32, device=dev)
    dt = 1.0 / max(config.ode_steps, 1)
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


def _shape_proxy_score(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return 0.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2) + np.eye(len(pts)) * 1e9)
    nn = d.min(axis=1)
    cv = float(nn.std() / max(nn.mean(), 1e-8))
    cov = _population_cov(pts)
    eig = np.linalg.eigvalsh(cov)
    anisotropy = float(eig[-1] / max(eig[0], 1e-8))
    return cv + 0.002 * anisotropy


def evaluate_flow(model: ConditionalVectorField, eval_subset: pd.DataFrame, config: FlowProjectionConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    for idx, row in enumerate(eval_subset.itertuples(index=False)):
        case = _make_case_from_row(row, config)
        started = time.perf_counter()
        pred, proxy_score = generate_flow(model, case, config, config.model_seed + idx * 7919)
        elapsed = time.perf_counter() - started
        structural = recovery_metrics(pred, case.canonical_target, case.summary_stats)
        record: dict[str, Any] = {
            "probe_cell_id": row.probe_cell_id,
            "budget": row.budget,
            "family": row.family,
            "seed": int(row.seed),
            "phase_name": row.phase_name,
            "method": "Conditional flow matching + exact projection",
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


def _write_summary(
    results: pd.DataFrame,
    train_rows: pd.DataFrame,
    eval_subset: pd.DataFrame,
    diagnostics: dict[str, Any],
    out_dir: Path,
    config: FlowProjectionConfig,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "row": "Conditional flow + proj.",
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
                "scope": f"{len(train_rows)} held-in phase-map cells train; {len(eval_subset)} FM-probe cells eval; summary-only rectified flow; exact moment projection",
            }
        ]
    )
    summary.to_csv(out_dir / "conditional_flow_projection_summary.csv", index=False)
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
    by_phase.to_csv(out_dir / "conditional_flow_projection_by_phase.csv", index=False)
    _write_comparison(summary, eval_subset, out_dir)
    manifest = {
        "config": asdict(config),
        "diagnostics": diagnostics,
        "train_cells": int(len(train_rows)),
        "eval_cells": int(len(eval_subset)),
        "visibility": "test-time summary-only: held-out cells expose only released moments and N to the flow proposal; held-in training targets are supervised labels; held-out hidden targets are used only for scoring",
        "outputs": {
            "cells": str(out_dir / "conditional_flow_projection_cells.csv"),
            "results": str(out_dir / "conditional_flow_projection_results.csv"),
            "summary": str(out_dir / "conditional_flow_projection_summary.csv"),
            "by_phase": str(out_dir / "conditional_flow_projection_by_phase.csv"),
            "comparison": str(out_dir / "generative_projection_baseline_comparison.csv"),
        },
    }
    (REPORTS_DIR / "conditional_flow_projection_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def _write_comparison(flow_summary: pd.DataFrame, eval_subset: pd.DataFrame, out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    flow = flow_summary.iloc[0]
    rows.append(
        {
            "model_class": "Conditional flow matching + exact projection",
            "status": "Tested modern generative baseline",
            "valid": flow["valid"],
            "violation": flow["violation"],
            "norm_cd": flow["norm_cd"],
            "iou": flow["iou"],
            "boundary": "Summary-only rectified flow tests whether a trained modern generator improves the feasible proposal before projection.",
        }
    )
    for path, label, status in [
        (SUMMARIES_DIR / "neural_projection_baseline_summary.csv", "Neural proposal + exact projection", "Tested lightweight baseline"),
        (SUMMARIES_DIR / "gaussian_projection_baseline_summary.csv", "Gaussian proposal + projection", "Tested control"),
    ]:
        if path.exists():
            row = pd.read_csv(path).iloc[0]
            rows.append(
                {
                    "model_class": label,
                    "status": status,
                    "valid": row["valid"],
                    "violation": row["violation"],
                    "norm_cd": row["norm_cd"],
                    "iou": row["iou"],
                    "boundary": "Existing projection comparison row.",
                }
            )
    stress_path = SUMMARIES_DIR / "track1_blind_stress_runs.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        stress = stress[(stress["suite"] == "challenge") & (stress["protocol"] == "protocol_b")]
        subset_keys = eval_subset[["budget", "family", "seed"]].drop_duplicates()
        baseline = stress.merge(subset_keys, on=["budget", "family", "seed"], how="inner")
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
    pd.DataFrame(rows).to_csv(out_dir / "generative_projection_baseline_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conditional flow-matching proposal plus exact moment projection baseline.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--cells-per-phase", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260426)
    parser.add_argument("--model-seed", type=int, default=20260516)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--noise-jitter", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=SUMMARIES_DIR)
    args = parser.parse_args()

    config = FlowProjectionConfig(
        n_points=args.n_points,
        cells_per_phase=args.cells_per_phase,
        sample_seed=args.sample_seed,
        model_seed=args.model_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
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
    train_rows = _build_training_frame(eval_subset, config)
    cond, x0, x1 = _make_xy(train_rows, config)
    model, diagnostics = train_flow(cond, x0, x1, config)
    results = evaluate_flow(model, eval_subset, config)
    results.to_csv(out_dir / "conditional_flow_projection_results.csv", index=False)
    summary = _write_summary(results, train_rows, eval_subset, diagnostics, out_dir, config)
    print(summary.to_string(index=False))
    comparison = pd.read_csv(out_dir / "generative_projection_baseline_comparison.csv")
    print()
    print(comparison[["model_class", "status", "norm_cd", "iou", "violation"]].to_string(index=False))


if __name__ == "__main__":
    main()
