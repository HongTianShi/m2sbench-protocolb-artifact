from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m2sbench.evaluator import split_demo_cells


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _points_from_submission(row: dict[str, Any]) -> np.ndarray | None:
    if "candidate" in row:
        return np.asarray(row["candidate"], dtype=float)
    if "points" in row:
        return np.asarray(row["points"], dtype=float)
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict) and "points" in first:
            return np.asarray(first["points"], dtype=float)
        return np.asarray(first, dtype=float)
    return None


def _moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    return pts.mean(axis=0), np.cov(pts.T, bias=True)


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))


def _norm_cd(prediction: np.ndarray, target: np.ndarray, covariance: np.ndarray) -> float:
    distances = _pairwise(prediction, target)
    norm = float(np.sqrt(np.trace(covariance)) + 1e-8)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / norm)


def _coverage(prediction: np.ndarray, target: np.ndarray) -> float:
    distances = _pairwise(prediction, target)
    target_distances = _pairwise(target, target)
    np.fill_diagonal(target_distances, np.inf)
    radius = float(np.quantile(target_distances.min(axis=1), 0.5))
    return float((distances.min(axis=0) <= radius).mean())


def _raster(points: np.ndarray, resolution: int = 64) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scaled = ((pts - lo) / span * (resolution - 1)).astype(int)
    scaled = np.clip(scaled, 0, resolution - 1)
    grid = np.zeros((resolution, resolution), dtype=bool)
    grid[scaled[:, 1], scaled[:, 0]] = True
    return grid


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ra = _raster(a)
    rb = _raster(b)
    union = np.logical_or(ra, rb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(ra, rb).sum() / union)


def _view_metrics(row: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | str]:
    ranked = row.get("ranked_views")
    if not isinstance(ranked, list):
        ranked = []
    ranked = [str(item) for item in ranked]
    oracle = str(reference.get("oracle_view", "point"))
    rank = ranked.index(oracle) + 1 if oracle in ranked else 0
    utilities = reference.get("view_utilities", {})
    route = row.get("route")
    if route == "abstain":
        selected = "abstain"
    elif isinstance(route, str) and route in utilities:
        selected = route
    else:
        selected = ranked[0] if ranked else "abstain"
    selected_utility = float(utilities.get(selected, 0.0))
    oracle_utility = float(max(utilities.values())) if utilities else 0.0
    return {
        "oracle_view": oracle,
        "selected_view": selected,
        "top1_view_correct": float(rank == 1),
        "top3_view_hit": float(0 < rank <= 3),
        "mrr": float(1.0 / rank) if rank else 0.0,
        "utility_at_1": selected_utility,
        "regret": float(oracle_utility - selected_utility),
    }


def _evaluate_cell(cell: dict[str, Any], submission: dict[str, Any] | None, reference: dict[str, Any]) -> dict[str, Any]:
    base = {
        "cell_id": cell["cell_id"],
        "method": submission.get("method", "unknown") if submission else "missing",
        "valid_output": 0.0,
        "feasible": 0.0,
        "mean_linf_error": np.nan,
        "cov_linf_error": np.nan,
        "norm_cd": np.nan,
        "iou": np.nan,
        "coverage": np.nan,
        "runtime_s": float(submission.get("runtime_s", np.nan)) if submission else np.nan,
    }
    if submission is None:
        base.update(_view_metrics({}, reference))
        return base
    try:
        points = _points_from_submission(submission)
        if points is None or points.ndim != 2 or points.shape[1] != 2:
            base.update(_view_metrics(submission, reference))
            return base
        expected_n = int(cell["n_points"])
        if len(points) != expected_n or not np.isfinite(points).all():
            base.update(_view_metrics(submission, reference))
            return base
        target = np.asarray(reference["target_points"], dtype=float)
        target_mean = np.asarray(cell["summary"]["mean"], dtype=float)
        target_cov = np.asarray(cell["summary"]["covariance"], dtype=float)
        pred_mean, pred_cov = _moments(points)
        mean_error = float(np.abs(pred_mean - target_mean).max())
        cov_error = float(np.abs(pred_cov - target_cov).max())
        tolerance = float(cell["summary"].get("tolerance", 1e-2))
        base.update(
            {
                "valid_output": 1.0,
                "feasible": float(mean_error <= tolerance and cov_error <= tolerance),
                "mean_linf_error": mean_error,
                "cov_linf_error": cov_error,
                "norm_cd": _norm_cd(points, target, target_cov),
                "iou": _iou(points, target),
                "coverage": _coverage(points, target),
            }
        )
        base.update(_view_metrics(submission, reference))
        return base
    except Exception as exc:
        base["error"] = str(exc)
        base.update(_view_metrics(submission, reference))
        return base


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
    return float(np.mean(values)) if values else float("nan")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cells": len(rows),
        "valid_rate": _mean(rows, "valid_output"),
        "feasible_rate": _mean(rows, "feasible"),
        "mean_norm_cd": _mean(rows, "norm_cd"),
        "mean_iou": _mean(rows, "iou"),
        "mean_coverage": _mean(rows, "coverage"),
        "mean_mrr": _mean(rows, "mrr"),
        "mean_utility_at_1": _mean(rows, "utility_at_1"),
        "mean_regret": _mean(rows, "regret"),
        "mean_runtime_s": _mean(rows, "runtime_s"),
    }


def evaluate(
    submission_path: Path,
    challenge_path: Path,
    reference_path: Path | None,
    metrics_path: Path,
    summary_path: Path,
    leaderboard_path: Path,
) -> None:
    raw_cells = _load_jsonl(challenge_path)
    if reference_path is None and raw_cells and "evaluator_only" in raw_cells[0]:
        cells, reference_rows = split_demo_cells(raw_cells)
    else:
        cells = raw_cells
        if reference_path is None:
            reference_path = ROOT / "evaluator_only" / "hidden_reference.jsonl"
        reference_rows = _load_jsonl(reference_path)
    references = {row["cell_id"]: row for row in reference_rows}
    submissions = {row.get("cell_id"): row for row in _load_jsonl(submission_path)}
    rows = [_evaluate_cell(cell, submissions.get(cell["cell_id"]), references[cell["cell_id"]]) for cell in cells]
    _write_csv(metrics_path, rows)
    summary = _summarize(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    from make_leaderboard import make_leaderboard

    make_leaderboard(metrics_path, leaderboard_path, leaderboard_path.with_suffix(".md"))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _default_challenge(submission_path: Path) -> Path:
    demo_cells = ROOT / "data" / "demo_cells.jsonl"
    if submission_path.stem.startswith("demo") and demo_cells.exists():
        return demo_cells
    return ROOT / "challenge_cells.jsonl"


def _default_outputs(
    submission_path: Path,
    metrics_out: Path | None,
    summary_out: Path | None,
    leaderboard_out: Path | None,
) -> tuple[Path, Path, Path]:
    outputs_dir = ROOT / "outputs"
    stem = submission_path.stem
    prefix = stem[: -len("_submission")] if stem.endswith("_submission") else stem
    if stem == "example_submission":
        metrics = outputs_dir / "example_metrics.csv"
        summary = outputs_dir / "example_metrics_summary.json"
        leaderboard = outputs_dir / "leaderboard.csv"
    elif prefix == "demo":
        metrics = outputs_dir / "demo_metrics.csv"
        summary = outputs_dir / "demo_scores.json"
        leaderboard = outputs_dir / "demo_leaderboard.csv"
    else:
        metrics = outputs_dir / f"{prefix}_metrics.csv"
        summary = outputs_dir / f"{prefix}_scores.json"
        leaderboard = outputs_dir / f"{prefix}_leaderboard.csv"
    return metrics_out or metrics, summary_out or summary, leaderboard_out or leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Protocol B JSONL submission.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--challenge", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--leaderboard-out", type=Path, default=None)
    args = parser.parse_args()
    challenge = args.challenge or _default_challenge(args.submission)
    metrics_out, summary_out, leaderboard_out = _default_outputs(args.submission, args.metrics_out, args.summary_out, args.leaderboard_out)
    evaluate(args.submission, challenge, args.reference, metrics_out, summary_out, leaderboard_out)


if __name__ == "__main__":
    main()
