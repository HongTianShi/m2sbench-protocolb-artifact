from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2sbench.research.data import make_case

DEFAULT_CELLS = ROOT / "summaries" / "foundation_model_direct_generation_cells.csv"
COST_MENU = {"moments": 0.0, "occupancy": 0.28, "raster": 0.42, "point": 0.58}
TOLERANCE = 1e-2


def _rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), digits).tolist()
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, tuple):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _oracle_view(phase_name: str, covariance: np.ndarray) -> str:
    eigvals = np.linalg.eigvalsh(covariance)
    condition = float(eigvals.max() / max(eigvals.min(), 1e-8))
    phase = phase_name.lower()
    if "optimization" in phase or condition >= 5.0:
        return "point"
    if "decision" in phase:
        return "raster"
    if "collapsed" in phase or condition < 1.5:
        return "moments"
    return "occupancy"


def _view_utilities(oracle: str) -> dict[str, float]:
    utilities = {"moments": 0.45, "occupancy": 0.55, "raster": 0.62, "point": 0.68, "abstain": 0.0}
    utilities[oracle] = 0.82
    if oracle == "point":
        utilities["raster"] = max(utilities["raster"], 0.70)
    if oracle == "moments":
        utilities["occupancy"] = max(utilities["occupancy"], 0.65)
    return utilities


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_submission_kit(n_cells: int) -> None:
    source = pd.read_csv(DEFAULT_CELLS).head(n_cells)
    challenge_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        cell_id = f"pb_{index:03d}"
        case = make_case(
            str(row["family"]),
            str(row["budget"]),
            int(row["seed"]),
            protocol="protocol_b",
            n_points=64,
            dense_support_points=512,
            support_pool_size=128,
        )
        summary = {
            "mean": _rounded(case.summary_stats["mean"]),
            "covariance": _rounded(case.summary_stats["covariance"]),
            "n_points": int(case.n_points),
            "tolerance": TOLERANCE,
        }
        support = case.allowed_support_view
        public_cell = {
            "cell_id": cell_id,
            "protocol": "Protocol B",
            "n_points": int(case.n_points),
            "summary": summary,
            "granted_views": {
                "support_pool_id": f"support_{cell_id}",
                "support_points": _rounded(support.dense_support),
                "support_bounds": _rounded(support.bounds),
                "visible_view_names": ["moments", "support_points"],
            },
            "cost_menu": COST_MENU,
            "output_schema": {
                "required": ["cell_id"],
                "accepted_prediction_fields": ["candidate", "candidates[0].points", "points"],
                "accepted_access_fields": ["ranked_views", "route"],
            },
        }
        oracle = _oracle_view(str(row["phase_name"]), case.summary_stats["covariance"])
        hidden_rows.append(
            {
                "cell_id": cell_id,
                "target_points": _rounded(case.canonical_target),
                "family": str(row["family"]),
                "budget_id": str(row["budget"]),
                "seed": int(row["seed"]),
                "phase_name": str(row["phase_name"]),
                "source_probe_cell_id": str(row["probe_cell_id"]),
                "oracle_view": oracle,
                "view_utilities": _view_utilities(oracle),
            }
        )
        challenge_rows.append(public_cell)

    challenge_path = ROOT / "challenge_cells.jsonl"
    hidden_path = ROOT / "evaluator_only" / "hidden_reference.jsonl"
    _write_jsonl(challenge_path, challenge_rows)
    _write_jsonl(hidden_path, hidden_rows)
    manifest = {
        "schema_version": "m2sbench-submission-kit-v1",
        "protocol": "Protocol B",
        "n_cells": len(challenge_rows),
        "method_visible_files": ["challenge_cells.jsonl", "visibility_manifest.json", "challenge_cell_schema.json", "submission_schema.json"],
        "evaluator_only_files": ["evaluator_only/hidden_reference.jsonl"],
        "solver_entrypoint": "submission_template/solver.py::solve",
        "validator": "scripts/validate_submission.py",
        "visibility_contract": {
            "method_visible_fields": [
                "cell_id",
                "n_points",
                "summary.mean",
                "summary.covariance",
                "summary.tolerance",
                "granted_views.support_points",
                "granted_views.support_bounds",
                "cost_menu",
            ],
            "evaluator_only_fields": [
                "target_points",
                "family",
                "budget_id",
                "seed",
                "phase_name",
                "source_probe_cell_id",
                "oracle_view",
                "view_utilities",
            ],
            "leakage_rule": "Solvers must never import evaluator_only/hidden_reference.jsonl or use hidden target/family/budget/seed fields.",
        },
        "quick_start": [
            "pip install -r requirements.txt",
            "bash scripts/run_baseline.sh",
            "python scripts/evaluate_submission.py outputs/example_submission.jsonl",
        ],
        "metrics": ["valid_output", "feasible", "mean_linf_error", "cov_linf_error", "norm_cd", "iou", "coverage", "mrr", "utility_at_1", "regret"],
        "hashes": {
            "challenge_cells.jsonl": _sha256(challenge_path),
            "evaluator_only/hidden_reference.jsonl": _sha256(hidden_path),
        },
    }
    manifest_path = ROOT / "visibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the public Protocol B submission kit files.")
    parser.add_argument("--n-cells", type=int, default=50)
    args = parser.parse_args()
    build_submission_kit(args.n_cells)
    print(f"Wrote {args.n_cells} Protocol B cells to {ROOT / 'challenge_cells.jsonl'}")


if __name__ == "__main__":
    main()
