from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["method", "n", "valid_rate", "feasible_rate", "norm_cd", "iou", "coverage", "mrr", "utility_at_1", "regret", "runtime_s"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [_float(row, key) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "| " + " | ".join(FIELDS) + " |"
    sep = "| " + " | ".join(["---"] * len(FIELDS)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[field]) for field in FIELDS) + " |")
    path.write_text("\n".join([header, sep, *body]) + "\n", encoding="utf-8")


def make_leaderboard(metrics_path: Path, out_path: Path, markdown_path: Path | None = None) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(metrics_path):
        grouped[row.get("method", "unknown")].append(row)
    rows: list[dict[str, Any]] = []
    for method, items in grouped.items():
        rows.append(
            {
                "method": method,
                "n": len(items),
                "valid_rate": round(_mean(items, "valid_output"), 3),
                "feasible_rate": round(_mean(items, "feasible"), 3),
                "norm_cd": round(_mean(items, "norm_cd"), 3),
                "iou": round(_mean(items, "iou"), 3),
                "coverage": round(_mean(items, "coverage"), 3),
                "mrr": round(_mean(items, "mrr"), 3),
                "utility_at_1": round(_mean(items, "utility_at_1"), 3),
                "regret": round(_mean(items, "regret"), 3),
                "runtime_s": round(_mean(items, "runtime_s"), 4),
            }
        )
    rows.sort(key=lambda row: (-float(row["feasible_rate"]), float(row["norm_cd"]), -float(row["mrr"])))
    _write_csv(out_path, rows)
    if markdown_path is not None:
        _write_markdown(markdown_path, rows)
    try:
        display_path = out_path.resolve().relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(f"Wrote leaderboard to {display_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact leaderboard from evaluator metrics.")
    parser.add_argument("metrics", type=Path, nargs="?", default=ROOT / "outputs" / "example_metrics.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "leaderboard.csv")
    parser.add_argument("--markdown-out", type=Path, default=ROOT / "outputs" / "leaderboard.md")
    args = parser.parse_args()
    make_leaderboard(args.metrics, args.out, args.markdown_out)


if __name__ == "__main__":
    main()
