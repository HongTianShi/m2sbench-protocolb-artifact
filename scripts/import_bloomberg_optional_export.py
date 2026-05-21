from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COST_MENU = {"moments": 0.0, "occupancy": 0.28, "raster": 0.42, "point": 0.58}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest)


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} inside {root}")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _to_float(value: str) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _load_px_last(path: Path) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("field") != "PX_LAST":
                continue
            value = _to_float(row.get("value", ""))
            if value is None or value <= 0:
                continue
            grouped.setdefault(row["security"], {})[row["date"]] = value
    return {security: sorted(points.items()) for security, points in grouped.items()}


def _mean(points: list[list[float]]) -> list[float]:
    n = len(points)
    return [sum(point[idx] for point in points) / n for idx in range(2)]


def _cov(points: list[list[float]], mean: list[float]) -> list[list[float]]:
    n = len(points)
    c00 = sum((point[0] - mean[0]) * (point[0] - mean[0]) for point in points) / n
    c01 = sum((point[0] - mean[0]) * (point[1] - mean[1]) for point in points) / n
    c11 = sum((point[1] - mean[1]) * (point[1] - mean[1]) for point in points) / n
    return [[c00, c01], [c01, c11]]


def _window_points(values: list[float]) -> list[list[float]]:
    logs = [math.log(value) for value in values]
    base = logs[0]
    cumulative = [item - base for item in logs]
    y_mean = sum(cumulative) / len(cumulative)
    y_var = sum((item - y_mean) ** 2 for item in cumulative) / len(cumulative)
    y_scale = math.sqrt(y_var) or 1.0
    n = len(values)
    points = []
    for idx, item in enumerate(cumulative):
        x = -1.0 + 2.0 * idx / max(n - 1, 1)
        y = 0.35 * ((item - y_mean) / y_scale)
        points.append([round(x, 6), round(y, 6)])
    return points


def _support_points(cell_id: str, mean: list[float], cov: list[list[float]], n: int = 128) -> list[list[float]]:
    rng = random.Random(cell_id)
    sx = math.sqrt(max(cov[0][0], 1e-6))
    sy = math.sqrt(max(cov[1][1], 1e-6))
    return [[round(rng.gauss(mean[0], sx), 6), round(rng.gauss(mean[1], sy), 6)] for _ in range(n)]


def _oracle_view(values: list[float]) -> str:
    returns = [math.log(values[idx] / values[idx - 1]) for idx in range(1, len(values))]
    if not returns:
        return "moments"
    vol = math.sqrt(sum(item * item for item in returns) / len(returns))
    drawdown = 0.0
    peak = values[0]
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak)
    if drawdown >= 0.18:
        return "point"
    if vol >= 0.018:
        return "raster"
    if vol >= 0.010:
        return "occupancy"
    return "moments"


def _utilities(oracle: str) -> dict[str, float]:
    utilities = {"moments": 0.45, "occupancy": 0.56, "raster": 0.63, "point": 0.70, "abstain": 0.0}
    utilities[oracle] = 0.84
    return utilities


def _build_cells(panel: dict[str, list[tuple[str, float]]], max_cells: int, window_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    per_security_counts: dict[str, int] = {}
    eligible_securities = [security for security in sorted(panel) if len(panel[security]) >= window_size]
    windows_per_security = max(3, math.ceil(max_cells / max(len(eligible_securities), 1)))
    for security in eligible_securities:
        series = panel[security]
        if len(series) <= window_size:
            windows = [series]
        else:
            max_start = len(series) - window_size
            if max_start + 1 <= windows_per_security:
                starts = list(range(max_start + 1))
            else:
                starts = [round(idx * max_start / max(windows_per_security - 1, 1)) for idx in range(windows_per_security)]
            starts = sorted(set(starts))
            windows = [series[start : start + window_size] for start in starts]
        for window in windows:
            if len(cells) >= max_cells:
                break
            dates = [item[0] for item in window]
            values = [item[1] for item in window]
            cell_id = f"bbg_{len(cells):03d}"
            points = _window_points(values)
            mean = _mean(points)
            cov = _cov(points, mean)
            oracle = _oracle_view(values)
            cell = {
                "cell_id": cell_id,
                "protocol": "Protocol B optional Bloomberg adapter",
                "n_points": len(points),
                "summary": {
                    "mean": [round(item, 6) for item in mean],
                    "covariance": [[round(item, 6) for item in row] for row in cov],
                    "n_points": len(points),
                    "tolerance": 0.01,
                },
                "granted_views": {
                    "support_pool_id": f"optional_bloomberg_support_{cell_id}",
                    "support_points": _support_points(cell_id, mean, cov),
                    "support_bounds": [[-1.2, -1.2], [1.2, 1.2]],
                    "visible_view_names": ["moments", "support_points"],
                },
                "cost_menu": COST_MENU,
                "output_schema": {
                    "required": ["cell_id"],
                    "accepted_prediction_fields": ["candidate", "candidates[0].points", "points"],
                    "accepted_access_fields": ["ranked_views", "route"],
                },
                "evaluator_only": {
                    "target_points": points,
                    "family": "optional_bloomberg_price_path",
                    "budget_id": "realized_return_window",
                    "seed": len(cells),
                    "phase_name": f"optional adapter/{oracle}",
                    "source_security": security,
                    "source_start": dates[0],
                    "source_end": dates[-1],
                    "oracle_view": oracle,
                    "view_utilities": _utilities(oracle),
                },
            }
            cells.append(cell)
            per_security_counts[security] = per_security_counts.get(security, 0) + 1
        if len(cells) >= max_cells:
            break
    summary = {
        "n_cells": len(cells),
        "window_size": window_size,
        "max_cells": max_cells,
        "per_security_counts": per_security_counts,
    }
    return cells, summary


def _visibility_manifest(n_cells: int, source_zip: Path) -> dict[str, Any]:
    return {
        "schema_version": "m2sbench-optional-bloomberg-adapter-v1",
        "protocol": "Protocol B optional Bloomberg adapter",
        "demo_name": "Optional Bloomberg Terminal Adapter Demo",
        "demo_cells": "data/bloomberg_optional_demo_cells.jsonl",
        "source_zip_sha256": _sha256(source_zip),
        "pipeline_stages": [
            "Bloomberg Terminal export zip",
            "local import and optional cell construction",
            "visibility manifest",
            "method-visible input",
            "solver adapter",
            "standard JSONL submission",
            "public evaluator",
            "optional leaderboard row",
        ],
        "method_visible_files": ["data/bloomberg_optional_demo_cells.jsonl", "data/bloomberg_optional_visibility_manifest.json"],
        "evaluator_only_fields": ["target_points", "source_security", "source_start", "source_end", "oracle_view", "view_utilities"],
        "n_cells": n_cells,
        "leakage_rule": "run_demo.py strips evaluator_only before solver execution; evaluate_submission.py reads it only during scoring.",
        "licensing_note": "The Bloomberg export remains local and is not required for the paper's main reproducibility path.",
    }


def import_export(zip_path: Path, max_cells: int, window_size: int) -> dict[str, Any]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    run_root = ROOT / "optional_bloomberg_imports" / zip_path.stem
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    _safe_extract(zip_path, run_root)
    manifest_path = _find_one(run_root, "adapter_manifest.json")
    audit_path = _find_one(run_root, "adapter_audit.json")
    long_path = _find_one(run_root, "bdh_daily_long.csv")
    manifest = _read_json(manifest_path)
    audit = _read_json(audit_path)
    panel = _load_px_last(long_path)
    cells, cell_summary = _build_cells(panel, max_cells=max_cells, window_size=window_size)
    if not cells:
        raise RuntimeError("No optional Bloomberg demo cells could be built. Check PX_LAST coverage in bdh_daily_long.csv.")
    cells_path = ROOT / "data" / "bloomberg_optional_demo_cells.jsonl"
    visibility_path = ROOT / "data" / "bloomberg_optional_visibility_manifest.json"
    _write_jsonl(cells_path, cells)
    visibility = _visibility_manifest(len(cells), zip_path)
    _write_json(visibility_path, visibility)
    summary = {
        "status": "ok",
        "source_zip": str(zip_path),
        "source_zip_sha256": _sha256(zip_path),
        "import_dir": str(run_root),
        "adapter_status": manifest.get("status"),
        "adapter_audit_status": audit.get("status"),
        "n_panel_securities": len(panel),
        "n_px_last_points": sum(len(items) for items in panel.values()),
        "cells_path": str(cells_path),
        "visibility_manifest": str(visibility_path),
        **cell_summary,
    }
    _write_json(ROOT / "outputs" / "bloomberg_optional_import_summary.json", summary)
    with (ROOT / "outputs" / "bloomberg_optional_import_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an optional Bloomberg adapter export into Protocol B demo cells.")
    parser.add_argument("export_zip", type=Path)
    parser.add_argument("--max-cells", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=64)
    args = parser.parse_args()
    summary = import_export(args.export_zip, args.max_cells, args.window_size)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
