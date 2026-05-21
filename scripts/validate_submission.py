from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALID_VIEWS = {"moments", "occupancy", "raster", "point"}
VALID_ROUTES = {"direct", "abstain", "escalate", "moments", "occupancy", "raster", "point"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _points(row: dict[str, Any]) -> Any:
    if "candidate" in row:
        return row["candidate"]
    if "points" in row:
        return row["points"]
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            return first.get("points")
        return first
    return None


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_points(points: Any, n_points: int, cell_id: str) -> list[str]:
    errors: list[str] = []
    if points is None:
        return errors
    if not isinstance(points, list):
        return [f"{cell_id}: points must be a list"]
    if len(points) != n_points:
        errors.append(f"{cell_id}: expected {n_points} points, got {len(points)}")
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2 or not all(_valid_number(value) for value in point):
            errors.append(f"{cell_id}: point {index} must be [finite_number, finite_number]")
            break
    return errors


def validate(submission_path: Path, challenge_path: Path) -> list[str]:
    cells = {row["cell_id"]: row for row in _load_jsonl(challenge_path)}
    rows = _load_jsonl(submission_path)
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str):
            errors.append("submission row is missing string cell_id")
            continue
        if cell_id not in cells:
            errors.append(f"{cell_id}: unknown cell_id")
            continue
        if cell_id in seen:
            errors.append(f"{cell_id}: duplicate submission row")
        seen.add(cell_id)
        ranked_views = row.get("ranked_views", [])
        if ranked_views and (not isinstance(ranked_views, list) or any(view not in VALID_VIEWS for view in ranked_views)):
            errors.append(f"{cell_id}: ranked_views must use {sorted(VALID_VIEWS)}")
        route = row.get("route", "direct")
        if route not in VALID_ROUTES:
            errors.append(f"{cell_id}: route must use {sorted(VALID_ROUTES)}")
        errors.extend(_validate_points(_points(row), int(cells[cell_id]["n_points"]), cell_id))
    missing = sorted(set(cells) - seen)
    if missing:
        errors.append(f"missing {len(missing)} cells; first missing id is {missing[0]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Protocol B submission before scoring.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--challenge", type=Path, default=ROOT / "challenge_cells.jsonl")
    args = parser.parse_args()
    errors = validate(args.submission, args.challenge)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    try:
        display_path = args.challenge.resolve().relative_to(ROOT)
    except ValueError:
        display_path = args.challenge
    print(f"Submission is valid for {display_path}.")


if __name__ == "__main__":
    main()
