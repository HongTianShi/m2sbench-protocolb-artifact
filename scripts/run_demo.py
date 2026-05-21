from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from materialize_pipeline_demo import materialize_demo
from m2sbench.evaluator import load_jsonl, write_jsonl
from m2sbench.visibility import assert_no_hidden_fields, method_visible_cell


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_solver(path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("m2s_protocol_b_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise RuntimeError(f"{path} must define solve(cell)")
    return solve


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_demo(
    solver_path: Path,
    cells_path: Path,
    manifest_path: Path,
    out_path: Path,
    visible_out: Path,
    trace_out: Path,
    refresh: bool = False,
) -> None:
    if refresh or not cells_path.exists() or not manifest_path.exists():
        materialize_demo()
    manifest = _load_json(manifest_path)
    full_cells = load_jsonl(cells_path)
    visible_cells = [method_visible_cell(cell, manifest) for cell in full_cells]
    for cell in visible_cells:
        assert_no_hidden_fields(cell)
    solve = _load_solver(solver_path)
    submissions: list[dict[str, Any]] = []
    for cell in visible_cells:
        start = time.perf_counter()
        row = solve(cell)
        row.setdefault("cell_id", cell["cell_id"])
        row["runtime_s"] = round(time.perf_counter() - start, 6)
        submissions.append(row)
    write_jsonl(visible_out, visible_cells)
    write_jsonl(out_path, submissions)
    trace = {
        "demo_name": "Adapter-Ready Protocol-B Evaluation Pipeline",
        "solver": _display_path(solver_path),
        "n_cells": len(visible_cells),
        "stages": manifest.get("pipeline_stages", []),
        "method_visible_cells": _display_path(visible_out),
        "submission": _display_path(out_path),
        "leakage_guard": "passed: evaluator_only fields were removed before solver execution",
    }
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    trace_out.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(trace, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adapter-ready Protocol B pipeline demo.")
    parser.add_argument("--solver", type=Path, default=ROOT / "solvers" / "template_solver.py")
    parser.add_argument("--cells", type=Path, default=ROOT / "data" / "demo_cells.jsonl")
    parser.add_argument("--visibility", type=Path, default=ROOT / "data" / "visibility_manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo_submission.jsonl")
    parser.add_argument("--visible-out", type=Path, default=ROOT / "outputs" / "demo_method_visible_cells.jsonl")
    parser.add_argument("--trace-out", type=Path, default=ROOT / "outputs" / "demo_pipeline_trace.json")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run_demo(args.solver, args.cells, args.visibility, args.out, args.visible_out, args.trace_out, args.refresh)


if __name__ == "__main__":
    main()
