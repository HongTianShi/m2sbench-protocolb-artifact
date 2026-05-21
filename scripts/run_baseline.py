from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_solver(path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("m2s_submission_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise RuntimeError(f"{path} must define a callable solve(cell) function")
    return solve


def run_baseline(challenge_path: Path, solver_path: Path, out_path: Path, limit: int | None = None) -> None:
    cells = _load_jsonl(challenge_path)
    if limit is not None:
        cells = cells[:limit]
    solve = _load_solver(solver_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for cell in cells:
            start = time.perf_counter()
            result = solve(cell)
            if "cell_id" not in result:
                result["cell_id"] = cell["cell_id"]
            result["runtime_s"] = round(time.perf_counter() - start, 6)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {len(cells)} submissions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Protocol B solver over public challenge cells.")
    parser.add_argument("--challenge", type=Path, default=ROOT / "challenge_cells.jsonl")
    parser.add_argument("--solver", type=Path, default=ROOT / "submission_template" / "solver.py")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "example_submission.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_baseline(args.challenge, args.solver, args.out, args.limit)


if __name__ == "__main__":
    main()
