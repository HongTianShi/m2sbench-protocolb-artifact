from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2sbench.evaluator import load_jsonl, write_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_demo(n_cells: int = 50) -> None:
    challenge_path = ROOT / "challenge_cells.jsonl"
    reference_path = ROOT / "evaluator_only" / "hidden_reference.jsonl"
    manifest_path = ROOT / "visibility_manifest.json"
    if not challenge_path.exists() or not reference_path.exists():
        from materialize_submission_kit import build_submission_kit

        build_submission_kit(n_cells)
    public_cells = load_jsonl(challenge_path)[:n_cells]
    references = {row["cell_id"]: row for row in load_jsonl(reference_path)}
    demo_cells = []
    for cell in public_cells:
        row = dict(cell)
        row["evaluator_only"] = references[cell["cell_id"]]
        demo_cells.append(row)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "demo_cells.jsonl", demo_cells)
    manifest = _load_json(manifest_path)
    manifest["demo_name"] = "Adapter-Ready Protocol-B Evaluation Pipeline"
    manifest["demo_cells"] = "data/demo_cells.jsonl"
    manifest["method_visible_projection"] = "m2sbench.visibility.method_visible_cell"
    manifest["pipeline_stages"] = [
        "public challenge cells",
        "visibility manifest",
        "method-visible input",
        "solver adapter",
        "standard JSONL submission",
        "public evaluator",
        "scores and leaderboard row",
    ]
    (data_dir / "visibility_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (data_dir / "README.md").write_text(
        "# Demo Data\n\n"
        "`demo_cells.jsonl` contains public Protocol B fields plus an `evaluator_only` block.\n"
        "`scripts/run_demo.py` removes `evaluator_only` before calling a solver. "
        "`scripts/evaluate_submission.py` reads it only during scoring.\n",
        encoding="utf-8",
    )
    print("Wrote adapter-ready demo data to data/demo_cells.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the adapter-ready Protocol B demo data.")
    parser.add_argument("--n-cells", type=int, default=50)
    args = parser.parse_args()
    materialize_demo(args.n_cells)


if __name__ == "__main__":
    main()
