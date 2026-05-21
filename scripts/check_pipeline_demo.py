from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, "scripts/materialize_pipeline_demo.py", "--n-cells", "50"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/run_demo.py", "--solver", "solvers/template_solver.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/validate_submission.py", "outputs/demo_submission.jsonl", "--challenge", "data/demo_cells.jsonl"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/evaluate_submission.py", "outputs/demo_submission.jsonl"], cwd=ROOT, check=True)
    print("Adapter-ready Protocol B pipeline demo check passed.")


if __name__ == "__main__":
    main()
