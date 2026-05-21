$ErrorActionPreference = "Stop"

python scripts/run_baseline.py `
  --solver submission_template/solver.py `
  --challenge challenge_cells.jsonl `
  --out outputs/example_submission.jsonl
