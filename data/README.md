# Demo Data

`demo_cells.jsonl` contains public Protocol B fields plus an `evaluator_only` block.
`scripts/run_demo.py` removes `evaluator_only` before calling a solver. `scripts/evaluate_submission.py` reads it only during scoring.
