# Adapter-Ready Protocol-B Evaluation Pipeline

The demo runs the full Protocol B loop from fixed challenge cells to visibility
filtering, solver execution, standardized submission, public evaluation, and
leaderboard-row generation.

## What It Demonstrates

- The benchmark is executable from fixed cells.
- Evaluator-only targets are not passed to solvers.
- New methods can be attached through a single `solve(cell)` adapter.
- Feasibility, CD/IoU, Top-k/MRR, utility/regret, coverage, and runtime are computed automatically.
- Results become a reproducible leaderboard row.

## Pipeline

```text
Fixed challenge cells
        |
Visibility manifest
        |
Method-visible input
        |
Solver adapter
        |
Standard JSONL submission
        |
Evaluator
        |
Scores + leaderboard row
```

## Three Commands

```bash
pip install -r requirements.txt
python scripts/run_demo.py --solver solvers/template_solver.py
python scripts/evaluate_submission.py outputs/demo_submission.jsonl
```

The commands write:

- `outputs/demo_submission.jsonl`
- `outputs/demo_scores.json`
- `outputs/demo_leaderboard.md`
- `outputs/demo_pipeline_trace.json`
- `outputs/demo_method_visible_cells.jsonl`

## Adapter Contract

Each solver file defines:

```python
def solve(cell):
    """
    Input: method-visible Protocol B cell only.
    Hidden target and evaluator-only views are never exposed.
    """
    return {
        "cell_id": cell["cell_id"],
        "candidate": None,
        "ranked_views": ["moments", "occupancy", "raster", "point"],
        "route": "moments",
    }
```

`scripts/run_demo.py` applies `data/visibility_manifest.json` before calling the
solver. `scripts/evaluate_submission.py` is the only step that reads evaluator-only
targets.

## Useful Variants

```bash
python scripts/run_demo.py --solver solvers/random_solver.py
python scripts/run_demo.py --solver solvers/blind_sa_solver.py
python scripts/validate_submission.py outputs/demo_submission.jsonl --challenge data/demo_cells.jsonl
python scripts/check_pipeline_demo.py
python scripts/make_leaderboard.py outputs/demo_metrics.csv --out outputs/demo_leaderboard.csv --markdown-out outputs/demo_leaderboard.md
```

The notebook version is `notebooks/protocol_b_quickstart.ipynb`.

## Optional Bloomberg Adapter

For a licensed Bloomberg Terminal machine, the optional adapter lives at
`optional_adapters/bloomberg_protocol_b_adapter/`. Copy that folder to the
Bloomberg machine, run the adapter there, and bring the generated export zip
back to this workspace. `scripts/import_bloomberg_optional_export.py` imports
that export into `data/bloomberg_optional_demo_cells.jsonl` for the same
Protocol B pipeline. This adapter is not needed for the paper's main
reproducibility path. A licensed local smoke run has been verified end-to-end;
`OPTIONAL_BLOOMBERG_VERIFIED_RUN.md` records aggregate status without
redistributing Bloomberg data.
