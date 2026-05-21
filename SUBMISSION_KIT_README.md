# M2S-Bench Submission Kit Quick Start

This file describes the legacy controlled-cell quickstart. It remains useful for
checking the adapter-ready pipeline, but it is not the paper-facing dense/2Wiki
route contract. The paper-facing Protocol B contract is:

- `schema/route_schema.json`
- `menus/*.view_menu.json`
- `menus/*.cost_menu.json`
- `scripts/validate_protocol_b_route.py`
- `docs/submission_contract.md`

For dense and 2Wiki route rows, use the validator commands in
`REVIEWER_QUICKSTART.md`. For the controlled-cell quickstart below, a new method
only reads `challenge_cells.jsonl`, writes a JSONL submission, and runs the public
evaluator. The hidden target file is present for local scoring, but must never be
imported by a solver.

## Three-Command Path

```bash
pip install -r requirements.txt
bash scripts/run_baseline.sh
python scripts/evaluate_submission.py outputs/example_submission.jsonl
```

Windows PowerShell users can replace the second command with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

The evaluator writes:

- `outputs/example_metrics.csv`: per-cell feasibility, CD/IoU, view-ranking, and regret.
- `outputs/example_metrics_summary.json`: aggregate paper-style metrics.
- `outputs/leaderboard.csv`: compact comparable result table.
- `outputs/leaderboard.md`: Markdown version for reports.

## What Solvers Can Read

Solvers receive one cell at a time from `challenge_cells.jsonl`. Public fields include
the opaque `cell_id`, the target `n_points`, moment summary, granted support view,
normalized access-cost menu, and the expected output schema.

Solvers must not read:

- `evaluator_only/hidden_reference.jsonl`
- any target/canonical-point file
- hidden family, budget, seed, or oracle-view fields

`visibility_manifest.json` records this contract explicitly.

## Submission Schema

Each line in a submission JSONL file should look like:

```json
{
  "cell_id": "pb_000",
  "method": "my_solver",
  "candidates": [{"points": [[0.0, 0.0], [0.1, 0.2]]}],
  "ranked_views": ["moments", "occupancy", "raster", "point"],
  "route": "direct"
}
```

The evaluator also accepts a top-level `points` field for single-candidate solvers.
Use `route: "abstain"` when a method deliberately declines to make a structural
proposal or downstream routing decision.

Before scoring, check format with:

```bash
python scripts/validate_submission.py outputs/example_submission.jsonl
```

Machine-readable schema files are provided as `challenge_cell_schema.json` and
`submission_schema.json`.

For the adapter-ready end-to-end pipeline demo, see `PIPELINE_DEMO_README.md`.

## Rebuilding the Kit Files

The shipped kit files are already materialized. To refresh them from the synchronized
paper artifacts, run:

```bash
python scripts/materialize_submission_kit.py --n-cells 50
```

This regenerates `challenge_cells.jsonl`, `visibility_manifest.json`, and
`evaluator_only/hidden_reference.jsonl` from the frozen Protocol B probe cells.
