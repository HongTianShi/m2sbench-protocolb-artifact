# Solver Template

Edit `solver.py` and keep the public `solve(cell)` function. The runner imports this
function, passes one public Protocol B cell at a time, and writes one JSON object per
cell to the submission JSONL file.

This directory also contains paper-facing Protocol B route fixtures:

- `protocol_b_valid_dense.jsonl`
- `protocol_b_valid_2wiki_structured.jsonl`
- `protocol_b_valid_2wiki_hyperlink.jsonl`
- `protocol_b_invalid_leak.jsonl`

Validate those rows with `scripts/validate_protocol_b_route.py` and the matching
`menus/*.view_menu.json` / `menus/*.cost_menu.json` files. The `solver.py`
template below is for the legacy controlled-cell quickstart.

The template is intentionally small:

- Input: method-visible summary and granted views only.
- Output: candidate structures, ranked views, and a route decision.
- Leakage guard: hidden targets are never passed to `solve(cell)`.

Run the template baseline with:

```bash
python scripts/run_baseline.py --solver submission_template/solver.py
python scripts/evaluate_submission.py outputs/example_submission.jsonl
```
