# New Evidence-View Onboarding Checklist

Protocol B adds new evidence views without changing the route JSONL schema. A new view is ready for public development or private testing when the following items are declared.

## 1. View Declaration

- Unique `view_id`.
- Slice id, such as `dense_semantic_access`, `2wiki_structured_evidence`, or `2wiki_hyperlink_stress`.
- Visibility tier (`B0`, `B1`, `B2`, or `paid`).
- Free fields visible before purchase.
- Paid fields materialized only after purchase.
- Evaluator-only fields forbidden before purchase.
- Parent/dependency views when cumulative costs differ from incremental costs.
- A `menus/<slice>.view_menu.json` record that passes JSON parsing and lists the frozen `scorer_id`.

## 2. Cost Declaration

- Cost-menu/profile id.
- Official `C_op` cost if the view is part of a leaderboard.
- Optional `C_mem` and `C_lat` rescoring fields.
- Parent/dependency views for cumulative profiles.
- Whether a cost is incremental after an existing first-stage pool or cumulative over the full evidence ladder.
- A `menus/<slice>.cost_menu.json` record whose id is the value submitted in each row's `cost_menu` field.

## 3. Scorer Contract

- Frozen scorer id and metric.
- Hidden evaluator fields used by the scorer.
- Raw quality field reported before cost adjustment.
- Utility and regret formula.
- Treatment of ties, abstention, and invalid routes.

## 4. Leakage Controls

- Validator rejects hidden qrels, support facts, labels, hidden targets, unpaid full/CE scores, source labels, and undeclared context.
- Public-dev rows may expose qrels only for development; private-test rows withhold the same evaluator-only fields.
- External tools may only use released fields and purchased evidence.

## 5. Required Artifact Records

- A schema/view-menu declaration under `menus/`.
- A cost-menu declaration under `menus/`.
- A frozen report under `reports/`.
- A short README or report note describing how the view supports the paper claim.
- If the view belongs to an official slice, an entry in `docs/official_benchmark_contract.md`.
- A script or documented command that regenerates or validates the report when data access permits.
- A passing dry-run command with `scripts/validate_protocol_b_route.py`.

Views that do not satisfy these items can still appear as portability checks, but they should not be described as official validated leaderboard slices.

## 6. Minimal Validator Dry Run

For each new view menu, add at least one legal JSONL row under
`submission_template/` and run:

```bash
python scripts/validate_protocol_b_route.py \
  submission_template/<slice>_valid.jsonl \
  --view-menu menus/<slice>.view_menu.json \
  --cost-menu menus/<slice>.cost_menu.json
```

If the new view introduces a paid score, add an invalid fixture showing that the
unpaid score is rejected before purchase.
