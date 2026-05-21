# Start Here: Paper-Facing Protocol B Artifact

This page is the shortest path through the artifact for reviewers and future
participants. It separates the paper-facing dense/2Wiki Protocol B contract
from the older controlled-cell demo retained for backward-compatible smoke
tests.

Use this file as the canonical first page. `REVIEWER_QUICKSTART.md` is the
command-oriented companion, and `README.md` is the broader inventory.

## What To Read First

| Need | File |
| --- | --- |
| Official slice/status map | `docs/official_benchmark_contract.md` |
| Submission fields, valid/invalid rows, private-test policy | `docs/submission_contract.md` |
| Route JSON schema | `schema/route_schema.json` |
| Declared evidence views and hidden fields | `menus/*.view_menu.json` |
| Declared cost profiles | `menus/*.cost_menu.json` |
| Public-dev feature dictionary | `docs/public_dev_feature_dictionary.md` |
| Canonical report map | `reports/README.md` |
| Paper claim to frozen-report map | `reports/PAPER_EVIDENCE_INDEX.md` |
| Structured evidence core/supporting/boundary map | `reports/structured_evidence_check_map.md` |
| Cost-profile derivation and profile semantics | `docs/cost_profile_derivation.md` |

## What Solvers May Read

For the paper-facing public-dev loop, solver code should read only:

- `public_dev/*.manifest.jsonl`
- `menus/*.view_menu.json`
- `menus/*.cost_menu.json`
- optional public documentation under `docs/`

Solvers must not read:

- `public_dev/evaluator_only/*.reference.jsonl`
- `evaluator_only/`
- frozen reports under `reports/` as training labels
- support facts, qrels, answers, CE scores, full-view scores, oracle utilities,
  or any field not declared method-visible by the view menu.

See `public_dev/DO_NOT_TRAIN_ON_REFERENCE.md` for the public-dev/private-test
split.

## Primary Route Semantics

Official leaderboard rows score one primary action.

- If `route` is present, the evaluator charges and scores that view.
- If `route` is absent and `ranked_views` is present, the first valid ranked
  view is used as the route.
- If both `route` and `ranked_views` are present, `route` is the scored action
  and `ranked_views` is a diagnostic ranking/tie-break field.
- `view_set` and `abstain` are rejected by current official menus unless a
  future menu declares deterministic scoring and cost rules for them.

Examples:

```json
{"query_id":"dense_dev_q0000","cell_id":"dense_dev_cell_0000","tier":"B1_score","cost_menu":"dense-semantic-v1-op","route":"int8"}
```

```json
{"query_id":"dense_dev_q0000","cell_id":"dense_dev_cell_0000","tier":"B1_score","cost_menu":"dense-semantic-v1-op","route":"int8","ranked_views":["int8","full","ce"]}
```

```json
{"query_id":"dense_dev_q0000","cell_id":"dense_dev_cell_0000","tier":"B1_score","cost_menu":"dense-semantic-v1-op","ranked_views":["int8","full","ce"]}
```

## Paper-Facing Slices

| Slice | Manifest | Menu | Status |
| --- | --- | --- | --- |
| Dense semantic access | `public_dev/dense_semantic.manifest.jsonl` | `menus/dense_semantic.*` | validated dense access / modest stress |
| 2Wiki structured evidence | `public_dev/2wiki_structured.manifest.jsonl` | `menus/2wiki_structured.*` | validated structured evidence core |
| 2Wiki hyperlink stress | `public_dev/2wiki_hyperlink.manifest.jsonl` | `menus/2wiki_hyperlink.*` | 7GB structured stress fixture |

HotpotQA and MuSiQue are supporting structured-evidence checks, not promoted
official leaderboard slices. Their machine-readable status cards are in
`manifests/supporting_structured_checks.json`.

## Legacy Demo Boundary

The root-level `challenge_cells.jsonl`, `visibility_manifest.json`,
`submission_schema.json`, `solvers/template_solver.py`, and `scripts/run_demo.py`
belong to the older controlled-cell demo. They are retained as a small runnable
smoke test, but they are not the dense/2Wiki paper-facing route contract.

For the paper-facing contract, use `schema/`, `menus/`, `public_dev/`,
`submission_template/protocol_b_*.jsonl`, `scripts/validate_protocol_b_route.py`,
and `scripts/score_protocol_b_public_dev.py`.

## Release Hygiene

This local working copy may contain `.git` metadata because it is a GitHub
upload clone. The anonymous web repository and clean release archives should
expose only tracked files, not `.git` history, remotes, caches, bytecode, local
logs, or notebook checkpoints. The maintainer checklist is in
`docs/release_checklist.md`, and `scripts/build_anonymous_release.py` is the
clean-export helper.
