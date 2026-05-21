# Reviewer Quickstart

This anonymous repository is organized so a reviewer can verify the public Protocol B loop in a few minutes without private data or vendor access.

Canonical navigation starts at `START_HERE_PROTOCOL_B.md`; this file is the
command-oriented companion, and `README.md` gives the broader inventory.

For a no-run paper/artifact inspection, start with `START_HERE_PROTOCOL_B.md`,
`docs/official_benchmark_contract.md`, `docs/submission_contract.md`,
`docs/public_dev_feature_dictionary.md`, `reports/README.md`,
`reports/PAPER_EVIDENCE_INDEX.md`, `reports/structured_evidence_check_map.md`,
`docs/cost_profile_derivation.md`, and `reports/REPORT_MANIFEST_SHA256.md`.

## Four reproduction tiers

| Tier | Command | Expected output | Notes |
| --- | --- | --- | --- |
| Quick check / dependency and material preflight | `python scripts/check_reproduction_prereqs.py --install-missing` | `REPRODUCTION PREFLIGHT PASSED` | Installs only missing lightweight packages from `requirements.txt`; checks required smoke materials. |
| Lightweight reproduction | `python scripts/run_smoke_reproduction.py` | `SMOKE REPRODUCTION PASSED`; smoke JSON files under the printed output directory | Replays public-dev dense/2Wiki scoring and cost-profile smoke checks. |
| Full reproduction | See `README.md` / `docs/reproduction_guide.md` | Heavy reports under `reports/` and `summaries/` | Optional; may need GPU and public dataset caches. |
| Custom partial reproduction | `python scripts/run_custom_reproduction.py` | User-selected smoke outputs | Interactive dense/2Wiki/cost subset. |

Generated files are not included in the clean archive. By default, smoke and
submission-kit checks write under the system temp directory rather than the
release tree. The smoke script also writes scorer detail files under a
`details/` subdirectory; all generated paths are listed in
`smoke_reproduction_summary.json`. Use `--out-dir` to choose a specific
reviewer-run directory.

## Legacy three-command check

This optional legacy check writes `outputs/` inside the checkout. For clean
paper-facing smoke review, use the four reproduction tiers above.

```bash
pip install -r requirements.txt
python scripts/run_demo.py --solver solvers/template_solver.py
python scripts/evaluate_submission.py outputs/demo_submission.jsonl
```

Expected generated output after running the commands:

- `outputs/demo_submission.jsonl`
- `outputs/demo_metrics.csv`
- `outputs/demo_scores.json`
- `outputs/demo_leaderboard.csv`
- `outputs/demo_leaderboard.md`

The demo exercises legacy public cells, visibility filtering, solver execution, standardized JSONL output, public scoring, and leaderboard-row generation. Solver code receives only method-visible fields; evaluator-only targets are used only by scoring scripts. The paper-facing dense/2Wiki smoke reproduction uses the four-tier guide above.

## Additional checks

These checks use the fuller reviewer/dev dependency set:

```bash
pip install -r requirements-full.txt
```

```bash
python scripts/check_pipeline_demo.py
python scripts/check_submission_kit.py --out-dir D:/m2sbench_reviewer_tmp/submission_kit_check
python scripts/verify_systems_profile.py
python -m pytest -q tests
```

These checks validate the demo pipeline, the public submission kit, schema/coverage behavior, and leakage-control invariants.

## Paper-facing Protocol B contract check

The controlled-cell demo above is intentionally small. The dense semantic and
2Wiki slices reported in the paper use the `schema/` and `menus/` contract. A
reviewer can check that contract without rerunning heavy experiments:

```bash
python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_dense.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json \
  --manifest manifests/dense_semantic.dev_manifest.jsonl

python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_2wiki_structured.jsonl \
  --view-menu menus/2wiki_structured.view_menu.json \
  --cost-menu menus/2wiki_structured.cost_menu.json \
  --manifest manifests/2wiki_structured.dev_manifest.jsonl

python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_2wiki_hyperlink.jsonl \
  --view-menu menus/2wiki_hyperlink.view_menu.json \
  --cost-menu menus/2wiki_hyperlink.cost_menu.json \
  --manifest manifests/2wiki_hyperlink.dev_manifest.jsonl
```

All three commands should print `VALID`. The companion fixture
`submission_template/protocol_b_invalid_leak.jsonl` should be rejected because
it contains an unpaid CE-score field.

## Paper-facing public-dev scorer

The dense and 2Wiki contract packs include a lightweight public-dev
submission/evaluation loop. These commands do not rerun the heavy paper audits;
they validate a route JSONL file and score it against evaluator-held public-dev
references:

Important: `public_dev/*.manifest.jsonl` files are method-visible solver inputs.
`public_dev/evaluator_only/*.reference.jsonl` files are evaluator-held public-dev references
for local scoring only; solvers should not train on them. See
`public_dev/DO_NOT_TRAIN_ON_REFERENCE.md`.

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_dense_semantic_router.jsonl \
  --manifest public_dev/dense_semantic.manifest.jsonl \
  --reference public_dev/evaluator_only/dense_semantic.reference.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json

python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl \
  --manifest public_dev/2wiki_structured.manifest.jsonl \
  --reference public_dev/evaluator_only/2wiki_structured.reference.jsonl \
  --view-menu menus/2wiki_structured.view_menu.json \
  --cost-menu menus/2wiki_structured.cost_menu.json

python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl \
  --manifest public_dev/2wiki_hyperlink.manifest.jsonl \
  --reference public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl \
  --view-menu menus/2wiki_hyperlink.view_menu.json \
  --cost-menu menus/2wiki_hyperlink.cost_menu.json
```

Each command writes per-row metrics, a JSON summary, and a one-row leaderboard
under the system temp directory unless explicit output paths are supplied. Use
`--out-dir`, or the `--metrics-out`, `--summary-out`, and `--leaderboard-out`
flags, to choose a specific reviewer-run directory. Private-test scoring
withholds the analogous reference rows while preserving the same route schema
and menu ids. The `2wiki_hyperlink` public-dev pack is a structured-stress
fixture; the paper-facing 7GB result is the frozen 10k report in `reports/`.

For a minimal paper-facing solver example, see
`solvers/protocol_b_public_dev_router.py`. It reads only method-visible
`public_dev/*.manifest.jsonl` rows and emits route JSONL.

## Static-overfit audit

The released 320 synthetic cells are a public development/review slice, not a permanent closed test set. The repository records a private-seed regeneration audit under the same Protocol B contract:

```bash
python scripts/run_static_overfit_audit.py
```

The default command reads frozen audit records and writes `summaries/static_overfit_audit/static_overfit_audit_summary.csv`. To rerun the full private-seed 320-cell computation, use `python scripts/run_static_overfit_audit.py --generate-private-seed-audit`; this is an extended check rather than part of the five-minute quickstart.

## Optional extended baseline

The conditional-flow baseline is outside the quickstart path because it depends on PyTorch. To reproduce the modern generative/projection row, run:

```bash
pip install -r requirements-full.txt
python scripts/run_conditional_flow_projection_baseline.py
```

It writes `summaries/conditional_flow_projection_summary.csv` and related per-cell/audit records.

## Files to inspect

- `docs/official_benchmark_contract.md`: one-page map of official slices, stress audits, diagnostics, portability checks, canonical reports, hidden fields, and private-test policy.
- `challenge_cells.jsonl` and `visibility_manifest.json`: method-visible legacy controlled-cell demo kit.
- `data/demo_cells.jsonl` and `data/visibility_manifest.json`: runnable demo cells and demo visibility manifest.
- `evaluator_only/hidden_reference.jsonl`: hidden references for local scoring; solvers must not import this file.
- `solvers/template_solver.py` and `submission_template/solver.py`: minimal solver adapters.
- `scripts/run_demo.py`, `scripts/validate_submission.py`, `scripts/evaluate_submission.py`, and `scripts/make_leaderboard.py`: public execution and scoring path.
- `outputs/demo_leaderboard.md`: generated leaderboard row after running the legacy quickstart; absent from the clean no-run archive by design.
- `schema/route_schema.json`, `menus/*.view_menu.json`, and `menus/*.cost_menu.json`: paper-facing route and evidence-menu declarations.
- `public_dev/`: method-visible public-dev manifests; evaluator-held scoring references are under `public_dev/evaluator_only/`.
- `manifests/official_slices.json`: machine-readable status card for official, stress, diagnostic, and portability records.
- `scripts/validate_protocol_b_route.py`: lightweight validator for official Protocol B JSONL route rows.
- `scripts/score_protocol_b_public_dev.py`: public-dev scorer for route JSONL submissions.
- `reports/PAPER_EVIDENCE_INDEX.md`: map from manuscript claims to frozen dense, 2Wiki, CE, cost-profile, and governance reports.
- `reports/structured_evidence_check_map.md`: compact core/supporting/boundary map for 2Wiki, HotpotQA, MuSiQue, DocRED, StrategyQA, and FEVER.
- `docs/legal_router_feature_tiers.md`: compact list of method-visible B0/B1 features and forbidden evaluator-held fields.
- `docs/public_dev_feature_dictionary.md`: concrete `visible_features` keys for dense, 2Wiki structured, and 2Wiki hyperlink public-dev rows.
- `docs/cost_profile_changelog.md`: profile-relative cost and CE accounting notes.
- `docs/cost_profile_derivation.md`: exact C_op values, secondary C_mem/C_lat interpretation, and 2Wiki C_op profile notes.
- `docs/provenance_and_genai.md`: artifact provenance, GenAI disclosure, and public-data memorization boundary.
- `ARTIFACT_INVENTORY.md`: file-level inventory and purpose annotation for every retained file.

## Optional licensed adapter

The Bloomberg adapter under `optional_adapters/bloomberg_protocol_b_adapter/` is not required for public reproduction or the core leaderboard. It contains scripts, schemas, and instructions only; returned vendor exports and row-level Bloomberg-derived data are not included.
