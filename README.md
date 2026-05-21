# M2S-Bench Anonymous Artifact Repository

This anonymous repository is the runnable artifact for the M2S-Bench CIKM submission. It contains the public Protocol B development cells, visibility manifests, solver templates, evaluators, baseline outputs, regeneration/audit records, tests, and a file-level inventory in `ARTIFACT_INVENTORY.md`.

Canonical review navigation starts with `START_HERE_PROTOCOL_B.md` and
`REVIEWER_QUICKSTART.md`; this README provides the broader repository inventory.

## Reviewer Inspection Path

If you only inspect a few files during review, start here:

1. `docs/official_benchmark_contract.md`: official slices, stress audits, hidden fields, metrics, and private-test policy.
2. `docs/submission_contract.md`: valid/invalid JSONL rows, visibility tiers, tool-use rules, and leakage boundaries.
3. `reports/PAPER_EVIDENCE_INDEX.md`: map from paper claims to frozen dense, 2Wiki, CE, systems, supporting structured-family, and diagnostic reports.
4. `reports/structured_evidence_check_map.md`: one-page map separating 2Wiki core, HotpotQA/MuSiQue supporting checks, and DocRED/StrategyQA/FEVER boundary checks.
5. `docs/2wiki_data_card.md`: structured-evidence data provenance, released B1 summaries, and controls.
6. `docs/supporting_structured_data_cards.md`: source/split/control cards for supporting and boundary structured checks.
7. `manifests/supporting_structured_checks.json`: machine-readable status map for HotpotQA/MuSiQue/DocRED/StrategyQA/FEVER.
8. `reports/statistical_protocol.md`: interval/repeated-split contract for frozen reports.
9. `docs/legal_router_feature_tiers.md`: compact list of B0/B1 router features and forbidden evaluator-held fields.
10. `docs/cost_profile_changelog.md`: `C_op`, `C_mem`, `C_lat`, lambda, and CE accounting notes.
11. `docs/reproduction_guide.md`: quick/material preflight, lightweight smoke reproduction, full rebuild notes, and custom partial reproduction.
12. `docs/late_interaction_protocol_b_adapter.md`: non-scored ColBERTv2/PLAID-style stage mapping for production late-interaction pipelines.
13. `menus/menu_integrity_report.md`: static view/cost/menu compatibility check.
14. `reports/REPORT_MANIFEST_SHA256.md`: hashes for the headline frozen reports used by the paper.

No execution is required for artifact review. The files above are the intended
static inspection path; commands below are optional smoke checks for reviewers
who want to exercise the validator or public-dev scorer locally.

Paper-facing Protocol B submissions use `schema/`, `menus/`, `manifests/`,
`submission_template/`, and `scripts/validate_protocol_b_route.py`. The older
root-level controlled-cell demo files are retained only for the small legacy
quickstart and are not the paper-facing dense/2Wiki contract.

## Paper-Facing Entry Points

The main paper reports compact tables. The supporting records are organized through stable entry points:

- `schema/route_schema.json`: Protocol B route JSONL schema for dense and 2Wiki evidence-access rows.
- `schema/view_menu_schema.json`: view-menu declaration schema for summary, compressed, ANN, full, CE, hyperlink, context, and portability views.
- `schema/cost_menu_schema.json`: cost-menu/profile schema for `C_op`, `C_mem`, and `C_lat`.
- `schema/public_dev_manifest_schema.json` and `schema/public_dev_reference_schema.json`: row schemas for released method-visible public-dev manifests and evaluator-held public-dev references.
- `menus/`: official dense semantic, 2Wiki structured, and 2Wiki hyperlink view/cost menu declarations.
- `menus/menu_integrity_report.md`: static cross-file check for accepted cost menus, cost-key coverage, hidden-field aliases, and row-level declared-view enforcement.
- `public_dev/`: method-visible public-dev manifests; evaluator-held public-dev references are isolated under `public_dev/evaluator_only/`.
- `manifests/official_slices.json`: machine-readable official/stress/diagnostic status card and canonical report map.
- `manifests/supporting_structured_checks.json`: machine-readable status card for supporting structured-family and boundary checks.
- `scripts/validate_protocol_b_route.py`: standard-library validator for route JSONL rows against a declared view/cost menu.
- `scripts/score_protocol_b_public_dev.py`: public-dev scorer that consumes a route JSONL file and emits raw quality, charged cost, utility, regret, and a leaderboard row.
- `docs/official_benchmark_contract.md`: one-page contract card separating official slices, stress audits, diagnostics, and portability checks.
- `docs/submission_contract.md`: participant-facing contract with valid/invalid rows, visibility tiers, private-test policy, and tool-use limits.
- `docs/new_view_onboarding.md`: checklist for adding new paid evidence views without changing the route schema.
- `docs/2wiki_data_card.md`: public-data provenance, layout, and controls for the 2Wiki structured/hyperlink reports.
- `docs/supporting_structured_data_cards.md`: source/split/control cards for HotpotQA, MuSiQue, DocRED, StrategyQA, and FEVER supporting or boundary checks.
- `docs/legal_router_feature_tiers.md`: which B0/B1 reference-router features are method-visible and which fields remain forbidden.
- `docs/dense_view_taxonomy.md`: compact taxonomy separating compressed representations, ANN-depth controls, full-vector evidence, and reranker views inside the flat dense menu.
- `docs/cost_profile_changelog.md`: cost-profile and CE incremental/cumulative accounting notes.
- `docs/reproduction_guide.md`: four-part reproduction guide: preflight, lightweight smoke, full rebuild, and custom partial reproduction.
- `docs/late_interaction_protocol_b_adapter.md`: non-scored adapter note mapping ColBERTv2/PLAID-style candidate generation, centroid/pruned late interaction, exact interaction, and reranking stages to Protocol B views.
- `docs/provenance_and_genai.md`: artifact provenance, GenAI disclosure, and public-data memorization boundary.
- `scripts/run_learner_family_sweep.py`, `configs/learner_sweep_dense.yaml`, and `configs/learner_sweep_2wiki.yaml`: one-command learner-family diagnostic summaries for dense weak-signal and 2Wiki structured-evidence audits.
- `figure_prompts/`: prompt records for the AI-assisted schematic drafting of paper Figures 1--3; these document visual provenance only, not experimental outputs or scored data.
- `docs/release_checklist.md`: maintainer checklist for anonymous artifact refreshes.
- `reports/PAPER_EVIDENCE_INDEX.md`: map from paper claims to frozen dense, 2Wiki, CE, systems, supporting structured-family, scale-stress, and mechanism reports.
- `reports/structured_evidence_check_map.md`: compact guide to which structured-evidence reports are validated core, supporting generalization checks, or boundary checks.
- `reports/statistical_protocol.md`: compact map of CI/repeated-split semantics across report families.
- `reports/REPORT_MANIFEST_SHA256.md`: checksum manifest for the headline frozen reports.
- `LICENSE` and `DATA_LICENSES.md`: code license and upstream data/model use notes.

Legacy note: the older root-level `submission_schema.json` is retained for the small controlled-cell demo and backward-compatible quickstart. The paper-facing Protocol B route contract is the `schema/` and `menus/` directories plus `scripts/validate_protocol_b_route.py`.

## Reproduction Tiers

For reproduction, start with the four-part guide in
`docs/reproduction_guide.md`.

| Tier | Command | Expected output | Scope |
| --- | --- | --- | --- |
| Quick check / dependency and material preflight | `python scripts/check_reproduction_prereqs.py --install-missing` | `REPRODUCTION PREFLIGHT PASSED` | Checks lightweight deps plus required schema/menu/public-dev/report materials. |
| Lightweight reproduction | `python scripts/run_smoke_reproduction.py` | `SMOKE REPRODUCTION PASSED` plus printed smoke JSON paths | Runs public-dev dense/2Wiki scoring and cost-profile smoke checks; defaults to the system temp directory. |
| Full reproduction | See "Paper Audit Records" below | Heavy reports under `reports/` / `summaries/` | Rebuilds expensive audits; may require GPU and public dataset caches. |
| Custom partial reproduction | `python scripts/run_custom_reproduction.py` | User-selected smoke outputs | Interactive subset of lightweight smoke components. |

Public-dev references are evaluator-held local scoring fixtures. They are
included so reviewers can score route JSONL locally; they are not method-visible
participant inputs and should not be used to train public-dev solvers.

## Protocol B Contract Smoke Test

These commands validate paper-facing JSONL route fixtures without running the
heavy experiments:

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

python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_invalid_leak.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json
```

The first three commands should pass. The final command should fail because it submits an
unpaid CE-score field. This is the quickest reviewer check that the JSONL route
schema, declared views, declared cost profile, and leakage rules agree.

## Paper-Facing Public-Dev Scoring Loop

The paper-facing dense and 2Wiki slices also include lightweight public-dev
submission packs. They are not the full heavy paper audits; they are complete
submit/validate/score fixtures for future methods. A solver receives the
method-visible manifest under `public_dev/`, emits the same route JSONL schema,
and the scorer joins evaluator-held public-dev references to produce raw
quality, charged cost, cost-adjusted utility, regret, and a leaderboard row:

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

By default the scorer writes generated metrics/summary/leaderboard files under
the system temp directory; pass `--out-dir` or explicit output paths to choose
a reviewer-run directory.

Private-test scoring withholds the analogous reference file while preserving
the same input schema, menu ids, scorer ids, and aggregate leaderboard columns.
Public-dev per-row oracle diagnostics are local scorer outputs only and are not
returned as private-test feedback.
The `2wiki_hyperlink` public-dev pack is a structured-stress fixture; the
paper-facing 7GB hyperlink result is the frozen 10k report in `reports/`.

For the 2Wiki routers, B1 features are not private feature engineering. They
are manifest-provided, method-visible summaries such as title/entity overlap
counts, graph/path counts, and context-length or agreement sketches. Future
participants may ignore them, use these generic fields directly, or declare new
low-cost summaries through `docs/new_view_onboarding.md`; evaluator-held
support facts, triples, answers, full-text rankings, and unpaid CE scores remain
hidden.

## Legacy Three-Command Demo

This legacy demo is optional and writes `outputs/` inside the checkout if run.
Use the four-tier paper-facing reproduction path above for clean smoke review.

```bash
pip install -r requirements.txt
python scripts/run_demo.py --solver solvers/template_solver.py
python scripts/evaluate_submission.py outputs/demo_submission.jsonl
```

The demo runs legacy controlled cells through the visibility filter, calls the solver adapter, writes a standardized JSONL submission, scores it with the public evaluator, and emits a leaderboard row under `outputs/`. The paper-facing dense/2Wiki reproduction path is the tiered guide above.

## Submission Kit

Use `challenge_cells.jsonl` and `visibility_manifest.json` as the legacy controlled-cell method-visible submission kit. New methods can edit `submission_template/solver.py` or pass a solver file to `scripts/run_baseline.py` / `scripts/run_demo.py`. `evaluator_only/hidden_reference.jsonl` is a local scoring fixture for the legacy demo, not a participant input for private-test benchmark submission, and must not be imported by solver code.

The files in this section are the legacy controlled-cell quickstart. Dense
semantic access and 2Wiki structured evidence access use the paper-facing
`schema/`, `menus/`, `public_dev/`, and `reports/` entry points listed above.

## Release Packaging

Anonymous release archives should be built from the tracked artifact files
after excluding `.git`, `*_cache` directories, binary caches (`.npy`, `.npz`,
`.pt`, `.pth`), build artifacts, notebook checkpoints, local logs, and raw
streaming caches. Reviewer quickstart commands default to temp directories;
the legacy demo still recreates `outputs/` if run without custom paths.
The release checklist is in
`docs/release_checklist.md`. A clean archive can be produced with:

```bash
python scripts/build_anonymous_release.py --zip
```

Useful checks. The first two use the minimal quickstart dependency set; the
pytest suite uses `requirements-full.txt` or `environment.yml`:

```bash
python scripts/check_pipeline_demo.py
python scripts/check_submission_kit.py --out-dir D:/m2sbench_reviewer_tmp/submission_kit_check
python scripts/verify_systems_profile.py
pip install -r requirements-full.txt
python -m pytest -q tests
```

## Dependencies

`requirements.txt` intentionally contains only `numpy` for the quickstart. `requirements-full.txt` and `environment.yml` list heavier dependencies for extended research utilities.

## Optional Conditional-Flow Baseline

The modern generative/projection baselines reported in the paper are reproducible from repository records:

```bash
pip install -r requirements-full.txt
python scripts/run_conditional_flow_projection_baseline.py
python scripts/run_gmm_projection_baseline_audit.py
python scripts/run_transformer_flow_projection_audit.py
```

These optional scripts train summary-only rectified-flow proposals on the held-in diagnostic-map cells, run a classical three-component GMM summary-prior reconstruction audit, apply exact moment projection, and write `summaries/conditional_flow_projection_summary.csv`, `summaries/gmm_projection_probe/`, and `summaries/transformer_flow_probe/transformer_flow_projection_summary.csv`. They are not part of the three-command reviewer quickstart.

## Learner-Family Sweep Utility

When a Protocol B slice has evaluator-only oracle headroom but weak learned
gains, this utility creates a single learner-family diagnostic table. It is a
diagnostic audit, not a model-selection leaderboard: the goal is to answer
whether changing learner family rescues the slice, which families are stable,
whether complex rankers overfit or over-select cheap views, and whether a weak
legal signal is still present.

```bash
python scripts/run_learner_family_sweep.py \
  --slice dense_semantic \
  --config configs/learner_sweep_dense.yaml \
  --out reports/dense_learner_family_sweep_summary.csv
```

To keep a local checkout unchanged, replace the `--out` path with a scratch
path such as `outputs/dense_learner_family_sweep_summary.csv`; the path above
is the frozen paper-facing summary location.

The CSV includes learner family, utility, delta versus the best fixed view,
95% interval, positive split share, oracle-gap closure, and notes or failure
modes. The dense config reads the released legal and supplemental learner-sweep
JSON reports, so reviewers can inspect the diagnostic without rerunning the
heavy embedding and learner stack. A paired 2Wiki config reads the structured
2k and hyperlink 10k learner-sweep reports, providing the same diagnostic
columns for the stronger structured-evidence regimes.

## Paper Audit Records

The paper-facing robustness checks are retained as scripts plus frozen
summaries. Reviewers normally inspect the frozen Markdown/JSON reports rather
than rerunning these commands. Full rebuilds may require GPU libraries, external
public datasets, or long local runs, and they write new summaries under
`reports/`, `summaries/`, or `outputs/`.

Do not run the commands in this section for smoke-review reproduction. Use the
four reproduction tiers above unless you intentionally want to rebuild heavy
audits from local public dataset caches.

```bash
python scripts/run_cost_model_audit.py
python scripts/plot_cost_free_pareto_frontier.py
python scripts/profile_access_costs.py --reps 200
python scripts/run_diagnostic_terrain_audit.py
python scripts/run_dimension_stress_audit.py
python scripts/run_highdim_manifold_audit.py
python scripts/run_standard_ir_access_audit.py --dataset beir/fiqa/test
python scripts/run_standard_ir_access_audit.py --dataset beir/scifact/test
python scripts/run_standard_ir_access_audit.py --dataset beir/nfcorpus/test
python scripts/run_standard_ir_access_audit.py --dataset beir/arguana
python scripts/run_standard_ir_access_audit.py --dataset antique/test
python scripts/run_evidence_cascade_router_audit.py
python scripts/run_nested_ir_refinement_audit.py --dataset beir/fiqa/test
python scripts/run_nested_ir_refinement_audit.py --dataset beir/scifact/test
python scripts/run_nested_ir_refinement_audit.py --dataset beir/nfcorpus/test
python scripts/run_nested_ir_refinement_audit.py --dataset beir/arguana
python scripts/run_nested_ir_refinement_audit.py --dataset antique/test
python scripts/run_ir_bias_router_summary.py
python scripts/run_ir_router_significance.py
python scripts/run_ir_strict_tie_breakdown.py
python scripts/run_ir_strong_router_headroom.py
python scripts/run_learner_family_sweep.py --slice dense_semantic --config configs/learner_sweep_dense.yaml --out reports/dense_learner_family_sweep_summary.csv
python scripts/run_learner_family_sweep.py --slice 2wiki_structured_evidence --config configs/learner_sweep_2wiki.yaml --out reports/2wiki_learner_family_sweep_summary.csv
python scripts/run_ann_depth_access_audit.py --dataset beir/fiqa/test
python scripts/run_ann_depth_access_audit.py --dataset antique/test
python scripts/run_hnsw_access_audit.py --dataset beir/fiqa/test
python scripts/run_hnsw_access_audit.py --dataset antique/test
python scripts/run_hnsw_access_audit.py --dataset beir/arguana
python scripts/run_hnsw_cascade_router_audit.py --dataset beir/fiqa/test
python scripts/run_hnsw_cascade_router_audit.py --dataset antique/test
python scripts/run_hnsw_cascade_router_audit.py --dataset beir/arguana
python scripts/run_gpu_faiss_docred_evidence_audit.py --max-queries 30000
python scripts/run_ivf_pq_rerank_adapter_audit.py
python scripts/run_large_ivf_pq_rerank_adapter_audit.py
python scripts/run_cross_encoder_reranker_access_audit.py
python scripts/run_compression_evidence_ladder_audit.py
python scripts/run_ce_label_budget_audit.py
python scripts/run_joint_dense_access_menu_audit.py
python scripts/run_unified_systems_profile_audit.py
python scripts/run_2wiki_structured_evidence_audit.py --max-rows 2000
python scripts/run_2wiki_hyperlink_corpus_stress.py --max-rows 10000 --one-hop-cap 16 --two-hop-cap 16
python scripts/run_hotpot_structured_evidence_audit.py --data-dir "$M2SBENCH_HOTPOT_DIR" --max-rows 5000 --no-ce
python scripts/run_musique_structured_evidence_audit.py --data-dir "$M2SBENCH_MUSIQUE_DIR" --max-rows 10000 --no-ce
python scripts/run_strategyqa_reasoning_evidence_audit.py --data-dir "$M2SBENCH_STRATEGYQA_DIR" --max-rows 2000
python scripts/run_fever_claim_evidence_audit.py --data-path "$M2SBENCH_FEVER_TRAIN_JSONL" --max-rows 100
python scripts/run_semantic_summary_budget_expansion_audit.py
python scripts/run_contextual_prior_flow_audit.py
python scripts/run_log_proxy_context_flow_audit.py
python scripts/run_gmm_projection_baseline_audit.py
python scripts/run_action_contract_audit.py
python scripts/run_static_overfit_audit.py
```

These rebuild or validate `summaries/cost_model_audit/`, `summaries/diagnostic_terrain_audit/`, `summaries/dimension_stress_audit/`, `summaries/ivf_pq_rerank_adapter/`, `summaries/ivf_pq_rerank_large_adapter/`, `summaries/contextual_prior_flow_probe/`, `summaries/log_proxy_context_flow_probe/`, `summaries/gmm_projection_probe/`, `summaries/action_contract_audit/`, and `summaries/static_overfit_audit/`; `profile_access_costs.py` writes local advisory files under `outputs/`. The 2Wiki hyperlink-corpus stress expects the public 2Wiki/HF data and 7GB paragraph corpus described in `docs/2wiki_data_card.md`; the frozen 10k report is preferred for review. The HotpotQA script expects a local HuggingFace `load_from_disk` distractor export and is retained as a supporting structured-family check, not a separate official slice. The MuSiQue script expects a local HuggingFace `load_from_disk` export and is retained as a decomposition-centric supporting structured-family replication, not a separate official slice. The StrategyQA script expects a local HuggingFace `load_from_disk` export and is retained as an exploratory reasoning-frontier check, not a validation row. The FEVER script expects the official train JSONL and is retained only as a stopped-after-smoke claim-evidence handle check because claim/title shortcuts dominate. The cost audit records fixed relative operation units, cost-free Pareto checks, switch thresholds, and perturbation stability; `plot_cost_free_pareto_frontier.py` renders the paper figure showing observed richer-view domination before access cost is applied. The optional profiler maps those units to local latency without changing leaderboard scores. The dimension audits include both ambient full-covariance stress checks and a two-sided high-dimensional audit: full-rank Gaussian controls mark the expected degeneration boundary, while low-rank manifold/mixture embeddings test whether non-Gaussian structure remains hidden by the same compact summary. The IVF-PQ rerank adapters map text vectors to coarse index cells, PQ reconstruction, and full-vector rerank views; the larger audit reports 159,040 queries over 659 IVF-style cells and retains only frozen aggregates plus a per-query sample. `run_cross_encoder_reranker_access_audit.py` is an optional GPU audit that treats a cross-encoder over a fixed bi-encoder candidate pool as an expensive Protocol B view and writes fixed, oracle, and selective-route records. `run_compression_evidence_ladder_audit.py` expands the semantic menu into binary, PQ, truncated, int8, full-dense, and cross-encoder views; `run_ce_label_budget_audit.py` audits surrogate-light CE-purchase features and small CE-label budgets. `run_joint_dense_access_menu_audit.py` puts summary, binary, PQ, int8, HNSW, full-dense, and CE views into one FiQA Protocol B menu and evaluates fixed, restricted, expanded-utility, and oracle solvers. `run_unified_systems_profile_audit.py` records the view-dependency parents, incremental and cumulative costs, measured p50/p95/p99 latency, latency-derived QPS, bytes touched, index/model footprint, and operation/memory/latency profile costs used by the systems-profile table. `run_2wiki_structured_evidence_audit.py` builds a structured multi-hop QA evidence slice where title sketches, 1-hop hyperlinks, 2-hop paths, full context, and CE are paid views scored against evaluator-held 2Wiki supporting facts and evidence triples; it also reports shuffled-link, degree-random, random-path controls and legal B0/B1 feature-tier learner ablations. `run_hotpot_structured_evidence_audit.py` applies the same evidence-purchase readout to HotpotQA distractor contexts using hidden supporting facts plus shuffled/random controls. `run_musique_structured_evidence_audit.py` applies the readout to MuSiQue decomposition questions and paragraph evidence while keeping support labels, answers, and decomposition answers evaluator-only. `run_strategyqa_reasoning_evidence_audit.py` applies a minimal readout to StrategyQA question decompositions and supporting facts while keeping facts and answers evaluator-only. The contextual-prior flow audits simulate declared side-information budgets while keeping hidden targets evaluator-only: `run_contextual_prior_flow_audit.py` conditions on a coarse diagnostic-region label, and `run_log_proxy_context_flow_audit.py` uses continuous log-derived proxy features (spectral/count/density features plus held-in nearest-neighbor history aggregates) to mimic query-log/cache priors without exposing evaluation-cell targets. The GMM projection audit is a classical summary-prior baseline: it samples target-blind three-component moment-matched mixture proposals and applies exact moment projection on the shared 50-cell probe. The action-contract audit defines the white-box topology, geometry, routing, and label-free fidelity tasks used by the paper's cost-adjusted utility discussion. The static-overfit audit treats the released 320 cells as a public development/review slice and records bootstrap, private-seed 320-cell, and constructor-slice checks.

`reports/semantic_summary_budget_expansion_audit.*` is generated by `scripts/run_semantic_summary_budget_expansion_audit.py` and combines cached semantic access records into the summary-budget expansion, cost-based optimizer, set-valued access, graph/KG mini-track, and sanity-control tables. The cross-encoder reranker audit also writes `reports/cross_encoder_access_boundary_*`, which records the cheap-only CE-purchase adapter trace used for the access-boundary figure. `reports/compression_evidence_ladder_*` and `reports/ce_label_budget_*` contain the compression-aware evidence ladder and CE-label-budget stress rows. `reports/joint_dense_access_menu_*` contains the one-menu FiQA dense-access audit, `reports/unified_systems_profile_audit.*` contains the reproducible systems-profile calibration, and `reports/2wiki_structured_evidence_*` contains the structured multi-hop evidence-access slice. `reports/hotpot_structured_evidence.*` and `reports/musique_structured_evidence.*` contain supporting structured-family checks with hidden support labels and shuffled/random controls. `reports/strategyqa_reasoning_evidence.*` contains an exploratory noisy reasoning-frontier check where decomposition-only is already strong. `reports/fever_claim_evidence.*` contains a stopped-after-smoke FEVER claim-evidence handle check where claim-only title matching already dominates. `reports/docred_structured_evidence.*` is a supporting candidate structured-evidence check: real DocRED entity/co-mention evidence beats shuffled/random controls, but the legal router does not improve over the best fixed co-mention view, so it is not used as a main-paper validation row.
`reports/2wiki_lambda_sensitivity.*` rescales frozen 2Wiki fixed/adaptive policy rows over a small lambda grid; it is intended as a reviewer-facing sensitivity check, not a rerun of the 7GB corpus or a retraining sweep.

For paper-review navigation, `docs/official_benchmark_contract.md` is the
shortest official-slice map. The following pointer files collect the relevant
frozen reports without duplicating the large JSON records:

- `reports/dense_joint_menu.md`
- `reports/dense_legal_learner_sweep.md` (supporting dense learner sweep; not a headline paper table)
- `reports/dense_additional_learner_sweep.md` (linear/rank/QPP-rule learner sensitivity; not a headline paper table)
- `reports/dense_learner_family_sweep_summary.csv` (one-command diagnostic summary across positive and negative learner-family rows)
- `reports/2wiki_structured_evidence.md`
- `reports/2wiki_7gb_hyperlink_stress.md`
- `reports/ce_purchase.md`
- `reports/cost_profiles.md`
- `reports/hidden_test_probe.md`
- `reports/structured_evidence_check_map.md` (core/supporting/boundary map for structured-evidence reports)
- `docs/legal_router_feature_tiers.md` (method-visible B0/B1 feature tiers and forbidden fields)
- `docs/cost_profile_changelog.md` (profile-relative costs and CE accounting)
- `reports/hotpot_structured_evidence.md` (supporting structured-family check, not a separate official slice)
- `reports/musique_structured_evidence.md` (supporting structured-family replication, not a separate official slice)
- `reports/strategyqa_reasoning_evidence.md` (exploratory reasoning-frontier check, not a validation row)
- `reports/fever_claim_evidence.md` (stopped-after-smoke claim-evidence handle check, not a validation row)
- `reports/docred_structured_evidence.md` (supporting candidate check, not a main validation row)

`reports/ce_purchase.md` is the reviewer entry point for the expensive-model-call stress audit: the cross-encoder is treated as a paid view, not a free teacher or an end-to-end RAG scorer.

The nested refinement reports (`reports/nested_ir_refinement_*.json/.md`) fix a full-flat top-200 candidate pool for each query, then rerank that same pool with centroid, PQ-reconstructed, and full-vector evidence. They are included to separate access-value effects from candidate-pool artifacts in the view-specific operator scorecard. `reports/ir_bias_router_summary.*` summarizes qrel sparsity, exact PQ/full NDCG ties, strict PQ wins/losses, and held-out QPP/RF/cascade router baselines; `reports/ir_router_significance.*` adds paired bootstrap intervals for router utility differences versus fixed full; `reports/ir_strict_tie_breakdown.*` reports per-dataset strict/tie/loss decompositions used in the paper scorecard; `reports/ir_strong_router_headroom.*` adds a pooled held-out oracle-route upper bound and stronger tree-based router references using method-visible B1 features. `reports/dense_legal_learner_sweep.*` is a supporting dense learner sweep over XGB/GBDT/LightGBM/CatBoost/TabNet/MLP/ensemble-style alternatives; `reports/dense_additional_learner_sweep.*` adds linear, LambdaMART, pairwise RankNet, and QPP/rule alternatives. Together they document small stable QPP/tree/ensemble gains, near-baseline CatBoost and some linear behavior, and neural/rank-learning underperformance, so they support the weak-signal interpretation rather than replacing the paper's main dense readout. `reports/evidence_cascade_router_audit.*` records the pooled evidence-tier cascade that uses only summary/PQ scores and result-list agreement before deciding whether to pay for full reranking, with repeated-split stability and leave-one-dataset-out stress rows. `reports/hnsw_cascade_router_*` records shallow-HNSW cascade routers and repeated-split comparisons against the best fixed HNSW depth.

## Optional Bloomberg Adapter

The optional licensed Bloomberg adapter is not required for public reproduction. Its scripts are kept under `optional_adapters/bloomberg_protocol_b_adapter/`; returned Bloomberg data are not included in this repository. Users with Bloomberg access should install `blpapi` from Bloomberg's official SDK or package index rather than relying on redistributed vendor data.


