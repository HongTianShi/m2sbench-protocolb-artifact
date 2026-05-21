# Paper Evidence Index

This index maps the main paper claims to frozen artifact records. The main paper intentionally reports compact tables; the full JSON/Markdown records remain here for audit.

For quick traceability, `reports/REPORT_MANIFEST_SHA256.md` records hashes for the headline reports and paper-facing contract files listed below.

Structured-evidence reports are summarized in `reports/structured_evidence_check_map.md`. That file is the fastest way to distinguish the 2Wiki validated core, HotpotQA/MuSiQue supporting generalization checks, and DocRED/StrategyQA/FEVER boundary checks.
Legal B0/B1 router features are summarized in `docs/legal_router_feature_tiers.md`; profile-relative costs and CE accounting are summarized in `docs/cost_profile_changelog.md`.
Interval and repeated-split semantics are summarized in `reports/statistical_protocol.md`.
The compact repeated-split audit index and CSV pointers are in
`reports/repeated_split_audit_index.md`.

## Validated Dense Semantic Access

Scope badge: dense semantic access is the validated IR-facing core and a
weak-signal stress regime. Standard-qrel rows span five public qrel datasets;
the joint dense menu is a focused FiQA boundary audit, and its repeated-split
learner interval includes zero. The broader learner-family sweep supports only
the modest claim that legal compact features contain small but stable signal.

- `reports/protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/evidence_cascade_router_audit.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_mem__lambda_0p08.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_lat__lambda_0p08.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_op__lambda_0p04.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_op__lambda_0p16.md`
- `reports/ir_router_significance.md`
- `reports/ir_strong_router_headroom.md`
- `reports/dense_legal_learner_sweep.md`
- `reports/dense_additional_learner_sweep.md`
- `reports/dense_learner_family_sweep_summary.csv`
- `scripts/run_learner_family_sweep.py` with `configs/learner_sweep_dense.yaml`
- `reports/hnsw_cascade_router_beir_fiqa_test.md`
- `reports/hnsw_cascade_router_antique_test.md`
- `reports/hnsw_cascade_router_beir_arguana.md`

Readout: dense learner gains are modest and treated as stress evidence; the key artifact claim is comparable route scoring under the same visibility/cost contract. The supporting learner sweeps check XGB/GBDT/LightGBM/CatBoost/TabNet/MLP/ensemble, linear, LambdaMART, pairwise RankNet, and QPP/rule alternatives. They find the same pattern: shallow QPP/tree/ensemble learners give small but stable positive gains, CatBoost and some linear variants stay near fixed full or modestly positive, and MLP/TabNet/rank-learning variants can underperform.
The one-command learner-family utility rebuilds a unified diagnostic CSV from these reports; it is explicitly a diagnostic audit, not a model-selection leaderboard.

## Compression-Aware Dense Ladder

- `reports/compression_evidence_ladder_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/unified_systems_profile_audit.md`

Readout: binary, PQ, int8, full dense, HNSW, and CE views form a storage-latency-quality menu; raw-best and utility-best views differ under declared profiles.

## Reranker as Paid View

- `reports/ce_decision_curve_audit.md`
- `reports/ce_purchase_calibration_metrics.json`
- `reports/cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.csv`
- `reports/cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.png`
- `reports/ce_label_budget_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/cross_encoder_reranker_access_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.md`

Readout: this is the expensive-model-call stress audit. Fixed CE improves raw quality but can overpay after cost; cheap-only gates trade buy rate, false-buy cost, and residual oracle headroom. These reports use the incremental CE-purchase accounting context (`ce_incremental=0.45` after a first-stage pool exists), not the cumulative dense-ladder CE cost (`ce=1.03`).

## Validated 2Wiki Structured Evidence Acquisition

- `reports/2wiki_structured_evidence_2000.md`
- `reports/2wiki_structured_evidence_2000.json`
- `reports/2wiki_hyperlink_corpus_stress_10000.md` (canonical 7GB stress row used in the paper)
- `reports/2wiki_hyperlink_corpus_stress_10000.json`
- `reports/2wiki_hyperlink_systems_profile_10000.md` (canonical 7GB systems/materialization profile)
- `reports/2wiki_hyperlink_systems_profile_10000.json`
- `reports/2wiki_lambda_sensitivity.md` (aggregate rescore over `lambda` values from frozen 2Wiki policy rows)
- `reports/2wiki_lambda_sensitivity.json`
- `reports/2wiki_structured_learner_family_sweep.md`
- `reports/2wiki_structured_learner_family_sweep.json`
- `reports/2wiki_hyperlink_learner_family_sweep.md`
- `reports/2wiki_hyperlink_learner_family_sweep.json`
- `reports/2wiki_learner_family_sweep_summary.csv`
- `scripts/run_learner_family_sweep.py` with `configs/learner_sweep_2wiki.yaml`
- `reports/2wiki_hyperlink_corpus_stress_5000.md`
- `reports/2wiki_hyperlink_corpus_stress_5000.json`
- `reports/2wiki_hyperlink_corpus_stress_2000.md`
- `reports/2wiki_hyperlink_corpus_stress_2000.json`
- `reports/2wiki_hyperlink_systems_profile_2000.md`
- `reports/2wiki_hyperlink_systems_profile_2000.json`
- `reports/2wiki_hyperlink_systems_profile_5000.md`
- `reports/2wiki_hyperlink_systems_profile_5000.json`
- `reports/overnight_mainline_summary.md`
- `docs/2wiki_data_card.md`

Readout: real hyperlink views beat shuffled/random controls; B1 title/entity/path/context sketches improve routing without exposing support facts, answers, or unpaid CE scores. The 10k hyperlink-corpus run is the paper-facing robustness number. The paired learner-family sweep shows that the same diagnostic wrapper that yields limited dense recovery recovers a larger fraction of oracle value in 2Wiki structured and hyperlink regimes, with the top learned rows positive on every repeated split. The 5k and 2k runs are retained as consistency checks.

Paper headline rows: the provided-context 2Wiki slice maps to
`reports/2wiki_structured_evidence_2000.md` fixed 1-hop `.900` versus B1
context router `.924`. The 7GB hyperlink stress maps to
`reports/2wiki_hyperlink_corpus_stress_10000.md` fixed 1-hop `.791` versus
valid adaptive `.819`.

## Supporting Structured-Family Generalization Checks

Machine-readable supporting status cards are in
`manifests/supporting_structured_checks.json`; source/split/control notes are in
`docs/supporting_structured_data_cards.md`.

- `reports/structured_evidence_check_map.md`
- `reports/hotpot_structured_evidence.md`
- `reports/hotpot_structured_evidence_5000_noce.md`
- `reports/hotpot_structured_evidence_2000_noce.md`
- `reports/hotpot_structured_evidence_2000_ce.md`
- `reports/musique_structured_evidence.md`
- `reports/musique_structured_evidence_10000_noce.md`
- `reports/musique_structured_evidence_5000_noce.md`
- `reports/musique_structured_evidence_2000_noce.md`

Readout: HotpotQA distractor provides a second public multi-hop QA family for the same evidence-purchase pattern, but it is not a separate official leaderboard slice in the current paper. In the 5k no-CE check, real paragraph/sentence evidence beats shuffled/random controls, the best legal adaptive router improves over the best fixed paragraph view, and oracle headroom remains. The CE report is retained as an expensive-view boundary record because CE has oracle share but can overpay under the declared cost.

MuSiQue provides a decomposition-centric replication check, also outside the official core. In the 10k no-CE check, real paragraph evidence beats shuffled/random paragraph controls, the best legal adaptive router improves over the best fixed paragraph view (+.0153 repeated adaptive-minus-fixed, 95% interval [.0116,.0186], all repeats positive), and oracle headroom remains.

## Systems and Cost Profiles

- `reports/unified_systems_profile_audit.md`
- `reports/unified_systems_profile_audit.json`
- `reports/semantic_access_cost_profile.md`
- `docs/cost_profile_changelog.md`
- `docs/cost_profile_derivation.md`
- `docs/cost_profile_derivation.json`
- `menus/dense_semantic.C_mem.cost_menu.json`
- `menus/dense_semantic.C_lat.cost_menu.json`
- `reports/2wiki_hyperlink_systems_profile_2000.md`
- `reports/2wiki_hyperlink_systems_profile_5000.md`
- `reports/2wiki_hyperlink_systems_profile_10000.md`

Readout: `C_op`, `C_mem`, and `C_lat` can choose different fixed winners, so Protocol B stores raw quality, declared cost, latency, throughput, bytes, and profile ids separately.

## Scale and Mechanism Diagnostics

- `reports/large_ivf_pq_rerank_adapter_audit.md`
- `reports/semantic_summary_budget_expansion_audit.md`
- `reports/conditional_flow_projection_audit.json`
- `reports/gmm_projection_baseline_audit.json`
- `reports/transformer_flow_projection_audit.json`
- `reports/proxy_diagnostic_routing_audit.md`

Readout: IVF-PQ is scale stress rather than semantic-qrel validation; controlled matched-summary cells explain why cheap summaries can hide decision-relevant structure.

## Supporting Candidate Structured Checks

- `reports/docred_structured_evidence.md`
- `reports/docred_structured_evidence_noce.md`
- `reports/docred_structured_evidence_ce.md`
- `reports/strategyqa_reasoning_evidence.md`
- `reports/strategyqa_reasoning_evidence_2000_noce.md`
- `reports/fever_claim_evidence.md`
- `reports/fever_claim_evidence_100_noce.md`

Readout: DocRED exact evidence annotations provide a second structured-evidence family check, but it is not a main-paper validation row. Real entity/co-mention evidence beats shuffled/random controls, while legal routers do not improve over the best fixed co-mention view; the result is retained to document a boundary case rather than to strengthen the main claim.

StrategyQA is retained as an exploratory reasoning-frontier check rather than a structured-evidence validation row. Shuffled/cross-question/random fact controls collapse, but the released decomposition sketch is already strong; the 2k run has only a small adaptive-minus-fixed gain (+.0039, 95% interval [.0005,.0072]).

FEVER is retained only as a stopped-after-smoke claim-evidence handle check. Real title/evidence handles beat shuffled, wrong-evidence, random, and high-frequency-page controls, but claim-only title matching already reaches .9883 utility on the 100-row smoke run; adaptive routing is slightly below the best fixed summary route and repeated adaptive-minus-fixed is -.0004 with 95% interval [-.0309,+.0334]. It is therefore not used as a paper validation row.

## Submission Contract and Governance

- `schema/route_schema.json`
- `schema/view_menu_schema.json`
- `schema/cost_menu_schema.json`
- `menus/menu_integrity_report.md`
- `menus/menu_integrity_report.json`
- `menus/dense_semantic.view_menu.json`
- `menus/dense_semantic.cost_menu.json`
- `menus/2wiki_structured.view_menu.json`
- `menus/2wiki_structured.cost_menu.json`
- `menus/2wiki_hyperlink.view_menu.json`
- `menus/2wiki_hyperlink.cost_menu.json`
- `manifests/official_slices.json`
- `scripts/validate_protocol_b_route.py`
- `scripts/score_protocol_b_public_dev.py`
- `public_dev/README.md`
- `public_dev/dense_semantic.manifest.jsonl`
- `public_dev/evaluator_only/dense_semantic.reference.jsonl`
- `public_dev/2wiki_structured.manifest.jsonl`
- `public_dev/evaluator_only/2wiki_structured.reference.jsonl`
- `public_dev/2wiki_hyperlink.manifest.jsonl`
- `public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl`
- `docs/submission_contract.md`
- `docs/official_benchmark_contract.md`
- `docs/new_view_onboarding.md`
- `docs/late_interaction_protocol_b_adapter.md`
- `docs/provenance_and_genai.md`
- `docs/release_checklist.md`
- `reports/statistical_protocol.md`
- `ARTIFACT_INVENTORY.md`
- `REVIEWER_QUICKSTART.md`

Readout: the benchmark is an executable Protocol B contract with declared visibility, hidden fields, cost menus, validators, public-dev rows, and private-test policy.
