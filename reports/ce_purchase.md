# Cross-Encoder Purchase Report Pointer

Canonical frozen reports:

- `reports/ce_decision_curve_audit.md`
- `reports/ce_decision_curve_audit.json`
- `reports/ce_label_budget_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/ce_purchase_calibration_metrics.json`
- `reports/ce_purchase_accounting_context.json`
- `reports/cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.csv`
- `reports/cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.png`

Purpose: supports the reranker-as-paid-view / expensive-model-call analysis. CE is a concrete reproducible proxy for the production question of when to invoke a costly reranker or LLM-style scoring model. CE scores are hidden before purchase; cheap-only gates are evaluated by buy rate, false-buy cost, utility, regret, calibration, and oracle headroom. This is not an end-to-end RAG or generation-quality claim.

Cost accounting: CE purchase reports use the incremental CE-purchase profile
(`ce_incremental=0.45`) after a first-stage candidate pool already exists. The
dense ladder cost menu charges `ce=1.03` when CE is treated as the cumulative
richest text-reranking view. The two contexts are intentionally reported
separately.
