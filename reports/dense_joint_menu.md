# Dense Joint Menu Report Pointer

Canonical frozen report:

- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.json`

Purpose: supports the paper row where summary, binary, PQ, int8, HNSW16/64, full dense, and CE are placed in one Protocol B menu. It is used to show that adjacent IR pipelines can be evaluated as restricted solvers under one hidden-qrel, paid-view, cost-regret contract.

For the representation/ANN-depth/full-vector/reranker taxonomy used by this
flat menu, see `docs/dense_view_taxonomy.md`.

Dense access is a weak-signal stress regime, not a dense-router SOTA claim. The
single FiQA joint-menu report is profile- and split-sensitive; the broader
standard-qrel and learner-family reports below document small but stable legal
tree/QPP gains while leaving most oracle headroom unresolved.

## Dense Evidence Index

| Purpose | Report(s) |
| --- | --- |
| Canonical one-menu FiQA audit | `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`; `.json` |
| Secondary profile rescoring | `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_mem__lambda_0p08.md`; `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_lat__lambda_0p08.md` |
| Lambda sensitivity examples | `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_op__lambda_0p04.md`; `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__C_op__lambda_0p16.md` |
| Standard-qrel restricted solvers and hidden-test probe | `reports/protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.md`; `.json` |
| Dense learner-family sensitivity | `reports/dense_legal_learner_sweep.md`; `reports/dense_additional_learner_sweep.md` |
| Cost/profile derivation | `docs/cost_profile_derivation.md`; `docs/cost_profile_derivation.json`; `reports/unified_systems_profile_audit.md`; `.json` |

All rows use the same Protocol B idea: a method-visible compact state plus a
declared paid-view menu. Fixed views, QPP gates, cascades, ANN-depth gates,
CE-purchase gates, and utility predictors all emit route JSONL and are scored
under declared cost/regret profiles.
