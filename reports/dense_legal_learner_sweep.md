# Dense Legal Learner Sweep

Artifact-only supporting dense-access learner-family sweep over the five standard-qrel datasets. This report is not a headline paper table and does not propose a routing method; it tests learner-family sensitivity under the same legal compact features and repeated split protocol as the main dense semantic-access audit.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Lambda: 0.08
- Seed: 31
- Repetitions: 20
- Split protocol: 20 repeated stratified 60/20/20 splits within each of five standard-qrel datasets; all rows use the same B1-legal compact feature budget as the dense semantic-access audit
- Feature budget: B1 legal: dataset id, query embedding stats/head, text length stats, summary score stats, PQ score stats, score deltas

Takeaway: legal compact features contain a small but repeatable dense-access signal for tree/ensemble-style learners. CatBoost is effectively tied with fixed full, and MLP/TabNet underperform, so dense remains a weak-signal stress regime rather than a solved routing task.

Related supplemental sweep: `reports/dense_additional_learner_sweep.*` covers linear, calibrated/logistic, LambdaMART, pairwise RankNet, and QPP/rule alternatives under the same split protocol.

| model | utility | regret | cost | delta vs fixed full | positive splits | oracle gap closed | choices | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| oracle | 0.4241 | 0.0000 | 0.253 | +0.0754 [+0.0732,+0.0781] | 1.00 | 1.000 | summary 0.33, pq 0.36, full 0.31 | ok |
| rf_expanded | 0.3552 | 0.0688 | 0.421 | +0.0066 [+0.0047,+0.0085] | 0.95 | 0.087 | summary 0.03, pq 0.37, full 0.60 | ok |
| extra_trees_multioutput | 0.3549 | 0.0692 | 0.434 | +0.0062 [+0.0049,+0.0076] | 0.95 | 0.083 | summary 0.01, pq 0.37, full 0.62 | ok |
| dev_best_single_model | 0.3548 | 0.0693 | 0.439 | +0.0062 [+0.0047,+0.0076] | 1.00 | 0.082 | summary 0.02, pq 0.34, full 0.64 | ok |
| rf_multioutput | 0.3548 | 0.0693 | 0.441 | +0.0061 [+0.0048,+0.0075] | 1.00 | 0.081 | pq 0.35, full 0.64 | ok |
| dev_top3_ensemble | 0.3547 | 0.0694 | 0.446 | +0.0060 [+0.0044,+0.0076] | 0.90 | 0.080 | summary 0.01, pq 0.33, full 0.65 | ok |
| extra_trees_expanded | 0.3542 | 0.0698 | 0.425 | +0.0056 [+0.0042,+0.0069] | 0.95 | 0.074 | summary 0.02, pq 0.37, full 0.60 | ok |
| dev_top5_ensemble | 0.3541 | 0.0700 | 0.451 | +0.0055 [+0.0038,+0.0070] | 0.95 | 0.073 | summary 0.01, pq 0.32, full 0.67 | ok |
| knn_expanded | 0.3535 | 0.0706 | 0.463 | +0.0048 [+0.0030,+0.0067] | 0.85 | 0.064 | summary 0.04, pq 0.25, full 0.71 | ok |
| gbdt_expanded | 0.3519 | 0.0722 | 0.492 | +0.0032 [+0.0020,+0.0044] | 0.75 | 0.043 | pq 0.23, full 0.77 | ok |
| lightgbm_expanded | 0.3516 | 0.0725 | 0.490 | +0.0029 [+0.0017,+0.0042] | 0.90 | 0.039 | pq 0.23, full 0.77 | ok |
| hist_gbdt_expanded | 0.3514 | 0.0727 | 0.468 | +0.0028 [+0.0015,+0.0039] | 0.85 | 0.036 | summary 0.02, pq 0.26, full 0.72 | ok |
| kernel_ridge_rbf_expanded | 0.3511 | 0.0729 | 0.429 | +0.0025 [+0.0012,+0.0038] | 0.70 | 0.033 | summary 0.03, pq 0.35, full 0.62 | ok |
| xgb_expanded | 0.3509 | 0.0732 | 0.523 | +0.0022 [+0.0016,+0.0029] | 0.95 | 0.029 | pq 0.15, full 0.85 | ok |
| catboost_expanded | 0.3491 | 0.0750 | 0.553 | +0.0005 [-0.0001,+0.0011] | 0.55 | 0.006 | pq 0.07, full 0.93 | ok |
| fixed_full | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| best_fixed_test | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| ridge_expanded | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| elasticnet_expanded | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| adaboost_expanded | 0.3483 | 0.0758 | 0.343 | -0.0003 [-0.0022,+0.0016] | 0.45 | worse | pq 0.62, full 0.38 | ok |
| hgb_multioutput | 0.3447 | 0.0793 | 0.398 | -0.0039 [-0.0062,-0.0016] | 0.25 | worse | summary 0.06, pq 0.38, full 0.55 | ok |
| svr_rbf_expanded | 0.3447 | 0.0794 | 0.393 | -0.0039 [-0.0059,-0.0017] | 0.20 | worse | summary 0.08, pq 0.37, full 0.55 | ok |
| tabnet_expanded | 0.3387 | 0.0854 | 0.452 | -0.0099 [-0.0152,-0.0056] | 0.20 | worse | summary 0.02, pq 0.31, full 0.67 | ok |
| mlp_expanded | 0.3363 | 0.0878 | 0.376 | -0.0124 [-0.0150,-0.0100] | 0.00 | worse | summary 0.13, pq 0.34, full 0.53 | ok |
| mlp_multioutput | 0.3361 | 0.0880 | 0.376 | -0.0126 [-0.0149,-0.0101] | 0.00 | worse | summary 0.10, pq 0.38, full 0.52 | ok |
| fixed_pq | 0.3201 | 0.1040 | 0.200 | -0.0285 [-0.0314,-0.0256] | 0.00 | worse | pq 1.00 | ok |
| decision_tree_expanded | 0.2962 | 0.1279 | 0.187 | -0.0524 [-0.0565,-0.0482] | 0.00 | worse | summary 0.26, pq 0.64, full 0.10 | ok |
| fixed_summary | 0.1352 | 0.2889 | 0.000 | -0.2134 [-0.2175,-0.2090] | 0.00 | worse | summary 1.00 | ok |
