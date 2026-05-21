# Dense Additional Learner Sweep

Artifact-only supplemental sweep for linear, rank-learning, and QPP/rule learner families. It uses the same dense Protocol B split and legal compact feature budget as the main dense learner-family report, and is included to document attempted alternatives rather than to propose a new router.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Lambda: 0.08
- Seed: 31
- Repetitions: 20
- Families: linear, rank, rules
- Split protocol: 20 repeated stratified 60/20/20 splits within each of five standard-qrel datasets
- Feature budget: B1 legal compact features: dataset id, query embedding stats/head, text stats, summary/PQ score stats, qrel-free deltas

Takeaway: shallow QPP/rule ensembles and simple multi-output linear models recover small positive signal, while rank-learning objectives and calibrated/logistic variants can underperform by over-selecting cheaper views. This supports the paper's weak-signal dense interpretation.

| model | utility | regret | cost | delta vs fixed full | positive splits | gap closed | choices | status / rule note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| oracle | 0.4241 | 0.0000 | 0.253 | +0.0754 [+0.0731,+0.0782] | 1.00 | 1.000 | summary 0.33, pq 0.36, full 0.31 | ok |
| top5_rule_vote | 0.3550 | 0.0691 | 0.422 | +0.0064 [+0.0051,+0.0076] | 1.00 | 0.084 | pq 0.42, full 0.58 | rule_pq_std_ge_0.0401_then_pq_else_full; rule_pq_std_le_0.0401_then_full_else_pq; rule_pq_std_ge_0.0372_then_pq_else_full |
| top9_rule_vote | 0.3546 | 0.0695 | 0.422 | +0.0059 [+0.0047,+0.0072] | 1.00 | 0.079 | pq 0.42, full 0.58 | rule_pq_std_ge_0.0401_then_pq_else_full; rule_pq_std_le_0.0401_then_full_else_pq; rule_pq_std_ge_0.0372_then_pq_else_full |
| best_single_qpp_rule | 0.3544 | 0.0696 | 0.426 | +0.0058 [+0.0046,+0.0070] | 1.00 | 0.077 | pq 0.41, full 0.59 | rule_pq_std_ge_0.0401_then_pq_else_full |
| top3_rule_vote | 0.3544 | 0.0696 | 0.426 | +0.0058 [+0.0046,+0.0070] | 1.00 | 0.077 | pq 0.41, full 0.59 | rule_pq_std_ge_0.0401_then_pq_else_full; rule_pq_std_le_0.0401_then_full_else_pq; rule_pq_std_ge_0.0372_then_pq_else_full |
| elasticnet_multioutput_linear | 0.3524 | 0.0717 | 0.428 | +0.0038 [+0.0023,+0.0051] | 0.80 | 0.050 | summary 0.04, pq 0.34, full 0.62 | ok |
| lasso_multioutput | 0.3522 | 0.0719 | 0.428 | +0.0035 [+0.0021,+0.0049] | 0.80 | 0.047 | summary 0.03, pq 0.35, full 0.62 | ok |
| ridge_multioutput_linear | 0.3521 | 0.0719 | 0.425 | +0.0035 [+0.0023,+0.0046] | 0.90 | 0.046 | summary 0.04, pq 0.35, full 0.61 | ok |
| best_fixed_test | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| fixed_full | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| ridge_expanded_linear | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| lasso_expanded | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| elasticnet_expanded_linear | 0.3486 | 0.0754 | 0.580 | +0.0000 [+0.0000,+0.0000] | 0.00 | 0.000 | full 1.00 | ok |
| meta_qpp_margin_cascade | 0.3475 | 0.0765 | 0.414 | -0.0011 [-0.0029,+0.0005] | 0.40 | worse | summary 0.05, pq 0.36, full 0.59 | S-margin ge 0.277; PQ-margin ge 0.044 |
| calibrated_ridge_isotonic | 0.3427 | 0.0814 | 0.498 | -0.0059 [-0.0078,-0.0041] | 0.05 | worse | pq 0.22, full 0.78 | ok |
| xgboost_pairwise_ranker | 0.3290 | 0.0951 | 0.234 | -0.0196 [-0.0227,-0.0167] | 0.00 | worse | summary 0.08, pq 0.79, full 0.13 | ok |
| calibrated_logistic_oracle_view | 0.3267 | 0.0974 | 0.272 | -0.0220 [-0.0260,-0.0178] | 0.00 | worse | summary 0.21, pq 0.48, full 0.30 | ok |
| lightgbm_lambdamart_ranker | 0.3254 | 0.0987 | 0.255 | -0.0232 [-0.0267,-0.0199] | 0.00 | worse | summary 0.22, pq 0.52, full 0.26 | ok |
| fixed_pq | 0.3201 | 0.1040 | 0.200 | -0.0285 [-0.0314,-0.0256] | 0.00 | worse | pq 1.00 | ok |
| logistic_oracle_view | 0.3093 | 0.1148 | 0.237 | -0.0393 [-0.0436,-0.0350] | 0.00 | worse | summary 0.33, pq 0.40, full 0.27 | ok |
| pairwise_ranknet | 0.3070 | 0.1170 | 0.278 | -0.0416 [-0.0478,-0.0358] | 0.00 | worse | summary 0.33, pq 0.30, full 0.38 | ok |
| fixed_summary | 0.1352 | 0.2889 | 0.000 | -0.2134 [-0.2174,-0.2093] | 0.00 | worse | summary 1.00 | ok |
