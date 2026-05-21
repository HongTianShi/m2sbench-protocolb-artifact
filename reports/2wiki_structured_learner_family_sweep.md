# 2Wiki Structured Learner-Family Sweep

- Scope: released 2Wiki structured learner-family diagnostic; appendix robustness for the structured-evidence regime
- Rows: 2000; repeats: 20; feature tier: B1_context
- CE scoring disabled: False
- Dev-selected winners: {'extra_trees_multioutput': 8, 'extra_trees_expanded': 8, 'rf_multioutput': 4}
- Best fixed on full sample: fixed_one_hop utility 0.8989
- Oracle on full sample: utility 0.9523

| model | family | utility | delta vs best fixed | 95% CI | positive share | oracle gap closed | choice share |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| oracle | evaluator-only oracle | 0.9524 | +0.0513 | [+0.0490,+0.0535] | 1.00 | 1.000 | summary:0.386; one_hop:0.432; two_hop:0.106; full_context:0.035; ce:0.041 |
| extra_trees_multioutput | tree ensemble | 0.9257 | +0.0246 | [+0.0226,+0.0267] | 1.00 | 0.477 | summary:0.362; one_hop:0.316; two_hop:0.301; full_context:0.001; ce:0.021 |
| extra_trees_expanded | tree ensemble | 0.9252 | +0.0241 | [+0.0224,+0.0258] | 1.00 | 0.469 | summary:0.348; one_hop:0.356; two_hop:0.272; full_context:0.002; ce:0.021 |
| dev_best_single_model | dev-selected learner | 0.9250 | +0.0238 | [+0.0223,+0.0254] | 1.00 | 0.464 | summary:0.353; one_hop:0.336; two_hop:0.289; full_context:0.002; ce:0.021 |
| rf_multioutput | tree ensemble | 0.9245 | +0.0233 | [+0.0215,+0.0254] | 1.00 | 0.453 | summary:0.342; one_hop:0.310; two_hop:0.326; full_context:0.001; ce:0.020 |
| rf_expanded | tree ensemble | 0.9239 | +0.0228 | [+0.0213,+0.0241] | 1.00 | 0.443 | summary:0.323; one_hop:0.408; two_hop:0.251; full_context:0.001; ce:0.018 |
| hist_gbdt_expanded | boosted tree | 0.9208 | +0.0197 | [+0.0179,+0.0215] | 1.00 | 0.383 | summary:0.304; one_hop:0.184; two_hop:0.494; full_context:0.006; ce:0.012 |
| ridge_multioutput | linear multi-output | 0.9166 | +0.0155 | [+0.0136,+0.0173] | 1.00 | 0.298 | summary:0.162; one_hop:0.367; two_hop:0.468; full_context:0.000; ce:0.003 |
| elasticnet_multioutput | linear multi-output | 0.9165 | +0.0154 | [+0.0135,+0.0173] | 1.00 | 0.296 | summary:0.155; one_hop:0.367; two_hop:0.476; ce:0.002 |
| knn_expanded | nonparametric | 0.9164 | +0.0153 | [+0.0135,+0.0172] | 1.00 | 0.296 | summary:0.253; one_hop:0.381; two_hop:0.358; full_context:0.000; ce:0.008 |
| mlp_expanded | neural | 0.9132 | +0.0121 | [+0.0100,+0.0142] | 1.00 | 0.234 | summary:0.189; one_hop:0.378; two_hop:0.389; full_context:0.008; ce:0.036 |
| mlp_multioutput | neural | 0.9131 | +0.0120 | [+0.0104,+0.0137] | 1.00 | 0.232 | summary:0.184; one_hop:0.372; two_hop:0.404; full_context:0.012; ce:0.029 |
| gbdt_expanded | boosted tree | 0.9119 | +0.0108 | [+0.0092,+0.0123] | 1.00 | 0.209 | summary:0.050; one_hop:0.401; two_hop:0.546; full_context:0.001; ce:0.002 |
| adaboost_expanded | boosted tree | 0.9065 | +0.0054 | [+0.0040,+0.0069] | 0.95 | 0.104 | summary:0.074; one_hop:0.922; two_hop:0.004 |
| decision_tree_expanded | tree | 0.9062 | +0.0051 | [+0.0031,+0.0071] | 0.90 | 0.099 | summary:0.304; one_hop:0.557; two_hop:0.069; full_context:0.030; ce:0.041 |
| logistic_oracle_view | classifier | 0.9050 | +0.0039 | [+0.0015,+0.0062] | 0.75 | 0.073 | summary:0.425; one_hop:0.533; two_hop:0.040; full_context:0.002; ce:0.001 |
| ridge_expanded | linear | 0.9043 | +0.0032 | [+0.0023,+0.0040] | 0.95 | 0.061 | summary:0.002; one_hop:0.380; two_hop:0.618; ce:0.001 |
| elasticnet_expanded | linear | 0.9037 | +0.0026 | [+0.0017,+0.0035] | 0.90 | 0.049 | summary:0.001; one_hop:0.365; two_hop:0.634 |
