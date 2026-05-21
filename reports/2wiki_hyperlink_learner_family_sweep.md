# 2Wiki Hyperlink Learner-Family Sweep

- Scope: released 2Wiki hyperlink learner-family diagnostic; appendix robustness for the 10k corpus-scale structured-evidence regime
- Rows: 10000; repeats: 10
- Best fixed on full sample: fixed_hyperlink_1hop utility 0.7939
- Oracle on full sample: utility 0.8839
- Dev-selected winners: {'extra_trees_multioutput': 8, 'rf_multioutput': 1, 'hist_gbdt_expanded': 1}

| model | family | utility | delta vs best fixed | 95% CI | positive share | oracle gap closed | choice share |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| oracle | evaluator-only oracle | 0.8839 | +0.0901 | [+0.0901,+0.0901] | 1.00 | 1.000 | summary:0.563; provided_context:0.170; hyperlink_1hop:0.172; hyperlink_2hop:0.083; full_local_pool:0.013 |
| extra_trees_multioutput | tree ensemble | 0.8253 | +0.0316 | [+0.0305,+0.0326] | 1.00 | 0.347 | summary:0.497; provided_context:0.045; hyperlink_1hop:0.263; hyperlink_2hop:0.192; full_local_pool:0.003 |
| dev_best_single_model | dev-selected learner | 0.8249 | +0.0312 | [+0.0297,+0.0323] | 1.00 | 0.342 | summary:0.498; provided_context:0.044; hyperlink_1hop:0.281; hyperlink_2hop:0.174; full_local_pool:0.003 |
| rf_multioutput | tree ensemble | 0.8240 | +0.0303 | [+0.0287,+0.0318] | 1.00 | 0.331 | summary:0.498; provided_context:0.043; hyperlink_1hop:0.254; hyperlink_2hop:0.203; full_local_pool:0.002 |
| hist_gbdt_expanded | boosted tree | 0.8232 | +0.0295 | [+0.0284,+0.0305] | 1.00 | 0.323 | summary:0.494; provided_context:0.046; hyperlink_1hop:0.414; hyperlink_2hop:0.036; full_local_pool:0.010 |
| extra_trees_expanded | tree ensemble | 0.8230 | +0.0293 | [+0.0276,+0.0309] | 1.00 | 0.321 | summary:0.508; provided_context:0.073; hyperlink_1hop:0.245; hyperlink_2hop:0.149; full_local_pool:0.025 |
| rf_expanded | tree ensemble | 0.8201 | +0.0264 | [+0.0249,+0.0280] | 1.00 | 0.289 | summary:0.499; provided_context:0.076; hyperlink_1hop:0.260; hyperlink_2hop:0.147; full_local_pool:0.017 |
| elasticnet_multioutput | linear multi-output | 0.8185 | +0.0248 | [+0.0238,+0.0258] | 1.00 | 0.272 | summary:0.381; provided_context:0.018; hyperlink_1hop:0.281; hyperlink_2hop:0.320; full_local_pool:0.000 |
| ridge_multioutput | linear multi-output | 0.8183 | +0.0246 | [+0.0236,+0.0255] | 1.00 | 0.269 | summary:0.383; provided_context:0.021; hyperlink_1hop:0.277; hyperlink_2hop:0.319; full_local_pool:0.000 |
| knn_expanded | nonparametric | 0.8177 | +0.0240 | [+0.0230,+0.0251] | 1.00 | 0.263 | summary:0.400; provided_context:0.047; hyperlink_1hop:0.305; hyperlink_2hop:0.245; full_local_pool:0.003 |
| gbdt_expanded | boosted tree | 0.8173 | +0.0236 | [+0.0229,+0.0243] | 1.00 | 0.259 | summary:0.438; provided_context:0.001; hyperlink_1hop:0.514; hyperlink_2hop:0.023; full_local_pool:0.024 |
| mlp_expanded | neural | 0.8162 | +0.0225 | [+0.0210,+0.0240] | 1.00 | 0.246 | summary:0.391; provided_context:0.051; hyperlink_1hop:0.270; hyperlink_2hop:0.246; full_local_pool:0.042 |
| mlp_multioutput | neural | 0.8152 | +0.0215 | [+0.0199,+0.0234] | 1.00 | 0.236 | summary:0.357; provided_context:0.075; hyperlink_1hop:0.290; hyperlink_2hop:0.238; full_local_pool:0.039 |
| decision_tree_expanded | tree | 0.8061 | +0.0124 | [+0.0096,+0.0150] | 1.00 | 0.134 | summary:0.583; provided_context:0.140; hyperlink_1hop:0.249; hyperlink_2hop:0.006; full_local_pool:0.021 |
| fixed_hyperlink_1hop | fixed reference | 0.7939 | +0.0000 | [+0.0000,+0.0000] | 1.00 | 0.000 | hyperlink_1hop:1.000 |
| ridge_expanded | linear | 0.7934 | -0.0003 | [-0.0010,+0.0000] | 0.10 | -0.004 | hyperlink_1hop:1.000 |
| elasticnet_expanded | linear | 0.7934 | -0.0003 | [-0.0010,+0.0000] | 0.10 | -0.004 | hyperlink_1hop:1.000 |
| logistic_oracle_view | classifier | 0.7918 | -0.0019 | [-0.0041,+0.0002] | 0.20 | -0.022 | summary:0.851; provided_context:0.073; hyperlink_1hop:0.077; hyperlink_2hop:0.000 |
