# Fitted-Q Access Router Audit

One-step offline RL/contextual-bandit router. State = method-visible summary/proxy features; action = access view; reward = cost-adjusted utility.

## Summary
| scope | policy | cases | utility | oracle_utility | regret | best_view_acc | avg_cost | non_richest_share | ndcg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_access | fixed_summary | 114 | 0.518 | 0.901 | 0.384 | 0.605 | 0.000 | 1.000 |  |
| synthetic_access | fixed_occupancy | 114 | 0.530 | 0.901 | 0.371 | 0.237 | 0.280 | 1.000 |  |
| synthetic_access | fixed_raster | 114 | 0.537 | 0.901 | 0.365 | 0.114 | 0.420 | 1.000 |  |
| synthetic_access | fixed_point | 114 | 0.568 | 0.901 | 0.333 | 0.044 | 0.580 | 0.000 |  |
| synthetic_access | fitted_q_router | 114 | 0.687 | 0.901 | 0.215 | 0.544 | 0.190 | 0.851 |  |
| synthetic_access | oracle_eval_only | 114 | 0.901 | 0.901 | 0.000 | 1.000 | 0.140 | 0.956 |  |
| ivf_pq_semantic | fixed_summary | 25000 | 0.266 | 0.368 | 0.102 | 0.456 | 0.000 | 1.000 | 0.266 |
| ivf_pq_semantic | fixed_pq | 25000 | 0.271 | 0.368 | 0.097 | 0.309 | 0.200 | 1.000 | 0.287 |
| ivf_pq_semantic | fixed_full | 25000 | 0.250 | 0.368 | 0.118 | 0.235 | 0.580 | 0.000 | 0.297 |
| ivf_pq_semantic | fitted_q_router_b0 | 25000 | 0.266 | 0.368 | 0.102 | 0.356 | 0.192 | 0.811 | 0.281 |
| ivf_pq_semantic | cluster_log_bandit_b1 | 25000 | 0.284 | 0.368 | 0.084 | 0.416 | 0.158 | 0.869 | 0.296 |
| ivf_pq_semantic | oracle_eval_only | 25000 | 0.368 | 0.368 | 0.000 | 1.000 | 0.198 | 0.765 | 0.384 |

## Cross-validation folds
| scope | fold | n | utility | oracle_utility | regret | best_view_acc |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic_access | 0 | 30 | 0.295 | 0.781 | 0.486 | 0.433 |
| synthetic_access | 1 | 36 | 0.835 | 0.992 | 0.157 | 0.389 |
| synthetic_access | 2 | 24 | 0.778 | 0.825 | 0.048 | 0.667 |
| synthetic_access | 3 | 24 | 0.862 | 0.990 | 0.128 | 0.792 |
| ivf_pq_semantic | 0 | 5000 | 0.287 | 0.394 | 0.107 | 0.361 |
| ivf_pq_semantic | 1 | 5000 | 0.232 | 0.328 | 0.095 | 0.345 |
| ivf_pq_semantic | 2 | 5000 | 0.248 | 0.352 | 0.105 | 0.357 |
| ivf_pq_semantic | 3 | 5000 | 0.285 | 0.388 | 0.103 | 0.369 |
| ivf_pq_semantic | 4 | 5000 | 0.276 | 0.378 | 0.102 | 0.348 |