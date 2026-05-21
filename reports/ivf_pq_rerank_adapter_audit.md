# IVF-PQ Rerank Adapter Audit

This adapter maps M2S-Bench cells to IVF coarse clusters over real citation-text vectors. PQ reconstruction is the compressed structural view; full dense-vector reranking is the richest view.

## Aggregate

| scope | queries | clusters | summary_ndcg | pq_ndcg | full_ndcg | summary_utility | pq_utility | full_utility | full_gain | pq_gain | full_dominated_by_pq_share | best_not_full_share | best_summary_share | best_pq_share | best_full_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 5995 | 155 | 0.513 | 0.537 | 0.575 | 0.513 | 0.521 | 0.528 | 0.061 | 0.023 | 0.344 | 0.721 | 0.444 | 0.277 | 0.279 |
| citeseer | 3297 | 77 | 0.52 | 0.525 | 0.562 | 0.52 | 0.509 | 0.515 | 0.042 | 0.006 | 0.358 | 0.733 | 0.472 | 0.261 | 0.267 |
| cora | 2698 | 78 | 0.506 | 0.551 | 0.591 | 0.506 | 0.535 | 0.544 | 0.085 | 0.045 | 0.327 | 0.706 | 0.41 | 0.297 | 0.294 |

## Sufficiency slices

| slice | queries | cluster_radius_mean | summary_ndcg | pq_ndcg | full_ndcg | full_gain | best_not_full_share | near_zero_full_gain_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_30pct | 1827 | 0.395 | 0.581 | 0.602 | 0.633 | 0.053 | 0.766 | 0.55 |
| diffuse_30pct | 1850 | 0.498 | 0.421 | 0.439 | 0.489 | 0.067 | 0.675 | 0.476 |

## Routing policies

| policy | utility | ndcg | avg_cost | best_view_acc | utility_sd | regret | non_full_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.513 | 0.513 | 0 | 0.444 |  |  |  |
| fixed_pq | 0.521 | 0.537 | 0.2 | 0.277 |  |  |  |
| fixed_full | 0.528 | 0.575 | 0.58 | 0.279 |  |  |  |
| proxy_router_rf | 0.53 | 0.552 | 0.276 | 0.381 | 0.027 | 0.102 | 0.627 |
