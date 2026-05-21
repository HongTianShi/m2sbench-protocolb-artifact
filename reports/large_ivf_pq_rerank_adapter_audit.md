# Large IVF-PQ Rerank Adapter Audit

Scored 159,040 queries over 659 evaluated IVF-style cells in 6.3 minutes
(692 raw coarse cells before filtering empty or below-threshold cells).

## Aggregate

| scope | queries | cells | summary_ndcg | pq_ndcg | full_ndcg | summary_utility | pq_utility | full_utility | full_gain | pq_gain | full_dominated_by_pq_share | best_not_full_share | best_summary_share | best_pq_share | best_full_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 159040 | 659 | 0.268 | 0.288 | 0.299 | 0.268 | 0.272 | 0.253 | 0.031 | 0.02 | 0.385 | 0.764 | 0.46 | 0.304 | 0.236 |
| docred | 69218 | 287 | 0.161 | 0.187 | 0.21 | 0.161 | 0.171 | 0.164 | 0.049 | 0.026 | 0.286 | 0.755 | 0.511 | 0.245 | 0.245 |
| douban | 54987 | 220 | 0.271 | 0.28 | 0.283 | 0.271 | 0.264 | 0.236 | 0.012 | 0.009 | 0.465 | 0.764 | 0.415 | 0.349 | 0.236 |
| jd_reviews | 29947 | 120 | 0.465 | 0.485 | 0.488 | 0.465 | 0.469 | 0.441 | 0.023 | 0.021 | 0.447 | 0.782 | 0.443 | 0.339 | 0.218 |
| taptap | 4888 | 32 | 0.561 | 0.599 | 0.593 | 0.561 | 0.583 | 0.546 | 0.032 | 0.038 | 0.499 | 0.778 | 0.356 | 0.423 | 0.222 |

## Sufficiency slices

| slice | queries | cluster_radius_mean | summary_ndcg | pq_ndcg | full_ndcg | full_gain | full_dominated_by_pq_share | best_not_full_share | near_zero_full_gain_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact_30pct | 48538 | 0.064 | 0.354 | 0.37 | 0.374 | 0.019 | 0.461 | 0.764 | 0.51 |
| diffuse_30pct | 47735 | 0.202 | 0.18 | 0.203 | 0.222 | 0.042 | 0.327 | 0.757 | 0.571 |

## Policies

| policy | utility | ndcg | avg_cost | best_view_acc | non_full_share | utility_sd | utility_ci95 | regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.268 | 0.268 | 0 | 0.46 | 1 |  |  |  |
| fixed_pq | 0.272 | 0.288 | 0.2 | 0.304 | 1 |  |  |  |
| fixed_full | 0.253 | 0.299 | 0.58 | 0.236 | 0 |  |  |  |
| proxy_router_rf | 0.271 | 0.284 | 0.156 | 0.364 | 0.893 | 0.01 | 0.008 | 0.099 |
