# Standard IR Access Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- Positive/graded qrel pairs retained: 6,589
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=2048, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.051 | 0.054 | 0.025 | 0.385 | 0.00 | 0.054 | 6.12 |
| pq | 0.292 | 0.276 | 0.104 | 0.865 | 0.20 | 0.260 | 67.48 |
| full | 0.428 | 0.399 | 0.147 | 0.955 | 0.58 | 0.352 | 63.83 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.295
- PQ dominates full before cost share: 0.190
- Best-view shares: summary 0.115, PQ 0.180, full 0.705
- Break-even lambda*: summary->PQ 1.109, PQ->full 0.323
- Bootstrap 95% CI, non-full best share: [0.235, 0.365]
- Bootstrap 95% CI, PQ dominates full before cost: [0.130, 0.245]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 80 | 0.073 | 0.000 | 0.073 | 0.282 |
| pq | 80 | 0.253 | 0.200 | 0.237 | 0.117 |
| full | 80 | 0.380 | 0.580 | 0.334 | 0.021 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 80 | 0.380 | 0.580 | 0.334 | 0.021 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_rf_b0 | 80 | 0.380 | 0.580 | 0.334 | 0.021 | summary 0.00, pq 0.00, full 1.00 |
| summary_pq_qpp_b1 | 80 | 0.382 | 0.566 | 0.336 | 0.019 | summary 0.00, pq 0.04, full 0.96 |
| query_summary_pq_rf_b1 | 80 | 0.380 | 0.575 | 0.334 | 0.021 | summary 0.00, pq 0.01, full 0.99 |
| selective_rerank_cascade_b1 | 80 | 0.355 | 0.481 | 0.316 | 0.039 | summary 0.14, pq 0.05, full 0.81; summary if top ge 0.7424; else PQ if mean ge 0.6959; else full |
