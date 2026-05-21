# Standard IR Access Audit: beir/arguana

- Documents: 8,674
- Queries with qrels: 1,401
- Positive/graded qrel pairs retained: 1,401
- Encoder: `BAAI/bge-small-en-v1.5`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=433, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.161 | 0.225 | 0.465 | 0.465 | 0.00 | 0.225 | 271.24 |
| pq | 0.321 | 0.377 | 0.763 | 0.763 | 0.20 | 0.361 | 191.12 |
| full | 0.379 | 0.430 | 0.844 | 0.844 | 0.58 | 0.384 | 58.80 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.699
- PQ dominates full before cost share: 0.605
- Best-view shares: summary 0.308, PQ 0.390, full 0.301
- Break-even lambda*: summary->PQ 0.756, PQ->full 0.141
- Bootstrap 95% CI, non-full best share: [0.675, 0.723]
- Bootstrap 95% CI, PQ dominates full before cost: [0.580, 0.629]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 560 | 0.219 | 0.000 | 0.219 | 0.253 |
| pq | 560 | 0.369 | 0.200 | 0.353 | 0.119 |
| full | 560 | 0.419 | 0.580 | 0.373 | 0.099 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 560 | 0.419 | 0.569 | 0.374 | 0.098 | summary 0.00, pq 0.03, full 0.97 |
| query_summary_rf_b0 | 560 | 0.413 | 0.484 | 0.375 | 0.097 | summary 0.03, pq 0.21, full 0.76 |
| summary_pq_qpp_b1 | 560 | 0.414 | 0.526 | 0.372 | 0.100 | summary 0.01, pq 0.13, full 0.86 |
| query_summary_pq_rf_b1 | 560 | 0.420 | 0.460 | 0.383 | 0.089 | summary 0.03, pq 0.28, full 0.70 |
| selective_rerank_cascade_b1 | 560 | 0.400 | 0.437 | 0.365 | 0.107 | summary 0.16, pq 0.13, full 0.71; summary if top ge 0.8858; else PQ if std ge 0.03707; else full |
