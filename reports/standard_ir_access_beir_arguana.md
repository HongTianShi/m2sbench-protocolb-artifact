# Standard IR Access Audit: beir/arguana

- Documents: 8,674
- Queries with qrels: 1,401
- Positive/graded qrel pairs retained: 1,401
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=433, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.130 | 0.197 | 0.423 | 0.423 | 0.00 | 0.197 | 13.74 |
| pq | 0.284 | 0.350 | 0.732 | 0.732 | 0.20 | 0.334 | 23.43 |
| full | 0.309 | 0.371 | 0.768 | 0.768 | 0.58 | 0.325 | 1.29 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.777
- PQ dominates full before cost share: 0.687
- Best-view shares: summary 0.370, PQ 0.408, full 0.223
- Break-even lambda*: summary->PQ 0.763, PQ->full 0.057
- Bootstrap 95% CI, non-full best share: [0.755, 0.799]
- Bootstrap 95% CI, PQ dominates full before cost: [0.664, 0.712]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 560 | 0.185 | 0.000 | 0.185 | 0.239 |
| pq | 560 | 0.346 | 0.200 | 0.330 | 0.094 |
| full | 560 | 0.374 | 0.580 | 0.328 | 0.096 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 560 | 0.346 | 0.208 | 0.329 | 0.095 | summary 0.00, pq 0.97, full 0.02 |
| query_summary_rf_b0 | 560 | 0.359 | 0.309 | 0.335 | 0.089 | summary 0.02, pq 0.68, full 0.30 |
| summary_pq_qpp_b1 | 560 | 0.356 | 0.299 | 0.332 | 0.092 | summary 0.01, pq 0.72, full 0.27 |
| query_summary_pq_rf_b1 | 560 | 0.354 | 0.305 | 0.330 | 0.094 | summary 0.03, pq 0.67, full 0.29 |
| selective_rerank_cascade_b1 | 560 | 0.353 | 0.266 | 0.332 | 0.092 | summary 0.11, pq 0.66, full 0.23; summary if top ge 0.8287; else PQ if range ge 0.1403; else full |
