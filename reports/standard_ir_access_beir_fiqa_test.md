# Standard IR Access Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- Positive/graded qrel pairs retained: 1,706
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=512, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.012 | 0.016 | 0.027 | 0.060 | 0.00 | 0.016 | 38.86 |
| pq | 0.224 | 0.249 | 0.320 | 0.515 | 0.20 | 0.233 | 37.54 |
| full | 0.345 | 0.369 | 0.441 | 0.657 | 0.58 | 0.322 | 7.47 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.562
- PQ dominates full before cost share: 0.537
- Best-view shares: summary 0.316, PQ 0.245, full 0.438
- Break-even lambda*: summary->PQ 1.162, PQ->full 0.316
- Bootstrap 95% CI, non-full best share: [0.520, 0.603]
- Bootstrap 95% CI, PQ dominates full before cost: [0.497, 0.577]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 259 | 0.017 | 0.000 | 0.017 | 0.353 |
| pq | 259 | 0.259 | 0.200 | 0.243 | 0.127 |
| full | 259 | 0.365 | 0.580 | 0.318 | 0.051 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 259 | 0.365 | 0.580 | 0.318 | 0.051 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_rf_b0 | 259 | 0.363 | 0.573 | 0.318 | 0.052 | summary 0.00, pq 0.02, full 0.98 |
| summary_pq_qpp_b1 | 259 | 0.364 | 0.573 | 0.319 | 0.051 | summary 0.00, pq 0.02, full 0.98 |
| query_summary_pq_rf_b1 | 259 | 0.358 | 0.558 | 0.314 | 0.056 | summary 0.00, pq 0.06, full 0.94 |
| selective_rerank_cascade_b1 | 259 | 0.342 | 0.490 | 0.302 | 0.067 | summary 0.09, pq 0.10, full 0.81; summary if top ge 0.7029; else PQ if range ge 0.09389; else full |
