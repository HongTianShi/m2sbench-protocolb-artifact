# Standard IR Access Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- Positive/graded qrel pairs retained: 6,589
- Encoder: `BAAI/bge-small-en-v1.5`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=512, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.005 | 0.005 | 0.003 | 0.075 | 0.00 | 0.005 | 3106.46 |
| pq | 0.263 | 0.240 | 0.081 | 0.850 | 0.20 | 0.224 | 349.00 |
| full | 0.435 | 0.403 | 0.149 | 0.965 | 0.58 | 0.356 | 4166.04 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.225
- PQ dominates full before cost share: 0.155
- Best-view shares: summary 0.050, PQ 0.175, full 0.775
- Break-even lambda*: summary->PQ 1.171, PQ->full 0.429
- Bootstrap 95% CI, non-full best share: [0.165, 0.285]
- Bootstrap 95% CI, PQ dominates full before cost: [0.105, 0.210]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 80 | 0.009 | 0.000 | 0.009 | 0.334 |
| pq | 80 | 0.204 | 0.200 | 0.188 | 0.155 |
| full | 80 | 0.377 | 0.580 | 0.331 | 0.012 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 80 | 0.377 | 0.580 | 0.331 | 0.012 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_rf_b0 | 80 | 0.377 | 0.580 | 0.331 | 0.012 | summary 0.00, pq 0.00, full 1.00 |
| summary_pq_qpp_b1 | 80 | 0.377 | 0.580 | 0.331 | 0.012 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_pq_rf_b1 | 80 | 0.377 | 0.580 | 0.331 | 0.012 | summary 0.00, pq 0.00, full 1.00 |
| selective_rerank_cascade_b1 | 80 | 0.299 | 0.467 | 0.262 | 0.081 | summary 0.16, pq 0.05, full 0.79; summary if top ge 0.7868; else PQ if mean ge 0.7497; else full |
