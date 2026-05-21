# Standard IR Access Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- Positive/graded qrel pairs retained: 1,706
- Encoder: `BAAI/bge-small-en-v1.5`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=512, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.017 | 0.022 | 0.034 | 0.086 | 0.00 | 0.022 | 752.54 |
| pq | 0.187 | 0.213 | 0.271 | 0.461 | 0.20 | 0.197 | 144.83 |
| full | 0.364 | 0.385 | 0.440 | 0.640 | 0.58 | 0.338 | 58.15 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.523
- PQ dominates full before cost share: 0.508
- Best-view shares: summary 0.353, PQ 0.170, full 0.477
- Break-even lambda*: summary->PQ 0.955, PQ->full 0.453
- Bootstrap 95% CI, non-full best share: [0.488, 0.560]
- Bootstrap 95% CI, PQ dominates full before cost: [0.471, 0.546]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 259 | 0.021 | 0.000 | 0.021 | 0.354 |
| pq | 259 | 0.221 | 0.200 | 0.205 | 0.171 |
| full | 259 | 0.382 | 0.580 | 0.336 | 0.040 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 259 | 0.382 | 0.580 | 0.336 | 0.040 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_rf_b0 | 259 | 0.381 | 0.577 | 0.335 | 0.041 | summary 0.00, pq 0.01, full 0.99 |
| summary_pq_qpp_b1 | 259 | 0.382 | 0.577 | 0.336 | 0.040 | summary 0.00, pq 0.01, full 0.99 |
| query_summary_pq_rf_b1 | 259 | 0.381 | 0.579 | 0.335 | 0.041 | summary 0.00, pq 0.00, full 1.00 |
| selective_rerank_cascade_b1 | 259 | 0.340 | 0.489 | 0.301 | 0.075 | summary 0.10, pq 0.09, full 0.81; summary if range ge 0.08169; else PQ if top ge 0.816; else full |
