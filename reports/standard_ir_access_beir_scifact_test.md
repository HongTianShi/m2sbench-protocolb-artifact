# Standard IR Access Audit: beir/scifact/test

- Documents: 5,183
- Queries with qrels: 300
- Positive/graded qrel pairs retained: 339
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=256, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.112 | 0.144 | 0.272 | 0.287 | 0.00 | 0.144 | 98.21 |
| pq | 0.528 | 0.554 | 0.710 | 0.717 | 0.20 | 0.538 | 116.69 |
| full | 0.629 | 0.645 | 0.783 | 0.793 | 0.58 | 0.599 | 2.55 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.717
- PQ dominates full before cost share: 0.703
- Best-view shares: summary 0.247, PQ 0.470, full 0.283
- Break-even lambda*: summary->PQ 2.051, PQ->full 0.240
- Bootstrap 95% CI, non-full best share: [0.667, 0.770]
- Bootstrap 95% CI, PQ dominates full before cost: [0.653, 0.757]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 120 | 0.114 | 0.000 | 0.114 | 0.536 |
| pq | 120 | 0.562 | 0.200 | 0.546 | 0.104 |
| full | 120 | 0.637 | 0.580 | 0.591 | 0.059 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 120 | 0.635 | 0.545 | 0.591 | 0.058 | summary 0.00, pq 0.09, full 0.91 |
| query_summary_rf_b0 | 120 | 0.629 | 0.498 | 0.589 | 0.061 | summary 0.00, pq 0.22, full 0.78 |
| summary_pq_qpp_b1 | 120 | 0.641 | 0.504 | 0.601 | 0.049 | summary 0.00, pq 0.20, full 0.80 |
| query_summary_pq_rf_b1 | 120 | 0.626 | 0.456 | 0.590 | 0.060 | summary 0.00, pq 0.33, full 0.68 |
| selective_rerank_cascade_b1 | 120 | 0.547 | 0.330 | 0.521 | 0.129 | summary 0.12, pq 0.47, full 0.41; summary if range ge 0.3233; else PQ if range ge 0.08814; else full |
