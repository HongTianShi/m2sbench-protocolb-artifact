# Standard IR Access Audit: beir/scifact/test

- Documents: 5,183
- Queries with qrels: 300
- Positive/graded qrel pairs retained: 339
- Encoder: `BAAI/bge-small-en-v1.5`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=259, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.114 | 0.143 | 0.273 | 0.280 | 0.00 | 0.143 | 613.85 |
| pq | 0.618 | 0.633 | 0.752 | 0.763 | 0.20 | 0.617 | 160.42 |
| full | 0.692 | 0.720 | 0.845 | 0.857 | 0.58 | 0.674 | 4.62 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.757
- PQ dominates full before cost share: 0.717
- Best-view shares: summary 0.203, PQ 0.553, full 0.243
- Break-even lambda*: summary->PQ 2.446, PQ->full 0.230
- Bootstrap 95% CI, non-full best share: [0.710, 0.803]
- Bootstrap 95% CI, PQ dominates full before cost: [0.667, 0.767]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 120 | 0.120 | 0.000 | 0.120 | 0.624 |
| pq | 120 | 0.658 | 0.200 | 0.642 | 0.102 |
| full | 120 | 0.734 | 0.580 | 0.688 | 0.057 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 120 | 0.734 | 0.580 | 0.688 | 0.057 | summary 0.00, pq 0.00, full 1.00 |
| query_summary_rf_b0 | 120 | 0.715 | 0.551 | 0.671 | 0.074 | summary 0.00, pq 0.07, full 0.93 |
| summary_pq_qpp_b1 | 120 | 0.733 | 0.526 | 0.690 | 0.054 | summary 0.00, pq 0.14, full 0.86 |
| query_summary_pq_rf_b1 | 120 | 0.729 | 0.510 | 0.688 | 0.056 | summary 0.00, pq 0.18, full 0.82 |
| selective_rerank_cascade_b1 | 120 | 0.671 | 0.411 | 0.638 | 0.106 | summary 0.10, pq 0.29, full 0.61; summary if top ge 0.8507; else PQ if margin ge 0.02117; else full |
