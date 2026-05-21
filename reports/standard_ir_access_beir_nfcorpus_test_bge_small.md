# Standard IR Access Audit: beir/nfcorpus/test

- Documents: 3,633
- Queries with qrels: 323
- Positive/graded qrel pairs retained: 12,334
- Encoder: `BAAI/bge-small-en-v1.5`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=181, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.151 | 0.153 | 0.054 | 0.372 | 0.00 | 0.153 | 306.35 |
| pq | 0.317 | 0.297 | 0.136 | 0.672 | 0.20 | 0.281 | 169.66 |
| full | 0.372 | 0.339 | 0.158 | 0.700 | 0.58 | 0.293 | 2.86 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.706
- PQ dominates full before cost share: 0.598
- Best-view shares: summary 0.424, PQ 0.282, full 0.294
- Break-even lambda*: summary->PQ 0.715, PQ->full 0.112
- Bootstrap 95% CI, non-full best share: [0.653, 0.752]
- Bootstrap 95% CI, PQ dominates full before cost: [0.545, 0.647]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 129 | 0.180 | 0.000 | 0.180 | 0.176 |
| pq | 129 | 0.284 | 0.200 | 0.268 | 0.089 |
| full | 129 | 0.330 | 0.580 | 0.283 | 0.073 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 129 | 0.329 | 0.533 | 0.286 | 0.071 | summary 0.00, pq 0.12, full 0.88 |
| query_summary_rf_b0 | 129 | 0.310 | 0.465 | 0.273 | 0.084 | summary 0.00, pq 0.30, full 0.70 |
| summary_pq_qpp_b1 | 129 | 0.326 | 0.512 | 0.285 | 0.072 | summary 0.03, pq 0.13, full 0.84 |
| query_summary_pq_rf_b1 | 129 | 0.315 | 0.451 | 0.279 | 0.078 | summary 0.04, pq 0.28, full 0.68 |
| selective_rerank_cascade_b1 | 129 | 0.319 | 0.501 | 0.279 | 0.078 | summary 0.09, pq 0.08, full 0.84; summary if top ge 0.8253; else PQ if std ge 0.0183; else full |
