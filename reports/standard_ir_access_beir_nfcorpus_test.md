# Standard IR Access Audit: beir/nfcorpus/test

- Documents: 3,633
- Queries with qrels: 323
- Positive/graded qrel pairs retained: 12,334
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=181, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.142 | 0.149 | 0.064 | 0.393 | 0.00 | 0.149 | 68.07 |
| pq | 0.296 | 0.273 | 0.125 | 0.644 | 0.20 | 0.257 | 57.87 |
| full | 0.340 | 0.317 | 0.155 | 0.690 | 0.58 | 0.271 | 1.83 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.715
- PQ dominates full before cost share: 0.598
- Best-view shares: summary 0.412, PQ 0.303, full 0.285
- Break-even lambda*: summary->PQ 0.620, PQ->full 0.115
- Bootstrap 95% CI, non-full best share: [0.663, 0.762]
- Bootstrap 95% CI, PQ dominates full before cost: [0.545, 0.650]

## Held-out fixed-view baselines

| view | test queries | NDCG@10 | cost | utility@10 | regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 129 | 0.166 | 0.000 | 0.166 | 0.164 |
| pq | 129 | 0.280 | 0.200 | 0.264 | 0.066 |
| full | 129 | 0.311 | 0.580 | 0.265 | 0.065 |

## Held-out QPP/access routers

| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary_qpp_b0 | 129 | 0.307 | 0.562 | 0.262 | 0.068 | summary 0.00, pq 0.05, full 0.95 |
| query_summary_rf_b0 | 129 | 0.302 | 0.490 | 0.263 | 0.067 | summary 0.02, pq 0.20, full 0.78 |
| summary_pq_qpp_b1 | 129 | 0.307 | 0.542 | 0.264 | 0.066 | summary 0.00, pq 0.10, full 0.90 |
| query_summary_pq_rf_b1 | 129 | 0.292 | 0.450 | 0.256 | 0.074 | summary 0.05, pq 0.27, full 0.68 |
| selective_rerank_cascade_b1 | 129 | 0.294 | 0.407 | 0.262 | 0.068 | summary 0.12, pq 0.28, full 0.60; summary if mean le 0.1022; else PQ if top ge 0.5373; else full |
