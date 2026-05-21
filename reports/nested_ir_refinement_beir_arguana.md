# Same-Candidate Nested IR Refinement Audit: beir/arguana

- Documents: 8,674
- Queries with qrels: 1,401
- Positive/graded qrel pairs retained: 1,401
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Candidate pool: full flat top-200, shared by all views
- Nested evidence: centroid assignment -> PQ reconstruction -> full vector
- FAISS GPUs visible: 1

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.022 | 0.047 | 0.124 | 0.124 | 0.00 | 0.047 | 27.87 |
| pq | 0.260 | 0.333 | 0.707 | 0.707 | 0.20 | 0.317 | 27.27 |
| full | 0.309 | 0.371 | 0.768 | 0.768 | 0.58 | 0.325 | 23.74 |

## Diagnostics

- Non-full cost-adjusted best share: 0.673
- PQ beats full before cost share: 0.644
- Best-view shares: summary 0.230, PQ 0.443, full 0.327
- Break-even lambda*: summary->PQ 1.429, PQ->full 0.100
- Bootstrap 95% CI, non-full best share: [0.649, 0.699]
- Bootstrap 95% CI, PQ beats full before cost: [0.621, 0.668]
