# Same-Candidate Nested IR Refinement Audit: beir/scifact/test

- Documents: 5,183
- Queries with qrels: 300
- Positive/graded qrel pairs retained: 339
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Candidate pool: full flat top-200, shared by all views
- Nested evidence: centroid assignment -> PQ reconstruction -> full vector
- FAISS GPUs visible: 1

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.277 | 0.288 | 0.317 | 0.333 | 0.00 | 0.288 | 46.04 |
| pq | 0.557 | 0.580 | 0.722 | 0.737 | 0.20 | 0.564 | 47.72 |
| full | 0.629 | 0.645 | 0.783 | 0.793 | 0.58 | 0.599 | 42.90 |

## Diagnostics

- Non-full cost-adjusted best share: 0.780
- PQ beats full before cost share: 0.737
- Best-view shares: summary 0.450, PQ 0.330, full 0.220
- Break-even lambda*: summary->PQ 1.462, PQ->full 0.171
- Bootstrap 95% CI, non-full best share: [0.733, 0.830]
- Bootstrap 95% CI, PQ beats full before cost: [0.683, 0.787]
