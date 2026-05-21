# Same-Candidate Nested IR Refinement Audit: beir/nfcorpus/test

- Documents: 3,633
- Queries with qrels: 323
- Positive/graded qrel pairs retained: 12,334
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Candidate pool: full flat top-200, shared by all views
- Nested evidence: centroid assignment -> PQ reconstruction -> full vector
- FAISS GPUs visible: 1

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.137 | 0.132 | 0.052 | 0.424 | 0.00 | 0.132 | 25.75 |
| pq | 0.303 | 0.293 | 0.146 | 0.697 | 0.20 | 0.277 | 28.72 |
| full | 0.340 | 0.317 | 0.155 | 0.690 | 0.58 | 0.271 | 28.72 |

## Diagnostics

- Non-full cost-adjusted best share: 0.718
- PQ beats full before cost share: 0.628
- Best-view shares: summary 0.362, PQ 0.356, full 0.282
- Break-even lambda*: summary->PQ 0.804, PQ->full 0.064
- Bootstrap 95% CI, non-full best share: [0.666, 0.768]
- Bootstrap 95% CI, PQ beats full before cost: [0.573, 0.681]
