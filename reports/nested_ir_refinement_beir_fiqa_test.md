# Same-Candidate Nested IR Refinement Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- Positive/graded qrel pairs retained: 1,706
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Candidate pool: full flat top-200, shared by all views
- Nested evidence: centroid assignment -> PQ reconstruction -> full vector
- FAISS GPUs visible: 1

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.108 | 0.111 | 0.108 | 0.224 | 0.00 | 0.111 | 53.96 |
| pq | 0.271 | 0.287 | 0.341 | 0.545 | 0.20 | 0.271 | 50.37 |
| full | 0.345 | 0.369 | 0.441 | 0.657 | 0.58 | 0.322 | 42.29 |

## Diagnostics

- Non-full cost-adjusted best share: 0.645
- PQ beats full before cost share: 0.602
- Best-view shares: summary 0.403, PQ 0.242, full 0.355
- Break-even lambda*: summary->PQ 0.878, PQ->full 0.215
- Bootstrap 95% CI, non-full best share: [0.608, 0.681]
- Bootstrap 95% CI, PQ beats full before cost: [0.563, 0.640]
