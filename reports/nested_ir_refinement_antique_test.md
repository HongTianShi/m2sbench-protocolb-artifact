# Same-Candidate Nested IR Refinement Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- Positive/graded qrel pairs retained: 6,589
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Candidate pool: full flat top-200, shared by all views
- Nested evidence: centroid assignment -> PQ reconstruction -> full vector
- FAISS GPUs visible: 1

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.129 | 0.107 | 0.030 | 0.610 | 0.00 | 0.107 | 377.91 |
| pq | 0.347 | 0.318 | 0.116 | 0.925 | 0.20 | 0.302 | 55.78 |
| full | 0.428 | 0.399 | 0.148 | 0.955 | 0.58 | 0.353 | 47.23 |

## Diagnostics

- Non-full cost-adjusted best share: 0.435
- PQ beats full before cost share: 0.255
- Best-view shares: summary 0.110, PQ 0.325, full 0.565
- Break-even lambda*: summary->PQ 1.058, PQ->full 0.213
- Bootstrap 95% CI, non-full best share: [0.370, 0.500]
- Bootstrap 95% CI, PQ beats full before cost: [0.195, 0.315]
