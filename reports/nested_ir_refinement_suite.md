# Same-Candidate Nested IR Refinement Suite

- Datasets: beir/fiqa/test, beir/scifact/test, beir/nfcorpus/test, beir/arguana, antique/test
- Queries: 2,872
- Qrel pairs: 22,369
- Candidate pool: full flat top-200, fixed for all views
- Evidence ladder: assigned centroid -> PQ reconstruction -> full vector

| view | weighted NDCG@10 | weighted utility@10 | macro NDCG@10 | macro utility@10 |
| --- | ---: | ---: | ---: | ---: |
| summary | 0.101 | 0.101 | 0.137 | 0.137 |
| pq | 0.343 | 0.327 | 0.362 | 0.346 |
| full | 0.395 | 0.349 | 0.420 | 0.374 |

## Diagnostics

- Non-full cost-adjusted best share: 0.666
- PQ beats full before cost share: 0.615
- Best-view shares: summary 0.298, PQ 0.368, full 0.334

| dataset | queries | utility S/PQ/F | non-full best | PQ >= full pre-cost | lambda* PQ->full |
| --- | ---: | --- | ---: | ---: | ---: |
| beir/fiqa/test | 648 | 0.111/0.271/0.322 | 0.645 | 0.602 | 0.215 |
| beir/scifact/test | 300 | 0.288/0.564/0.599 | 0.780 | 0.737 | 0.171 |
| beir/nfcorpus/test | 323 | 0.132/0.277/0.271 | 0.718 | 0.628 | 0.064 |
| beir/arguana | 1401 | 0.047/0.317/0.325 | 0.673 | 0.644 | 0.100 |
| antique/test | 200 | 0.107/0.302/0.353 | 0.435 | 0.255 | 0.213 |
