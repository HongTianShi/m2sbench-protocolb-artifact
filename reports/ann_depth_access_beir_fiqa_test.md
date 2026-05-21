# ANN Depth Access Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- IVF/PQ: nlist=512, m=24, nbits=8, nprobes=[4, 8, 16, 32]

| view | NDCG@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: |
| pq4 | 0.231 | 0.069 | 0.226 | 89.11 |
| pq8 | 0.238 | 0.087 | 0.231 | 3.27 |
| pq16 | 0.246 | 0.125 | 0.236 | 4.90 |
| pq32 | 0.249 | 0.200 | 0.233 | 10.62 |
| full | 0.369 | 0.580 | 0.322 | 7.67 |

## Best-view shares

- pq4: 0.548
- pq8: 0.006
- pq16: 0.006
- pq32: 0.003
- full: 0.437

## Held-out random-forest depth router

- Utility@10: 0.316
- NDCG@10: 0.356
- Cost: 0.504
- Regret: 0.052
- Choices: pq4 0.07, pq8 0.05, pq16 0.03, pq32 0.00, full 0.85
