# ANN Depth Access Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- IVF/PQ: nlist=2048, m=24, nbits=8, nprobes=[4, 8, 16, 32]

| view | NDCG@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: |
| pq4 | 0.266 | 0.069 | 0.260 | 196.32 |
| pq8 | 0.270 | 0.087 | 0.263 | 5.74 |
| pq16 | 0.276 | 0.125 | 0.266 | 6.98 |
| pq32 | 0.276 | 0.200 | 0.260 | 11.12 |
| full | 0.399 | 0.580 | 0.352 | 59.23 |

## Best-view shares

- pq4: 0.290
- pq8: 0.015
- pq16: 0.000
- pq32: 0.005
- full: 0.690

## Held-out random-forest depth router

- Utility@10: 0.334
- NDCG@10: 0.377
- Cost: 0.532
- Regret: 0.016
- Choices: pq4 0.03, pq8 0.04, pq16 0.04, pq32 0.00, full 0.90
