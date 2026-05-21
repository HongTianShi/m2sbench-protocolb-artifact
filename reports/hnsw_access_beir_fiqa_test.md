# HNSW Access Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- HNSW: M=32, efConstruction=100, efSearch=[16, 64, 128]

| view | NDCG@10 | cost | utility@10 | mean us/q | p50 us/q | p95 us/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw16 | 0.351 | 0.138 | 0.340 | 59.56 | 54.80 | 76.28 |
| hnsw64 | 0.365 | 0.250 | 0.345 | 225.32 | 225.18 | 307.38 |
| hnsw128 | 0.367 | 0.400 | 0.335 | 227.96 | 221.86 | 303.02 |
| full | 0.369 | 0.580 | 0.322 | 3041.73 | 2918.22 | 3767.80 |

## Best-view shares

- hnsw16: 0.955
- hnsw64: 0.039
- hnsw128: 0.005
- full: 0.002

- Non-full cost-adjusted best share: 0.998
- Shallow-HNSW dominates full before cost: 0.957
- Bootstrap 95% CI, non-full best share: [0.995, 1.000]

## Held-out random-forest HNSW-depth router

- Utility@10: 0.340
- NDCG@10: 0.356
- Cost: 0.198
- Regret: 0.016
- Choices: hnsw16 0.46, hnsw64 0.54, hnsw128 0.00, full 0.00
