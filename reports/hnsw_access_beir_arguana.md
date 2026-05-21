# HNSW Access Audit: beir/arguana

- Documents: 8,674
- Queries with qrels: 1,401
- HNSW: M=32, efConstruction=100, efSearch=[16, 64, 128]

| view | NDCG@10 | cost | utility@10 | mean us/q | p50 us/q | p95 us/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw16 | 0.372 | 0.138 | 0.361 | 17.17 | 16.69 | 21.87 |
| hnsw64 | 0.371 | 0.250 | 0.351 | 46.60 | 45.58 | 59.44 |
| hnsw128 | 0.371 | 0.400 | 0.339 | 76.62 | 74.61 | 100.74 |
| full | 0.371 | 0.580 | 0.325 | 241.05 | 232.36 | 292.49 |

## Best-view shares

- hnsw16: 0.982
- hnsw64: 0.000
- hnsw128: 0.000
- full: 0.018

- Non-full cost-adjusted best share: 0.982
- Shallow-HNSW dominates full before cost: 0.974
- Bootstrap 95% CI, non-full best share: [0.975, 0.988]

## Held-out random-forest HNSW-depth router

- Utility@10: 0.363
- NDCG@10: 0.374
- Cost: 0.144
- Regret: 0.002
- Choices: hnsw16 0.95, hnsw64 0.04, hnsw128 0.00, full 0.00
