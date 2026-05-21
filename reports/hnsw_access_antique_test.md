# HNSW Access Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- HNSW: M=32, efConstruction=100, efSearch=[16, 64, 128]

| view | NDCG@10 | cost | utility@10 | mean us/q | p50 us/q | p95 us/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hnsw16 | 0.387 | 0.138 | 0.376 | 162.21 | 154.16 | 235.86 |
| hnsw64 | 0.398 | 0.250 | 0.378 | 302.50 | 258.19 | 424.62 |
| hnsw128 | 0.397 | 0.400 | 0.365 | 401.56 | 397.18 | 524.14 |
| full | 0.399 | 0.580 | 0.353 | 24050.32 | 23591.82 | 27776.63 |

## Best-view shares

- hnsw16: 0.920
- hnsw64: 0.065
- hnsw128: 0.005
- full: 0.010

- Non-full cost-adjusted best share: 0.990
- Shallow-HNSW dominates full before cost: 0.900
- Bootstrap 95% CI, non-full best share: [0.975, 1.000]

## Held-out random-forest HNSW-depth router

- Utility@10: 0.364
- NDCG@10: 0.380
- Cost: 0.195
- Regret: 0.007
- Choices: hnsw16 0.49, hnsw64 0.51, hnsw128 0.00, full 0.00
