# HNSW Cascade Router Audit: antique/test

- Documents: 403,666
- Queries with qrels: 200
- efSearch menu: [16, 64, 128] plus full

| route | utility@10 | NDCG@10 | cost | regret | diff vs best fixed | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| agreement_gate_same_rank | 0.381 | 0.393 | 0.152 | 0.005 | +0.009 | hnsw16 0.88, hnsw64 0.12; same_rank>=0.400: hnsw16, else hnsw64 |
| Ridge | 0.378 | 0.392 | 0.183 | 0.009 | +0.006 | hnsw16 0.60, hnsw64 0.40 |
| RandomForestRegressor | 0.376 | 0.392 | 0.205 | 0.012 | +0.004 | hnsw16 0.44, hnsw64 0.55, full 0.01 |
| ExtraTreesRegressor | 0.375 | 0.391 | 0.202 | 0.012 | +0.004 | hnsw16 0.50, hnsw64 0.47, full 0.03 |
| fixed_hnsw64 | 0.371 | 0.391 | 0.250 | 0.016 | +0.000 | hnsw64 1.00 |
| HistGradientBoostingRegressor | 0.371 | 0.394 | 0.289 | 0.016 | -0.000 | hnsw16 0.39, hnsw64 0.30, hnsw128 0.11, full 0.20 |
| fixed_hnsw16 | 0.369 | 0.380 | 0.138 | 0.018 | -0.002 | hnsw16 1.00 |
| fixed_hnsw128 | 0.361 | 0.393 | 0.400 | 0.026 | -0.010 | hnsw128 1.00 |
| fixed_full | 0.347 | 0.393 | 0.580 | 0.040 | -0.025 | full 1.00 |

## Split stability

- Repeated 60/40 splits: 20
- Best adaptive minus best fixed utility: +0.0063 [+0.0011, +0.0093]
- Positive split share: 0.950
