# HNSW Cascade Router Audit: beir/arguana

- Documents: 8,674
- Queries with qrels: 1,401
- efSearch menu: [16, 64, 128] plus full

| route | utility@10 | NDCG@10 | cost | regret | diff vs best fixed | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed_hnsw16 | 0.355 | 0.366 | 0.138 | 0.001 | +0.000 | hnsw16 1.00 |
| agreement_gate_top1_same | 0.355 | 0.366 | 0.138 | 0.000 | +0.000 | hnsw16 1.00; top1_same>=1.000: hnsw16, else hnsw64 |
| Ridge | 0.355 | 0.366 | 0.138 | 0.001 | +0.000 | hnsw16 1.00 |
| RandomForestRegressor | 0.353 | 0.364 | 0.145 | 0.003 | -0.002 | hnsw16 0.97, hnsw64 0.02, full 0.01 |
| ExtraTreesRegressor | 0.352 | 0.364 | 0.152 | 0.004 | -0.003 | hnsw16 0.92, hnsw64 0.06, full 0.02 |
| HistGradientBoostingRegressor | 0.346 | 0.364 | 0.222 | 0.009 | -0.008 | hnsw16 0.55, hnsw64 0.34, full 0.10 |
| fixed_hnsw64 | 0.346 | 0.366 | 0.250 | 0.010 | -0.009 | hnsw64 1.00 |
| fixed_hnsw128 | 0.334 | 0.366 | 0.400 | 0.022 | -0.021 | hnsw128 1.00 |
| fixed_full | 0.318 | 0.364 | 0.580 | 0.038 | -0.037 | full 1.00 |

## Split stability

- Repeated 60/40 splits: 20
- Best adaptive minus best fixed utility: +0.0000 [+0.0000, +0.0000]
- Positive split share: 0.000
