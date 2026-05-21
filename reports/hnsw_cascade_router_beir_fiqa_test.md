# HNSW Cascade Router Audit: beir/fiqa/test

- Documents: 57,638
- Queries with qrels: 648
- efSearch menu: [16, 64, 128] plus full

| route | utility@10 | NDCG@10 | cost | regret | diff vs best fixed | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ridge | 0.344 | 0.358 | 0.173 | 0.011 | +0.009 | hnsw16 0.72, hnsw64 0.25, hnsw128 0.03 |
| agreement_gate_top1_same | 0.342 | 0.353 | 0.141 | 0.005 | +0.007 | hnsw16 0.97, hnsw64 0.03; top1_same>=1.000: hnsw16, else hnsw64 |
| ExtraTreesRegressor | 0.339 | 0.354 | 0.195 | 0.016 | +0.004 | hnsw16 0.60, hnsw64 0.31, hnsw128 0.08 |
| HistGradientBoostingRegressor | 0.337 | 0.359 | 0.269 | 0.018 | +0.002 | hnsw16 0.41, hnsw64 0.27, hnsw128 0.22, full 0.10 |
| RandomForestRegressor | 0.337 | 0.353 | 0.202 | 0.018 | +0.002 | hnsw16 0.53, hnsw64 0.39, hnsw128 0.08 |
| fixed_hnsw64 | 0.335 | 0.355 | 0.250 | 0.020 | +0.000 | hnsw64 1.00 |
| fixed_hnsw16 | 0.333 | 0.344 | 0.138 | 0.022 | -0.002 | hnsw16 1.00 |
| fixed_hnsw128 | 0.327 | 0.359 | 0.400 | 0.028 | -0.008 | hnsw128 1.00 |
| fixed_full | 0.317 | 0.363 | 0.580 | 0.038 | -0.018 | full 1.00 |

## Split stability

- Repeated 60/40 splits: 20
- Best adaptive minus best fixed utility: +0.0067 [+0.0022, +0.0130]
- Positive split share: 1.000
