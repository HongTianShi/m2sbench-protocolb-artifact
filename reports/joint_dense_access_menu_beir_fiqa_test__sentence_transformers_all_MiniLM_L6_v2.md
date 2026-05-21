# Joint Dense Access Menu Audit

- Dataset/model: `beir/fiqa/test` / `sentence-transformers/all-MiniLM-L6-v2`
- Boundary queries: 260; split train/dev/test: 156/52/52
- Menu: summary, binary, pq, int8, hnsw16, hnsw64, full, ce
- Cost profile: C_op; lambda: 0.08

| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed_hnsw64 | 0.2958 | 0.1267 | 0.000 | 0.000 | hnsw64 1.00 | -0.0000 [-0.0893,+0.0851] |
| ce_full_gate_.020 | 0.3212 | 0.1012 | 0.201 | 0.365 | full 0.63, ce 0.37 | +0.0254 [-0.0185,+0.0800] |
| expanded_et | 0.3160 | 0.1065 | 0.159 | 0.038 | hnsw16 0.56, hnsw64 0.38 | +0.0202 [-0.0093,+0.0504] |
| joint_oracle | 0.4224 | 0.0000 | 1.000 | 0.212 | summary 0.23, binary 0.17, pq 0.12, hnsw16 0.27, ce 0.21 | -- |

## All fixed/restricted/learned routes

| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_summary | 0.0051 | 0.4173 | -2.295 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_binary | 0.2254 | 0.1970 | -0.556 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_pq | 0.2223 | 0.2001 | -0.580 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_int8 | 0.2908 | 0.1316 | -0.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw16 | 0.2940 | 0.1285 | -0.014 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw64 | 0.2958 | 0.1267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_full | 0.2693 | 0.1531 | -0.209 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_ce | 0.2853 | 0.1371 | -0.083 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ce_full_gate_.020 | 0.3212 | 0.1012 | 0.201 | 0.365 | 0.000 | 0.000 | 0.000 | 0.000 |
| hnsw_margin_gate | 0.2910 | 0.1314 | -0.038 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| expanded_linear | 0.2958 | 0.1267 | 0.000 | 0.000 | 0.000 | 0.059 | 0.111 | 0.135 |
| expanded_rf | 0.2990 | 0.1234 | 0.026 | 0.019 | 0.154 | -0.023 | 0.109 | 0.021 |
| expanded_et | 0.3160 | 0.1065 | 0.159 | 0.038 | 0.231 | 0.019 | 0.110 | 0.097 |
| expanded_hgb | 0.3066 | 0.1158 | 0.086 | 0.115 | 0.135 | 0.025 | 0.110 | 0.002 |
| pairwise_logistic | 0.2940 | 0.1285 | -0.014 | 0.000 | 0.269 | -0.054 | 0.120 | 0.264 |

## Repeated 60/20/20 split

- Best learned minus best fixed: -0.0000 [-0.0132,+0.0148]
- Positive split share: 0.467
