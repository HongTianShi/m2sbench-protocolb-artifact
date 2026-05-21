# Joint Dense Access Menu Audit

- Dataset/model: `beir/fiqa/test` / `sentence-transformers/all-MiniLM-L6-v2`
- Boundary queries: 260; split train/dev/test: 156/52/52
- Menu: summary, binary, pq, int8, hnsw16, hnsw64, full, ce
- Cost profile: C_op; lambda: 0.04

| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed_ce | 0.3265 | 0.1093 | 0.000 | 1.000 | ce 1.00 | +0.0000 [-0.0944,+0.0845] |
| ce_full_gate_.020 | 0.3510 | 0.0848 | 0.224 | 0.365 | full 0.63, ce 0.37 | +0.0245 [-0.0097,+0.0611] |
| expanded_rf | 0.3624 | 0.0734 | 0.329 | 0.327 | hnsw64 0.56, full 0.08, ce 0.33 | +0.0359 [+0.0022,+0.0721] |
| joint_oracle | 0.4358 | 0.0000 | 1.000 | 0.000 | summary 0.23, binary 0.13, pq 0.12, hnsw16 0.25, ce 0.25 | -- |

## All fixed/restricted/learned routes

| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_summary | 0.0051 | 0.4307 | -2.940 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_binary | 0.2286 | 0.2072 | -0.896 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_pq | 0.2303 | 0.2055 | -0.880 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_int8 | 0.3028 | 0.1330 | -0.217 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw16 | 0.3035 | 0.1323 | -0.210 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw64 | 0.3058 | 0.1300 | -0.190 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_full | 0.2925 | 0.1433 | -0.311 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_ce | 0.3265 | 0.1093 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ce_full_gate_.020 | 0.3510 | 0.0848 | 0.224 | 0.365 | 0.000 | 0.000 | 0.000 | 0.000 |
| hnsw_margin_gate | 0.3023 | 0.1336 | -0.222 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| expanded_linear | 0.3028 | 0.1330 | -0.217 | 0.000 | 0.019 | 0.114 | 0.111 | 0.115 |
| expanded_rf | 0.3624 | 0.0734 | 0.329 | 0.327 | 0.154 | 0.051 | 0.110 | 0.021 |
| expanded_et | 0.3510 | 0.0848 | 0.224 | 0.269 | 0.154 | 0.037 | 0.110 | 0.020 |
| expanded_hgb | 0.3191 | 0.1167 | -0.068 | 0.269 | 0.135 | 0.001 | 0.110 | 0.003 |
| pairwise_logistic | 0.3035 | 0.1323 | -0.210 | 0.000 | 0.250 | -0.067 | 0.122 | 0.261 |

## Repeated 60/20/20 split

- Best learned minus best fixed: +0.0076 [-0.0202,+0.0383]
- Positive split share: 0.600
