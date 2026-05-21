# Joint Dense Access Menu Audit

- Dataset/model: `beir/fiqa/test` / `sentence-transformers/all-MiniLM-L6-v2`
- Boundary queries: 260; split train/dev/test: 156/52/52
- Menu: summary, binary, pq, int8, hnsw16, hnsw64, full, ce
- Cost profile: C_lat; lambda: 0.08

| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed_full | 0.3157 | 0.1109 | 0.000 | 0.000 | full 1.00 | -0.0000 [-0.0894,+0.0850] |
| ce_full_gate_.020 | 0.3515 | 0.0751 | 0.323 | 0.365 | full 0.63, ce 0.37 | +0.0358 [-0.0064,+0.0885] |
| expanded_linear | 0.3157 | 0.1109 | 0.000 | 0.000 | full 1.00 | +0.0000 [+0.0000,+0.0000] |
| joint_oracle | 0.4266 | 0.0000 | 1.000 | 0.212 | summary 0.23, binary 0.12, pq 0.12, int8 0.27, ce 0.21 | -- |

## All fixed/restricted/learned routes

| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_summary | 0.0051 | 0.4215 | -2.799 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_binary | 0.2318 | 0.1948 | -0.756 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_pq | 0.2383 | 0.1883 | -0.698 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_int8 | 0.3148 | 0.1119 | -0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw16 | 0.3097 | 0.1169 | -0.054 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw64 | 0.3156 | 0.1110 | -0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_full | 0.3157 | 0.1109 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_ce | 0.2877 | 0.1389 | -0.252 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ce_full_gate_.020 | 0.3515 | 0.0751 | 0.323 | 0.365 | 0.000 | 0.000 | 0.000 | 0.000 |
| hnsw_margin_gate | 0.3185 | 0.1081 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| expanded_linear | 0.3157 | 0.1109 | 0.000 | 0.000 | 0.019 | 0.005 | 0.111 | 0.116 |
| expanded_rf | 0.3157 | 0.1110 | -0.000 | 0.000 | 0.019 | -0.021 | 0.110 | 0.114 |
| expanded_et | 0.3156 | 0.1110 | -0.000 | 0.000 | 0.000 | -0.055 | 0.110 | 0.134 |
| expanded_hgb | 0.2816 | 0.1450 | -0.307 | 0.058 | 0.154 | 0.069 | 0.110 | 0.017 |
| pairwise_logistic | 0.3148 | 0.1119 | -0.008 | 0.000 | 0.269 | -0.044 | 0.130 | 0.340 |

## Repeated 60/20/20 split

- Best learned minus best fixed: +0.0014 [-0.0026,+0.0121]
- Positive split share: 0.300
