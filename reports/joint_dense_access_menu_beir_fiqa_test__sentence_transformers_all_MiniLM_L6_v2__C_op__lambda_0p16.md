# Joint Dense Access Menu Audit

- Dataset/model: `beir/fiqa/test` / `sentence-transformers/all-MiniLM-L6-v2`
- Boundary queries: 260; split train/dev/test: 156/52/52
- Menu: summary, binary, pq, int8, hnsw16, hnsw64, full, ce
- Cost profile: C_op; lambda: 0.16

| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed_hnsw16 | 0.2853 | 0.1151 | 0.000 | 0.000 | hnsw16 1.00 | -0.0000 [-0.0933,+0.0878] |
| hnsw_margin_gate | 0.2836 | 0.1168 | -0.015 | 0.000 | hnsw16 0.90, hnsw64 0.10 | -0.0017 [-0.0031,-0.0003] |
| expanded_rf | 0.2903 | 0.1101 | 0.043 | 0.000 | hnsw16 0.71, hnsw64 0.29 | +0.0050 [-0.0167,+0.0334] |
| joint_oracle | 0.4004 | 0.0000 | 1.000 | 0.000 | summary 0.29, binary 0.19, pq 0.13, hnsw16 0.25, ce 0.13 | -- |

## All fixed/restricted/learned routes

| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_summary | 0.0051 | 0.3953 | -2.434 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_binary | 0.2190 | 0.1814 | -0.576 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_pq | 0.2063 | 0.1941 | -0.686 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_int8 | 0.2668 | 0.1336 | -0.161 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw16 | 0.2853 | 0.1151 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw64 | 0.2757 | 0.1247 | -0.084 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_full | 0.2229 | 0.1775 | -0.542 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_ce | 0.2029 | 0.1975 | -0.716 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ce_full_gate_.020 | 0.2616 | 0.1388 | -0.206 | 0.365 | 0.000 | 0.000 | 0.000 | 0.000 |
| hnsw_margin_gate | 0.2836 | 0.1168 | -0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| expanded_linear | 0.2853 | 0.1151 | 0.000 | 0.000 | 0.250 | 0.010 | 0.111 | 0.113 |
| expanded_rf | 0.2903 | 0.1101 | 0.043 | 0.000 | 0.231 | 0.058 | 0.110 | 0.097 |
| expanded_et | 0.2889 | 0.1115 | 0.031 | 0.000 | 0.250 | 0.055 | 0.110 | 0.115 |
| expanded_hgb | 0.2800 | 0.1204 | -0.046 | 0.019 | 0.250 | 0.019 | 0.110 | 0.112 |
| pairwise_logistic | 0.2853 | 0.1151 | 0.000 | 0.000 | 0.250 | -0.040 | 0.122 | 0.287 |

## Repeated 60/20/20 split

- Best learned minus best fixed: +0.0018 [+0.0000,+0.0071]
- Positive split share: 0.400
