# Joint Dense Access Menu Audit

- Dataset/model: `beir/fiqa/test` / `sentence-transformers/all-MiniLM-L6-v2`
- Boundary queries: 260; split train/dev/test: 156/52/52
- Menu: summary, binary, pq, int8, hnsw16, hnsw64, full, ce
- Cost profile: C_mem; lambda: 0.08

| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed_ce | 0.3263 | 0.1035 | 0.000 | 1.000 | ce 1.00 | +0.0000 [-0.0944,+0.0845] |
| ce_full_gate_.020 | 0.3149 | 0.1150 | -0.111 | 0.365 | full 0.63, ce 0.37 | -0.0114 [-0.0453,+0.0231] |
| expanded_et | 0.3540 | 0.0758 | 0.268 | 0.404 | int8 0.54, ce 0.40 | +0.0277 [-0.0069,+0.0646] |
| joint_oracle | 0.4298 | 0.0000 | 1.000 | 0.250 | summary 0.23, binary 0.13, pq 0.15, int8 0.21, ce 0.25 | -- |

## All fixed/restricted/learned routes

| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_summary | 0.0044 | 0.4254 | -3.109 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_binary | 0.2293 | 0.2005 | -0.937 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_pq | 0.2373 | 0.1925 | -0.860 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_int8 | 0.2948 | 0.1350 | -0.304 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw16 | 0.2502 | 0.1796 | -0.735 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_hnsw64 | 0.2723 | 0.1575 | -0.522 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_full | 0.2357 | 0.1941 | -0.875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_ce | 0.3263 | 0.1035 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ce_full_gate_.020 | 0.3149 | 0.1150 | -0.111 | 0.365 | 0.000 | 0.000 | 0.000 | 0.000 |
| hnsw_margin_gate | 0.2502 | 0.1796 | -0.735 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| expanded_linear | 0.2948 | 0.1350 | -0.304 | 0.000 | 0.212 | 0.029 | 0.110 | 0.076 |
| expanded_rf | 0.3381 | 0.0918 | 0.114 | 0.385 | 0.269 | 0.071 | 0.109 | 0.137 |
| expanded_et | 0.3540 | 0.0758 | 0.268 | 0.404 | 0.327 | 0.074 | 0.110 | 0.193 |
| expanded_hgb | 0.2749 | 0.1549 | -0.497 | 0.269 | 0.192 | 0.082 | 0.110 | 0.054 |
| pairwise_logistic | 0.2948 | 0.1350 | -0.304 | 0.000 | 0.212 | -0.051 | 0.111 | 0.183 |

## Repeated 60/20/20 split

- Best learned minus best fixed: +0.0055 [-0.0224,+0.0312]
- Positive split share: 0.667
