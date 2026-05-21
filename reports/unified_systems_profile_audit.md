# Unified Systems Profile Audit

- Dataset: `beir/fiqa/test`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- CPU: Intel Core i7-14700HX (28 logical threads)
- Environment: WSL2/Ubuntu; Python 3.10.20; PyTorch 2.5.1+cu121; FAISS-GPU 1.11.0
- Documents/queries: 57,638/648
- Timing: batch size 64, concurrency 1, 220 index repetitions, 5 warmup queries.
- QPS* is `1e6 / p50_us`; sustained QPS is measured by repeated full passes over all profiler queries at concurrency 1.

| view | parents | inc C | cum C | device | bytes touched | index bytes | p50 us | p95 us | p99 us | QPS* | sustained QPS | C_op | C_mem | C_lat | raw NDCG |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | - | 0.000 | 0.000 | gpu | 786432 | 786432 | 2.12 | 3.54 | 4.57 | 470671 | 410454 | 0.000 | 0.009 | 2.92148e-05 | 0.019 |
| int8_dense | summary | 0.300 | 0.300 | gpu | 22132992 | 22132992 | 11.50 | 13.13 | 14.53 | 86936 | 74024 | 0.300 | 0.250 | 0.000108244 | 0.361 |
| binary_sign | summary | 0.080 | 0.080 | gpu | 2766624 | 2766624 | 11.42 | 13.29 | 15.23 | 87587 | 80746 | 0.080 | 0.031 | 0.000109511 | 0.287 |
| ivf_pq | summary | 0.200 | 0.200 | gpu | 86016 | 2169744 | 13.23 | 15.34 | 16.37 | 75604 | 72571 | 0.200 | 0.013 | 0.000126404 | 0.255 |
| hnsw16 | summary | 0.138 | 0.138 | cpu | 26624 | 95909632 | 32.57 | 70.56 | 110.69 | 30702 | 35349 | 0.138 | 0.542 | 0.000581604 | 0.354 |
| hnsw64 | hnsw16 | 0.113 | 0.250 | cpu | 106496 | 95909632 | 71.03 | 122.57 | 172.22 | 14078 | 11428 | 0.250 | 0.542 | 0.00101028 | 0.364 |
| full_dense | summary | 0.580 | 0.580 | gpu | 88531968 | 88531968 | 12.17 | 14.01 | 14.70 | 82186 | 79551 | 0.580 | 1.000 | 0.000115518 | 0.364 |
| cross_encoder_top50 | full_dense | 0.450 | 1.030 | gpu | 1638400 | 90000000 | 116900.67 | 121320.29 | 367335.02 | 9 | 8 | 1.030 | 0.518 | 1 | 0.376 |

## Profile formulas

- `materialized_bytes = bytes_touched + index_size_bytes`.
- `C_op(v) = cumulative_cost(v)` from the versioned dependency DAG.
- `C_mem(v) = materialized_bytes(v) / max_v materialized_bytes(v)`.
- `C_lat(v) = p95_us(v) / max_v p95_us(v)`; small non-CE values are printed with significant digits rather than rounded to zero.
- CE has two accounting contexts: incremental CE purchase cost `.45` after a first-stage pool exists, and cumulative ladder cost `1.03` when charged as a full text-reranking view.

## Profile winners

| profile | winner | utility |
| --- | --- | ---: |
| C_op | hnsw64 | 0.3436 |
| C_mem | int8_dense | 0.3410 |
| C_lat | full_dense | 0.3635 |

Rank correlations: op/mem 0.619, op/lat 0.524, mem/lat 0.548.
The intended readout is profile sensitivity: winners can change when the cost profile changes, so Protocol B stores raw quality, declared cost, latency, and bytes separately.
