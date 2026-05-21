# IR Router Paired Uncertainty

| policy | queries | utility | paired diff vs fixed full (95% CI) |
| --- | ---: | ---: | --- |
| fixed_pq | 1148 | 0.3189 | -0.0276 [-0.0385, -0.0168] |
| fixed_full | 1148 | 0.3465 | -- |
| summary_qpp_b0 | 1148 | 0.3468 | 0.0003 [-0.0056, 0.0065] |
| query_summary_rf_b0 | 1148 | 0.3493 | 0.0029 [-0.0031, 0.0085] |
| summary_pq_qpp_b1 | 1148 | 0.3498 | 0.0033 [-0.0021, 0.0087] |
| query_summary_pq_rf_b1 | 1148 | 0.3454 | -0.0010 [-0.0070, 0.0048] |

## Per-dataset held-out utility

| dataset | test q | fixed full | B0 QPP | B0 RF | B1 QPP+PQ | B1 RF+PQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beir/fiqa/test | 259 | 0.318 | 0.318 | 0.318 | 0.319 | 0.314 |
| beir/scifact/test | 120 | 0.591 | 0.591 | 0.589 | 0.601 | 0.590 |
| beir/nfcorpus/test | 129 | 0.265 | 0.262 | 0.263 | 0.264 | 0.256 |
| beir/arguana | 560 | 0.328 | 0.329 | 0.335 | 0.332 | 0.330 |
| antique/test | 80 | 0.334 | 0.334 | 0.334 | 0.336 | 0.334 |
