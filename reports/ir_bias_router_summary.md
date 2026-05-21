# IR Qrel-Bias and Router Summary

## Held-out route baselines

| policy | queries | utility | ndcg | cost | regret |
| --- | --- | --- | --- | --- | --- |
| summary_pq_qpp_b1 | 1148 | 0.350 | 0.384 | 0.428 | 0.070 |
| query_summary_rf_b0 | 1148 | 0.349 | 0.384 | 0.427 | 0.071 |
| summary_qpp_b0 | 1148 | 0.347 | 0.378 | 0.393 | 0.073 |
| fixed_full | 1148 | 0.346 | 0.393 | 0.580 | 0.074 |
| query_summary_pq_rf_b1 | 1148 | 0.345 | 0.378 | 0.413 | 0.075 |
| selective_rerank_cascade_b1 | 1148 | 0.336 | 0.364 | 0.354 | 0.084 |
| fixed_pq | 1148 | 0.319 | 0.335 | 0.200 | 0.101 |
| fixed_summary | 1148 | 0.130 | 0.130 | 0.000 | 0.290 |

## Same-candidate nested audit: qrel and tie diagnostics

- Weighted candidate-pool recall@200: 0.836; hit@200: 0.957.
- Positive qrel density inside top-200 candidate pools: 0.0164.
- PQ >= full before cost decomposes into strict PQ wins 0.180, exact NDCG ties 0.435, and strict PQ losses 0.385.
- Mean PQ-full NDCG@10 difference is -0.052; full remains best on mean utility in the nested audit.

| dataset | queries | mean_qrels_per_query | candidate_top200_recall | positive_density_in_top200 | pq_strict_better_share | pq_equal_full_share | pq_strict_worse_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beir/fiqa/test | 648 | 2.633 | 0.781 | 0.010 | 0.125 | 0.477 | 0.398 |
| beir/scifact/test | 300 | 1.130 | 0.960 | 0.005 | 0.113 | 0.623 | 0.263 |
| beir/nfcorpus/test | 323 | 38.186 | 0.383 | 0.053 | 0.235 | 0.393 | 0.372 |
| beir/arguana | 1401 | 1.000 | 0.994 | 0.005 | 0.203 | 0.440 | 0.356 |
| antique/test | 200 | 32.945 | 0.462 | 0.075 | 0.205 | 0.050 | 0.745 |
