# IR Strong Router and Oracle Headroom

Held-out pooled 60/40 split over the five public standard-qrel datasets. Features are method-visible: dataset ID, query embedding summaries, summary score summaries, and declared PQ score summaries for B1-style rows. Utility is NDCG@10 - 0.08 C.

| policy | utility | NDCG | cost | regret | choices | paired diff vs fixed full |
|---|---:|---:|---:|---:|---|---:|
| oracle_route | 0.4200 | 0.4404 | 0.2558 | 0.0000 | summary 0.33, pq 0.35, full 0.32 | +0.0735 [+0.0664,+0.0813] |
| rf_reg | 0.3509 | 0.3862 | 0.4410 | 0.0691 | summary 0.00, pq 0.36, full 0.64 | +0.0044 [-0.0008,+0.0094] |
| extra_trees_reg | 0.3494 | 0.3839 | 0.4308 | 0.0705 | summary 0.01, pq 0.38, full 0.61 | +0.0030 [-0.0024,+0.0085] |
| fixed_full | 0.3465 | 0.3929 | 0.5800 | 0.0735 | summary 0.00, pq 0.00, full 1.00 | -- |
| ridge_reg | 0.3436 | 0.3781 | 0.4310 | 0.0763 | summary 0.05, pq 0.31, full 0.64 | -0.0028 [-0.0090,+0.0033] |
| hist_gbr_reg | 0.3372 | 0.3694 | 0.4031 | 0.0828 | summary 0.05, pq 0.39, full 0.56 | -0.0093 [-0.0162,-0.0021] |
| extra_trees_cls | 0.3327 | 0.3557 | 0.2873 | 0.0873 | summary 0.22, pq 0.43, full 0.35 | -0.0137 [-0.0235,-0.0033] |
| rf_cls | 0.3280 | 0.3508 | 0.2840 | 0.0919 | summary 0.22, pq 0.45, full 0.34 | -0.0184 [-0.0281,-0.0081] |
| mlp_reg | 0.3231 | 0.3502 | 0.3390 | 0.0969 | summary 0.15, pq 0.41, full 0.44 | -0.0234 [-0.0327,-0.0143] |
| fixed_pq | 0.3189 | 0.3349 | 0.2000 | 0.1011 | summary 0.00, pq 1.00, full 0.00 | -0.0276 [-0.0387,-0.0165] |
| logistic_cls | 0.3104 | 0.3314 | 0.2633 | 0.1096 | summary 0.32, pq 0.34, full 0.34 | -0.0361 [-0.0482,-0.0242] |
| hist_gbc_cls | 0.3092 | 0.3293 | 0.2513 | 0.1108 | summary 0.29, pq 0.42, full 0.29 | -0.0372 [-0.0493,-0.0254] |
| mlp_cls | 0.2822 | 0.3000 | 0.2218 | 0.1378 | summary 0.37, pq 0.38, full 0.25 | -0.0642 [-0.0784,-0.0503] |
| fixed_summary | 0.1300 | 0.1300 | 0.0000 | 0.2900 | summary 1.00, pq 0.00, full 0.00 | -0.2165 [-0.2357,-0.1960] |
