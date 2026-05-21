# Protocol B Standard-Qrel 60/20/20 Learning Audit

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Total queries: 2,872
- Displayed split: train 1724, dev 575, test 573
- Sanity raw NDCG: summary 0.1353, PQ 0.3360, full 0.3951

## Displayed hidden-test split

| solver | utility | NDCG | cost | regret | gap closed | choices | paired diff vs best fixed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| oracle_route | 0.4412 | 0.4623 | 0.2629 | 0.0000 | 1.00 | summary 0.31, pq 0.37, full 0.33 | +0.0849 [+0.0733,+0.0970] |
| rf_utility_b1 | 0.3660 | 0.3994 | 0.4183 | 0.0753 | 0.11 | summary 0.02, pq 0.39, full 0.59 | +0.0097 [+0.0012,+0.0183] |
| cascade_exit_gate | 0.3647 | 0.3993 | 0.4327 | 0.0766 | 0.10 | summary 0.04, pq 0.33, full 0.63 | +0.0084 [-0.0002,+0.0173] |
| rf_utility_b0 | 0.3622 | 0.3993 | 0.4632 | 0.0790 | 0.07 | summary 0.01, pq 0.29, full 0.70 | +0.0059 [-0.0019,+0.0138] |
| et_utility_b1 | 0.3622 | 0.3964 | 0.4269 | 0.0790 | 0.07 | summary 0.02, pq 0.37, full 0.61 | +0.0059 [-0.0024,+0.0142] |
| ridge_utility_b1 | 0.3613 | 0.3949 | 0.4200 | 0.0800 | 0.06 | summary 0.05, pq 0.34, full 0.61 | +0.0050 [-0.0047,+0.0140] |
| hgb_utility_b1 | 0.3611 | 0.3920 | 0.3867 | 0.0801 | 0.06 | summary 0.10, pq 0.36, full 0.54 | +0.0048 [-0.0054,+0.0148] |
| qpp_uncertainty_gate | 0.3596 | 0.4000 | 0.5057 | 0.0816 | 0.04 | summary 0.00, pq 0.20, full 0.80 | +0.0033 [-0.0020,+0.0083] |
| selective_rerank_rf | 0.3583 | 0.3993 | 0.5117 | 0.0829 | 0.02 | summary 0.00, pq 0.18, full 0.82 | +0.0020 [-0.0039,+0.0074] |
| summary_margin_gate | 0.3578 | 0.4027 | 0.5608 | 0.0834 | 0.02 | summary 0.00, pq 0.05, full 0.95 | +0.0015 [-0.0011,+0.0041] |
| fixed_full | 0.3563 | 0.4027 | 0.5800 | 0.0849 | 0.00 | summary 0.00, pq 0.00, full 1.00 | -- |
| fixed_pq | 0.3281 | 0.3441 | 0.2000 | 0.1131 | -0.33 | summary 0.00, pq 1.00, full 0.00 | -0.0282 [-0.0449,-0.0130] |
| learning_to_defer_hgb | 0.3199 | 0.3397 | 0.2480 | 0.1213 | -0.43 | summary 0.31, pq 0.40, full 0.29 | -0.0364 [-0.0538,-0.0200] |
| fixed_summary | 0.1443 | 0.1443 | 0.0000 | 0.2969 | -2.50 | summary 1.00, pq 0.00, full 0.00 | -0.2120 [-0.2428,-0.1851] |

## Restricted-menu Protocol B solvers

| menu | legal choices | best fixed | best adaptive | oracle | gap closed | residual regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| QPP cheap/full | summary/full | full 0.3563 | qpp_uncertainty_gate 0.3596 | 0.4110 | 0.060 | 0.0515 |
| Selective rerank PQ/full | pq/full | full 0.3563 | selective_rerank_rf 0.3583 | 0.4054 | 0.041 | 0.0471 |
| Cascade S/PQ/full | summary/pq/full | full 0.3563 | cascade_exit_gate 0.3647 | 0.4412 | 0.098 | 0.0766 |
| Full S/PQ/full | summary/pq/full | full 0.3563 | rf_utility_b1 0.3660 | 0.4412 | 0.114 | 0.0753 |

## Full-purchase calibration

| target | prevalence | AUPRC | Brier | ECE | buy rate | route utility | route regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U(full)>U(PQ) | 0.372 | 0.429 | 0.258 | 0.138 | 0.791 | 0.3603 | 0.0809 |

## Repeated stratified 60/20/20 hidden-test dry run

- Repetitions: 30
- Best adaptive minus best fixed: +0.0076 [-0.0018,+0.0134]
- Positive split share: 0.933

## Public-dev probing simulation

- Mean rank correlation between public-dev and private-test candidate utilities: 0.624

| submission cap | public-dev utility | private-test utility | overfit gap |
| ---: | ---: | ---: | ---: |
| 1 | 0.1321 | 0.1389 | -0.0069 [-0.0282,0.0110] |
| 3 | 0.3465 | 0.3495 | -0.0029 [-0.0318,0.0255] |
| 10 | 0.3542 | 0.3527 | 0.0015 [-0.0303,0.0336] |

## Lambda sweep over raw fixed views

| lambda | fixed S | fixed PQ | fixed F | best fixed | oracle | oracle choices |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0.00 | 0.1353 | 0.3360 | 0.3951 | full | 0.4460 | summary 0.32, pq 0.34, full 0.34 |
| 0.02 | 0.1353 | 0.3320 | 0.3835 | full | 0.4406 | summary 0.32, pq 0.34, full 0.34 |
| 0.05 | 0.1353 | 0.3260 | 0.3661 | full | 0.4327 | summary 0.32, pq 0.35, full 0.33 |
| 0.08 | 0.1353 | 0.3200 | 0.3487 | full | 0.4250 | summary 0.33, pq 0.36, full 0.31 |
| 0.12 | 0.1353 | 0.3120 | 0.3255 | full | 0.4151 | summary 0.34, pq 0.38, full 0.29 |
| 0.20 | 0.1353 | 0.2960 | 0.2791 | pq | 0.3964 | summary 0.35, pq 0.40, full 0.24 |
