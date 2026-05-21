# Evidence Cascade Router Audit

- Total queries: 2,872
- Test queries in displayed split: 1,149
- Features: dataset_onehot+query_features+summary_scores+pq_scores+summary_pq_result_agreement

| route | utility | NDCG | cost | regret | diff vs fixed full | choices |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rf_evidence_cascade | 0.3598 | 0.3939 | 0.4267 | 0.0711 | +0.0081 [+0.0025,+0.0138] | summary 0.02, pq 0.38, full 0.61 |
| et_evidence_cascade | 0.3582 | 0.3928 | 0.4328 | 0.0727 | +0.0065 [+0.0011,+0.0119] | summary 0.02, pq 0.36, full 0.62 |
| ridge_evidence_cascade | 0.3550 | 0.3885 | 0.4193 | 0.0759 | +0.0033 [-0.0030,+0.0095] | summary 0.07, pq 0.32, full 0.61 |
| fixed_full | 0.3517 | 0.3981 | 0.5800 | 0.0792 | -- | full 1.00 |
| hgb_evidence_cascade | 0.3504 | 0.3811 | 0.3840 | 0.0805 | -0.0013 [-0.0093,+0.0064] | summary 0.09, pq 0.38, full 0.53 |
| two_threshold_cascade | 0.3494 | 0.3914 | 0.5259 | 0.0815 | -0.0023 [-0.0059,+0.0013] | summary 0.05, pq 0.06, full 0.89; summary if f5 ge 0.2679; else pq if f4 ge 0.2; else full |
| fixed_pq | 0.3240 | 0.3400 | 0.2000 | 0.1069 | -0.0277 [-0.0387,-0.0171] | pq 1.00 |
| fixed_summary | 0.1389 | 0.1389 | 0.0000 | 0.2920 | -0.2128 [-0.2330,-0.1930] | summary 1.00 |

## Cost-sensitive selective reranking baselines

| policy | utility | NDCG | cost | diff vs fixed full | choices |
| --- | ---: | ---: | ---: | ---: | --- |
| selective_rf_B1_no_dataset | 0.3583 | 0.3883 | 0.3743 | +0.0066 [+0.0004,+0.0129] | pq 0.54, full 0.46; full if predicted U(full)-U(pq) > 0.03165; else pq |
| selective_ridge_B1_full | 0.3581 | 0.3904 | 0.4041 | +0.0064 [+0.0006,+0.0123] | pq 0.46, full 0.54; full if predicted U(full)-U(pq) > 0.01629; else pq |
| selective_rf_B1_full | 0.3579 | 0.3877 | 0.3730 | +0.0062 [-0.0002,+0.0124] | pq 0.54, full 0.46; full if predicted U(full)-U(pq) > 0.03394; else pq |
| selective_ridge_B1_scores_agreement | 0.3574 | 0.3923 | 0.4358 | +0.0057 [+0.0011,+0.0101] | pq 0.38, full 0.62; full if predicted U(full)-U(pq) > 0.004629; else pq |
| selective_et_B0_no_dataset | 0.3573 | 0.3911 | 0.4222 | +0.0056 [+0.0004,+0.0109] | pq 0.42, full 0.58; full if predicted U(full)-U(pq) > 0.02126; else pq |
| selective_et_B1_full | 0.3570 | 0.3882 | 0.3905 | +0.0053 [-0.0006,+0.0111] | pq 0.50, full 0.50; full if predicted U(full)-U(pq) > 0.02122; else pq |
| selective_et_B1_no_dataset | 0.3563 | 0.3883 | 0.3994 | +0.0046 [-0.0011,+0.0103] | pq 0.48, full 0.52; full if predicted U(full)-U(pq) > 0.01989; else pq |
| selective_ridge_B1_no_dataset | 0.3549 | 0.3912 | 0.4540 | +0.0032 [-0.0017,+0.0081] | pq 0.33, full 0.67; full if predicted U(full)-U(pq) > -0.001053; else pq |
| selective_ridge_B0_no_dataset | 0.3545 | 0.3900 | 0.4437 | +0.0028 [-0.0024,+0.0080] | pq 0.36, full 0.64; full if predicted U(full)-U(pq) > 0.006537; else pq |
| selective_et_B1_scores_agreement | 0.3544 | 0.3837 | 0.3654 | +0.0027 [-0.0041,+0.0096] | pq 0.56, full 0.44; full if predicted U(full)-U(pq) > 0.03225; else pq |
| selective_rf_B1_scores_agreement | 0.3542 | 0.3826 | 0.3551 | +0.0025 [-0.0047,+0.0093] | pq 0.59, full 0.41; full if predicted U(full)-U(pq) > 0.03965; else pq |
| selective_rf_B0_no_dataset | 0.3532 | 0.3827 | 0.3687 | +0.0015 [-0.0052,+0.0083] | pq 0.56, full 0.44; full if predicted U(full)-U(pq) > 0.03608; else pq |

## Feature ablation on the displayed split

| feature set | utility | NDCG | cost | diff vs fixed full | choices |
| --- | ---: | ---: | ---: | ---: | --- |
| rf_B1_full | 0.3585 | 0.3931 | 0.4330 | +0.0068 [+0.0015,+0.0122] | summary 0.02, pq 0.36, full 0.62 |
| rf_B1_scores_agreement | 0.3574 | 0.3908 | 0.4180 | +0.0057 [-0.0008,+0.0125] | summary 0.05, pq 0.36, full 0.60 |
| rf_score_only | 0.3569 | 0.3911 | 0.4279 | +0.0052 [-0.0008,+0.0114] | summary 0.03, pq 0.35, full 0.62 |
| rf_B0_no_dataset | 0.3565 | 0.3933 | 0.4608 | +0.0048 [-0.0003,+0.0099] | summary 0.01, pq 0.30, full 0.69 |
| rf_B1_no_dataset | 0.3563 | 0.3908 | 0.4321 | +0.0046 [-0.0008,+0.0100] | summary 0.02, pq 0.36, full 0.62 |
| rf_B0_query_summary | 0.3546 | 0.3915 | 0.4611 | +0.0029 [-0.0022,+0.0079] | summary 0.01, pq 0.29, full 0.69 |
| rf_agreement_only | 0.3449 | 0.3784 | 0.4190 | -0.0068 [-0.0145,+0.0006] | summary 0.06, pq 0.33, full 0.61 |

## Lambda frontier for fixed views and oracle routing

| lambda | fixed S | fixed PQ | fixed F | best fixed | oracle | oracle choices |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0.00 | 0.1353 | 0.3344 | 0.3951 | full | 0.4449 | summary 0.32, pq 0.33, full 0.35 |
| 0.02 | 0.1353 | 0.3304 | 0.3835 | full | 0.4395 | summary 0.32, pq 0.33, full 0.35 |
| 0.05 | 0.1353 | 0.3244 | 0.3661 | full | 0.4315 | summary 0.32, pq 0.34, full 0.33 |
| 0.08 | 0.1353 | 0.3184 | 0.3487 | full | 0.4238 | summary 0.33, pq 0.35, full 0.32 |
| 0.12 | 0.1353 | 0.3105 | 0.3255 | full | 0.4138 | summary 0.34, pq 0.37, full 0.29 |
| 0.20 | 0.1353 | 0.2945 | 0.2791 | pq | 0.3950 | summary 0.35, pq 0.40, full 0.25 |

## Per-dataset read of the displayed split

| dataset | test q | fixed S | fixed PQ | fixed F | adaptive | diff vs F | choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| beir/fiqa/test | 254 | 0.0142 | 0.2373 | 0.3294 | 0.3265 | -0.0029 [-0.0071,+0.0002] | pq 0.03, full 0.97 |
| beir/scifact/test | 123 | 0.1474 | 0.5489 | 0.6101 | 0.6115 | +0.0013 [-0.0076,+0.0096] | pq 0.20, full 0.80 |
| beir/nfcorpus/test | 138 | 0.1425 | 0.2453 | 0.2480 | 0.2471 | -0.0009 [-0.0051,+0.0026] | pq 0.16, full 0.84 |
| beir/arguana | 558 | 0.2062 | 0.3378 | 0.3267 | 0.3444 | +0.0177 [+0.0067,+0.0290] | summary 0.04, pq 0.67, full 0.30 |
| antique/test | 76 | 0.0416 | 0.2904 | 0.3798 | 0.3813 | +0.0016 [-0.0060,+0.0087] | pq 0.08, full 0.92 |

## Split stability

- Repeated 60/40 splits: 2
- Best adaptive minus best fixed utility: +0.0071 [+0.0070, +0.0072]
- Positive split share: 1.000

## Leave-one-dataset-out stress

| held-out dataset | best fixed | best adaptive | delta | adaptive choices |
| --- | --- | --- | ---: | --- |
| beir/fiqa/test | fixed_full 0.3223 | rf_evidence_cascade 0.3226 | +0.0003 | pq 0.02, full 0.98 |
| beir/scifact/test | fixed_full 0.5987 | et_evidence_cascade 0.6037 | +0.0050 | pq 0.14, full 0.86 |
| beir/nfcorpus/test | fixed_full 0.2708 | ridge_evidence_cascade 0.2721 | +0.0013 | summary 0.02, pq 0.08, full 0.89 |
| beir/arguana | fixed_pq 0.3336 | rf_evidence_cascade 0.3348 | +0.0012 | pq 0.70, full 0.30 |
| antique/test | fixed_full 0.3523 | two_threshold_cascade 0.3492 | -0.0031 | pq 0.01, full 0.99 |
