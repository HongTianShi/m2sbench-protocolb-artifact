# Existing Retrieval Pipelines as Protocol B Solvers

| solver | paradigm | tier/menu | utility | regret | gap closed | buy/full | CI or diagnostic |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Oracle route | evaluator-only upper bound | invalid / S/PQ/F | 0.4309 | 0.0000 | 1.000 | 0.67/0.32 | evaluator-only |
| CE cheap-only gate | expensive-view purchase | B0/B1 / cheap/CE | 0.3777 | 0.0410 | 0.380 | 0.44/0.44 | AUPRC 0.368; Brier 0.194; ECE 0.048 |
| RF utility predictor | cost-sensitive utility prediction | B1 / S/PQ/F | 0.3598 | 0.0711 | 0.102 | 0.98/0.61 | +0.0081 [+0.0025,+0.0138] |
| B1 RF feature ablation | legal feature ablation | B1 / S/PQ/F | 0.3585 | 0.0724 | 0.086 | 0.98/0.62 | +0.0068 [+0.0015,+0.0122] |
| Selective rerank RF | selective reranking | B1 / PQ/F | 0.3583 | 0.0726 | 0.084 | 1.00/0.46 | +0.0066 [+0.0004,+0.0129] |
| ET utility predictor | cost-sensitive utility prediction | B1 / S/PQ/F | 0.3582 | 0.0727 | 0.082 | 0.98/0.62 | +0.0065 [+0.0011,+0.0119] |
| B0 RF feature ablation | legal feature ablation | B0 / S/PQ/F | 0.3565 | 0.0744 | 0.060 | 0.99/0.69 | +0.0048 [-0.0003,+0.0099] |
| Ridge visible predictor | linear utility prediction | B1 / S/PQ/F | 0.3550 | 0.0759 | 0.042 | 0.93/0.61 | +0.0033 [-0.0030,+0.0095] |
| Fixed full | fixed richest view | paid / F | 0.3517 | 0.0792 | 0.000 | 1.00/1.00 | -- |
| HistGBR utility predictor | cost-sensitive utility prediction | B1 / S/PQ/F | 0.3504 | 0.0805 | -0.017 | 0.91/0.53 | -0.0013 [-0.0093,+0.0064] |
| Cascade exit gate | cascade | B1 / S/PQ/F | 0.3494 | 0.0815 | -0.029 | 0.95/0.89 | -0.0023 [-0.0059,+0.0013] |
| QPP uncertainty gate | QPP/uncertainty | B0 / S/F | 0.3468 | 0.0732 | -0.062 | 1.00/0.51 | -- |
| ANN-depth agreement gate | ANN-depth routing | B1 / H16/H64 | 0.3421 | 0.0048 | 0.587 | 1.00/0.00 | +.0069 (FiQA HNSW) |
| Learning-to-defer classifier | deferral classifier | B1 / PQ/F | 0.3367 | 0.0942 | -0.190 | 1.00/0.16 | -0.0150 [-0.0248,-0.0056] |
| Fixed PQ | fixed compressed view | B1 / PQ | 0.3240 | 0.1069 | -0.350 | 1.00/0.00 | -0.0277 [-0.0387,-0.0171] |
| Fixed summary | fixed view | B0 / S | 0.1389 | 0.2920 | -2.686 | 0.00/0.00 | -0.2128 [-0.2330,-0.1930] |

## Hidden-Test Dry Run

- Repeated split adaptive-best-fixed: +0.0066 [+0.0025,+0.0101], positive share 0.967.

| held-out dataset | best fixed | best adaptive | delta |
| --- | --- | --- | ---: |
| beir/fiqa/test | fixed_full 0.3223 | rf_evidence_cascade 0.3226 | +0.0003 |
| beir/scifact/test | fixed_full 0.5987 | et_evidence_cascade 0.6037 | +0.0050 |
| beir/nfcorpus/test | fixed_full 0.2708 | ridge_evidence_cascade 0.2721 | +0.0013 |
| beir/arguana | fixed_pq 0.3336 | rf_evidence_cascade 0.3348 | +0.0012 |
| antique/test | fixed_full 0.3523 | two_threshold_cascade 0.3492 | -0.0031 |

## Lambda Sweep

| lambda | fixed S | fixed PQ | fixed F | cascade | RF | selective RF | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.1353 | 0.3344 | 0.3951 | 0.3914 | 0.3939 | 0.3883 | 0.4449 |
| 0.02 | 0.1353 | 0.3304 | 0.3835 | 0.3809 | 0.3854 | 0.3808 | 0.4395 |
| 0.05 | 0.1353 | 0.3244 | 0.3661 | 0.3651 | 0.3726 | 0.3696 | 0.4315 |
| 0.08 | 0.1353 | 0.3184 | 0.3487 | 0.3494 | 0.3598 | 0.3583 | 0.4238 |
| 0.12 | 0.1353 | 0.3105 | 0.3255 | 0.3283 | 0.3427 | 0.3434 | 0.4138 |
| 0.20 | 0.1353 | 0.2945 | 0.2791 | 0.2863 | 0.3086 | 0.3134 | 0.3950 |
