# CE Label Budget Audit: beir/fiqa/test

- Boundary queries: 260
- Lambda: 0.080
- CE purchase cost: 0.450
- FAISS GPUs visible: 1

## Fixed policies on the CE-labeled split

| policy | raw NDCG | cost | utility | regret | CE buy rate | oracle gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed cheap | 0.364 | 0.000 | 0.364 | 0.059 | 0.000 | 0.000 |
| fixed CE | 0.376 | 0.450 | 0.340 | 0.082 | 1.000 | -0.398 |

## Surrogate-light CE adapter feature ablation

| feature budget | CE-labeled train share | utility | regret | CE buy rate | oracle gap closed |
| --- | ---: | ---: | ---: | ---: | ---: |
| score stats only | 0.60 | 0.371 | 0.047 | 0.269 | 0.284 |
| + margin/entropy | 0.60 | 0.378 | 0.041 | 0.442 | 0.380 |
| + cheap agreement | 0.60 | 0.363 | 0.056 | 0.356 | 0.155 |
| + cell/context proxies | 0.60 | 0.349 | 0.070 | 0.442 | -0.056 |

## Expensive-view labeling budget

| selector | CE label share | utility | regret | CE buy rate | oracle gap closed |
| --- | ---: | ---: | ---: | ---: | ---: |
| random CE labels | 0.05 | 0.363 | 0.066 | 0.382 | -0.322 |
| high-value cheap-ambiguous labels | 0.05 | 0.334 | 0.095 | 0.990 | -0.906 |
| random CE labels | 0.10 | 0.363 | 0.065 | 0.367 | -0.319 |
| high-value cheap-ambiguous labels | 0.10 | 0.361 | 0.067 | 0.413 | -0.356 |
| random CE labels | 0.20 | 0.364 | 0.064 | 0.346 | -0.297 |
| high-value cheap-ambiguous labels | 0.20 | 0.358 | 0.071 | 0.240 | -0.426 |

High-value labels are chosen by cheap-view ambiguity only: low margin, high entropy, and wide score spread. CE scores never enter method-visible features.
