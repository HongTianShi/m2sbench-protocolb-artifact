# 2Wiki Lambda Sensitivity From Frozen Reports

This audit rescales the frozen aggregate policy rows with `U_lambda = raw - lambda * mean_cost`.
It does not rebuild the 7GB corpus or retrain routers; the goal is to check whether the main 2Wiki gains vanish under small changes to lambda.

Readout: in both slices, the selected legal adaptive row has higher raw score and lower mean cost than the best fixed 1-hop row at the default profile, so the advantage is not a narrow `lambda=.08` crossing.

## 2wiki_context

- Source: `reports/2wiki_structured_evidence_2000.json`
- Frozen adaptive policy: `et_B1_context`
- Adaptive raw/cost: 0.936 / 0.146

| lambda | best fixed | fixed utility | adaptive utility | adaptive - fixed |
| ---: | --- | ---: | ---: | ---: |
| 0.00 | `fixed_two_hop` | 0.919 | 0.936 | +0.017 |
| 0.04 | `fixed_two_hop` | 0.908 | 0.930 | +0.022 |
| 0.08 | `fixed_one_hop` | 0.900 | 0.924 | +0.024 |
| 0.16 | `fixed_one_hop` | 0.887 | 0.912 | +0.025 |
| 0.32 | `fixed_one_hop` | 0.862 | 0.889 | +0.027 |

## 2wiki_hyperlink_10k

- Source: `reports/2wiki_hyperlink_corpus_stress_10000.json`
- Frozen adaptive policy: `best_legal_adaptive`
- Adaptive raw/cost: 0.828 / 0.113

| lambda | best fixed | fixed utility | adaptive utility | adaptive - fixed |
| ---: | --- | ---: | ---: | ---: |
| 0.00 | `fixed_hyperlink_2hop` | 0.817 | 0.828 | +0.011 |
| 0.04 | `fixed_hyperlink_2hop` | 0.804 | 0.823 | +0.020 |
| 0.08 | `fixed_hyperlink_1hop` | 0.791 | 0.819 | +0.028 |
| 0.16 | `fixed_hyperlink_1hop` | 0.775 | 0.810 | +0.035 |
| 0.32 | `fixed_summary` | 0.774 | 0.792 | +0.018 |

