# 2Wiki Structured Evidence Access Audit

Protocol B structured-evidence slice over `voidful/2wikimultihopqa`. A method sees a question plus a cheap title/entity sketch and may buy one of five declared views: summary, one-hop hyperlinks, two-hop paths, full context, or cross-encoder reranking. Evaluator-held supporting facts and evidence triples define the qrels; the primary score is support-title NDCG@k with cost-adjusted utility.

- Rows: 100 ({'train': 60, 'dev': 20, 'test': 20})
- Top-k: 4; lambda: 0.08
- View costs: {'summary': 0.0, 'one_hop': 0.16, 'two_hop': 0.26, 'full_context': 0.55, 'ce': 0.95}
- Oracle view share on hidden test: {'one_hop': 0.45, 'summary': 0.5, 'two_hop': 0.05}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | ce_buy | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7685 | 0.7685 | 0.7 | 0 | 0.1764 | -2.5148 | 0 | {'summary': 1.0} |
| fixed_one_hop | 0.8904 | 0.9032 | 0.9375 | 0.16 | 0.0545 | -0.086 | 0 | {'one_hop': 1.0} |
| fixed_two_hop | 0.8947 | 0.9155 | 0.95 | 0.26 | 0.0502 | 0 | 0 | {'two_hop': 1.0} |
| fixed_full_context | 0.6016 | 0.6456 | 0.65 | 0.55 | 0.3433 | -5.8393 | 0 | {'full_context': 1.0} |
| fixed_ce | 0.5696 | 0.6456 | 0.65 | 0.95 | 0.3753 | -6.4768 | 1 | {'ce': 1.0} |
| structured_gate | 0.7716 | 0.7781 | 0.725 | 0.081 | 0.1733 | -2.4525 | 0 | {'summary': 0.8, 'two_hop': 0.1, 'full_context': 0.1} |
| et_B1_path | 0.9252 | 0.9364 | 0.95 | 0.14 | 0.0197 | 0.607 | 0 | {'summary': 0.25, 'one_hop': 0.55, 'two_hop': 0.2} |
| oracle_route | 0.9449 | 0.9517 | 0.95 | 0.085 | 0 | 1 | 0 | {'summary': 0.5, 'one_hop': 0.45, 'two_hop': 0.05} |

## Legal feature-tier learner ablation

Rows use only released features at the named tier; paid CE/full/view scores are not visible before purchase.

| feature_tier | policy | utility | regret | gap_closed | diff_vs_best_fixed | ce_buy | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0_title | hgb_B0_title | 0.9196 | 0.0254 | 0.4947 | 0.0248 | 0.05 | {'summary': 0.15, 'one_hop': 0.45, 'two_hop': 0.35, 'ce': 0.05} |
| B1_entity | et_B1_entity | 0.8972 | 0.0477 | 0.0495 | 0.0025 | 0 | {'summary': 0.2, 'one_hop': 0.5, 'two_hop': 0.3} |
| B1_graph | rf_B1_graph | 0.8794 | 0.0655 | -0.3044 | -0.0153 | 0 | {'summary': 0.2, 'one_hop': 0.45, 'two_hop': 0.25, 'full_context': 0.1} |
| B1_path | et_B1_path | 0.9252 | 0.0197 | 0.607 | 0.0305 | 0 | {'summary': 0.25, 'one_hop': 0.55, 'two_hop': 0.2} |
| B1_context | et_B1_context | 0.872 | 0.073 | -0.4533 | -0.0228 | 0 | {'summary': 0.3, 'one_hop': 0.65, 'full_context': 0.05} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7685 | 0.7685 | 0.7 | 0.5167 | 0.361 | 0 | 0.1764 | -0.1262 |
| real_1hop | 0.8904 | 0.9032 | 0.9375 | 0.7042 | 0.5149 | 0.16 | 0.0545 | -0.0043 |
| real_2hop | 0.8947 | 0.9155 | 0.95 | 0.7167 | 0.5249 | 0.26 | 0.0502 | 0 |
| shuffled_1hop | 0.736 | 0.7488 | 0.725 | 0.5333 | 0.3753 | 0.16 | 0.2089 | -0.1587 |
| degree_random_1hop | 0.716 | 0.7288 | 0.725 | 0.5417 | 0.3801 | 0.16 | 0.229 | -0.1788 |
| random_2hop | 0.7436 | 0.7644 | 0.7625 | 0.5625 | 0.4005 | 0.26 | 0.2013 | -0.1511 |
| full_context | 0.6016 | 0.6456 | 0.65 | 0.4833 | 0.3494 | 0.55 | 0.3433 | -0.2931 |
| ce | 0.5696 | 0.6456 | 0.65 | 0.4833 | 0.3494 | 0.95 | 0.3753 | -0.3251 |

## Question-type breakdown

| type | n | best_fixed | fixed_utility | adaptive_utility | oracle_utility | gain | oracle_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bridge_comparison | 6 | two_hop | 0.7755 | 0.8547 | 0.856 | 0.0792 | 0.0805 |
| comparison | 6 | summary | 1 | 0.9425 | 1 | -0.0575 | 0 |
| compositional | 8 | one_hop | 0.9671 | 0.9651 | 0.9703 | -0.002 | 0.0032 |

## Repeated split stability

```json
{
  "mean_learned_minus_best_fixed": 0.018768679819136658,
  "ci95": [
    0.018768679819136658,
    0.018768679819136658
  ],
  "positive_share": 1.0,
  "selected_learners": {
    "ridge_B1_path": 1
  },
  "selected_tiers": {
    "B1_path": 1
  }
}
```

## Hard structured-evidence buckets

| bucket | share | ce_worth_share | oracle_regret_fixed_best | summary_ndcg | two_hop_ndcg | ce_ndcg |
| --- | --- | --- | --- | --- | --- | --- |
| low title margin | 0.35 | 0 | 0.0561 | 0.7817 | 0.9562 | 0.8114 |
| high graph density | 0.5 | 0 | 0.0834 | 0.653 | 0.8391 | 0.5969 |
| high full-context entropy | 0.35 | 0 | 0.0797 | 0.7745 | 0.8726 | 0.6462 |

## Notes

- This is a structured evidence-acquisition slice, not an end-to-end RAG or generation benchmark.
- The hyperlink corpus is used only to construct paid 1-hop/2-hop graph views; gold supporting facts and evidence triples remain evaluator-held.
- CE scores are treated as paid evidence and are not visible to summary/graph/full-context routers.