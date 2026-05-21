# 2Wiki Structured Evidence Access Audit

Protocol B structured-evidence slice over `voidful/2wikimultihopqa`. A method sees a question plus a cheap title/entity sketch and may buy one of five declared views: summary, one-hop hyperlinks, two-hop paths, full context, or cross-encoder reranking. Evaluator-held supporting facts and evidence triples define the qrels; the primary score is support-title NDCG@k with cost-adjusted utility.

- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Top-k: 4; lambda: 0.08
- View costs: {'summary': 0.0, 'one_hop': 0.16, 'two_hop': 0.26, 'full_context': 0.55, 'ce': 0.95}
- Oracle view share on hidden test: {'one_hop': 0.445, 'summary': 0.3925, 'two_hop': 0.0925, 'ce': 0.0475, 'full_context': 0.0225}

Metric aliases: the menus use `support_f1` for the report column sometimes
called support-fact F1 in prose. Control and feature-tier definitions are
collected in `docs/2wiki_data_card.md`.

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | ce_buy | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7774 | 0.7774 | 0.7331 | 0 | 0.1731 | -2.4398 | 0 | {'summary': 1.0} |
| fixed_one_hop | 0.9002 | 0.913 | 0.9363 | 0.16 | 0.0503 | 0 | 0 | {'one_hop': 1.0} |
| fixed_two_hop | 0.8977 | 0.9185 | 0.9381 | 0.26 | 0.0527 | -0.0482 | 0 | {'two_hop': 1.0} |
| fixed_full_context | 0.6602 | 0.7042 | 0.7294 | 0.55 | 0.2902 | -4.7689 | 0 | {'full_context': 1.0} |
| fixed_ce | 0.7546 | 0.8306 | 0.8119 | 0.95 | 0.1959 | -2.8935 | 1 | {'ce': 1.0} |
| structured_gate | 0.7883 | 0.7926 | 0.7588 | 0.0539 | 0.1622 | -2.2241 | 0 | {'summary': 0.7925, 'two_hop': 0.2075} |
| et_B1_context | 0.9239 | 0.9356 | 0.9419 | 0.1462 | 0.0266 | 0.4714 | 0.015 | {'summary': 0.335, 'one_hop': 0.37, 'two_hop': 0.28, 'ce': 0.015} |
| oracle_route | 0.9505 | 0.9627 | 0.9656 | 0.1527 | 0 | 1 | 0.0475 | {'summary': 0.3925, 'one_hop': 0.445, 'two_hop': 0.0925, 'full_context': 0.0225, 'ce': 0.0475} |

## Legal feature-tier learner ablation

Rows use only released features at the named tier; paid CE/full/view scores are not visible before purchase.

| feature_tier | policy | utility | regret | gap_closed | diff_vs_best_fixed | ce_buy | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0_title | et_B0_title | 0.9001 | 0.0504 | -0.0016 | -0.0001 | 0.005 | {'summary': 0.16, 'one_hop': 0.5125, 'two_hop': 0.3225, 'ce': 0.005} |
| B1_entity | et_B1_entity | 0.9125 | 0.038 | 0.2453 | 0.0123 | 0.01 | {'summary': 0.2, 'one_hop': 0.4425, 'two_hop': 0.3475, 'ce': 0.01} |
| B1_graph | et_B1_graph | 0.918 | 0.0325 | 0.3535 | 0.0178 | 0.0275 | {'summary': 0.3325, 'one_hop': 0.3425, 'two_hop': 0.2975, 'ce': 0.0275} |
| B1_path | et_B1_path | 0.9224 | 0.028 | 0.4425 | 0.0223 | 0.025 | {'summary': 0.35, 'one_hop': 0.3525, 'two_hop': 0.2725, 'ce': 0.025} |
| B1_context | et_B1_context | 0.9239 | 0.0266 | 0.4714 | 0.0237 | 0.015 | {'summary': 0.335, 'one_hop': 0.37, 'two_hop': 0.28, 'ce': 0.015} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7774 | 0.7774 | 0.7331 | 0.5265 | 0.3673 | 0 | 0.1731 | -0.1227 |
| real_1hop | 0.9002 | 0.913 | 0.9363 | 0.6837 | 0.4949 | 0.16 | 0.0503 | 0 |
| real_2hop | 0.8977 | 0.9185 | 0.9381 | 0.6873 | 0.4983 | 0.26 | 0.0527 | -0.0024 |
| shuffled_1hop | 0.7054 | 0.7182 | 0.7125 | 0.5129 | 0.36 | 0.16 | 0.245 | -0.1947 |
| degree_random_1hop | 0.7086 | 0.7214 | 0.7206 | 0.5194 | 0.3647 | 0.16 | 0.2418 | -0.1915 |
| random_2hop | 0.6886 | 0.7094 | 0.7244 | 0.5227 | 0.3703 | 0.26 | 0.2619 | -0.2116 |
| full_context | 0.6602 | 0.7042 | 0.7294 | 0.5285 | 0.3777 | 0.55 | 0.2902 | -0.2399 |
| ce | 0.7546 | 0.8306 | 0.8119 | 0.581 | 0.4151 | 0.95 | 0.1959 | -0.1456 |

## Question-type breakdown

| type | n | best_fixed | fixed_utility | adaptive_utility | oracle_utility | gain | oracle_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bridge_comparison | 83 | two_hop | 0.8873 | 0.8928 | 0.9088 | 0.0055 | 0.0216 |
| comparison | 127 | summary | 0.9857 | 0.9776 | 0.991 | -0.0081 | 0.0053 |
| compositional | 185 | two_hop | 0.8986 | 0.8993 | 0.9402 | 0.0007 | 0.0415 |
| inference | 5 | one_hop | 0.9872 | 0.9856 | 0.9949 | -0.0016 | 0.0077 |

## Repeated split stability

```json
{
  "mean_learned_minus_best_fixed": 0.024014684058544133,
  "ci95": [
    0.018562255184816752,
    0.029037799823964854
  ],
  "positive_share": 1.0,
  "selected_learners": {
    "et_B1_context": 4,
    "rf_B1_path": 4,
    "et_B1_path": 2
  },
  "selected_tiers": {
    "B1_context": 4,
    "B1_path": 6
  }
}
```

## Hard structured-evidence buckets

| bucket | share | ce_worth_share | oracle_regret_fixed_best | summary_ndcg | two_hop_ndcg | ce_ndcg |
| --- | --- | --- | --- | --- | --- | --- |
| low title margin | 0.33 | 0.0455 | 0.0684 | 0.8195 | 0.8993 | 0.8331 |
| high graph density | 0.3375 | 0.0222 | 0.0932 | 0.7174 | 0.8461 | 0.7342 |
| high full-context entropy | 0.33 | 0.053 | 0.0657 | 0.8047 | 0.9199 | 0.8235 |

## Protocol B legal/illegal trace

The JSON report contains a generated trace with visible fields, an allowed 2Wiki view purchase, the submitted JSONL route, charged cost, hidden evaluator fields, and an illegal variant containing unpaid CE/support fields. This is an evidence-acquisition trace rather than an end-to-end RAG transcript.

## Notes

- This is a structured evidence-acquisition slice, not an end-to-end RAG or generation benchmark.
- The hyperlink corpus is used only to construct paid 1-hop/2-hop graph views; gold supporting facts and evidence triples remain evaluator-held.
- CE scores are treated as paid evidence and are not visible to summary/graph/full-context routers.
