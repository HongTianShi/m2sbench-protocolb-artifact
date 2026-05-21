# StrategyQA Reasoning Evidence Audit

Exploratory Protocol B reasoning-evidence slice over StrategyQA. A method sees the question, term, description, and decomposition sketch, then may buy fact snippets or a full fact set before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: candidate_reasoning_slice
- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Split: `train`; top-k: 4; lambda: 0.08; candidate pool: 24
- Oracle view share: {'summary': 0.585, 'full_fact_set': 0.17, 'decomposition': 0.2425, 'fact_snippets': 0.0025}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.8624 | 0.8624 | 0.835 | 0.6491 | 0 | 0.0879 | -1.0158 | -0.0443 | {'summary': 1.0} |
| fixed_decomposition | 0.9067 | 0.9099 | 0.888 | 0.6913 | 0.04 | 0.0436 | 0 | 0 | {'decomposition': 1.0} |
| fixed_fact_snippets | 0.8872 | 0.9048 | 0.8839 | 0.6879 | 0.22 | 0.0631 | -0.447 | -0.0195 | {'fact_snippets': 1.0} |
| fixed_full_fact_set | 0.8852 | 0.9612 | 0.9454 | 0.7397 | 0.95 | 0.0651 | -0.4923 | -0.0215 | {'full_fact_set': 1.0} |
| threshold_gate(summary,decomposition) | 0.9041 | 0.9071 | 0.885 | 0.6886 | 0.0381 | 0.0462 | -0.06 | -0.0026 | {'summary': 0.0475, 'decomposition': 0.9525} |
| best_legal_router(hgb_B0_sketch) | 0.9093 | 0.9204 | 0.8975 | 0.6992 | 0.139 | 0.041 | 0.06 | 0.0026 | {'summary': 0.095, 'decomposition': 0.77, 'fact_snippets': 0.0275, 'full_fact_set': 0.1075} |
| best_legal_adaptive | 0.9093 | 0.9204 | 0.8975 | 0.6992 | 0.139 | 0.041 | 0.06 | 0.0026 | {'summary': 0.095, 'decomposition': 0.77, 'fact_snippets': 0.0275, 'full_fact_set': 0.1075} |
| oracle_route | 0.9503 | 0.964 | 0.9489 | 0.743 | 0.1717 | 0 | 1 | 0.0436 | {'summary': 0.585, 'decomposition': 0.2425, 'fact_snippets': 0.0025, 'full_fact_set': 0.17} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| summary_only | 0.8624 | 0.8624 | 0.835 | 0.6491 | 0 | 0.0879 | -0.0443 |
| decomposition_only | 0.9067 | 0.9099 | 0.888 | 0.6913 | 0.04 | 0.0436 | 0 |
| real_fact_snippets | 0.8872 | 0.9048 | 0.8839 | 0.6879 | 0.22 | 0.0631 | -0.0195 |
| full_fact_set | 0.8852 | 0.9612 | 0.9454 | 0.7397 | 0.95 | 0.0651 | -0.0215 |
| shuffled_facts | 0.1374 | 0.155 | 0.1777 | 0.1383 | 0.22 | 0.8129 | -0.7693 |
| cross_question_facts | 0.1249 | 0.1425 | 0.1634 | 0.1291 | 0.22 | 0.8254 | -0.7818 |
| same_count_random | 0.1343 | 0.1519 | 0.1716 | 0.1345 | 0.22 | 0.816 | -0.7724 |
| decomposition_shuffled | 0.6981 | 0.7157 | 0.7002 | 0.5419 | 0.22 | 0.2522 | -0.2086 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hgb_B1_fact_meta | B1_fact_meta | 0.9133 | 0.132 | 0.037 | 0.0067 | 0.1526 | {'summary': 0.055, 'decomposition': 0.8275, 'fact_snippets': 0.0175, 'full_fact_set': 0.1} |
| rf_B1_context_sketch | B1_context_sketch | 0.9126 | 0.2211 | 0.0377 | 0.0059 | 0.1352 | {'summary': 0.0675, 'decomposition': 0.7225, 'fact_snippets': 0.01, 'full_fact_set': 0.2} |
| et_B1_fact_meta | B1_fact_meta | 0.9125 | 0.1715 | 0.0378 | 0.0058 | 0.1338 | {'summary': 0.0575, 'decomposition': 0.7875, 'fact_snippets': 0.01, 'full_fact_set': 0.145} |
| rf_B0_sketch | B0_sketch | 0.9116 | 0.2212 | 0.0387 | 0.0049 | 0.1129 | {'summary': 0.0875, 'decomposition': 0.6975, 'fact_snippets': 0.015, 'full_fact_set': 0.2} |
| hgb_B1_context_sketch | B1_context_sketch | 0.9107 | 0.144 | 0.0397 | 0.004 | 0.0912 | {'summary': 0.05, 'decomposition': 0.8175, 'fact_snippets': 0.02, 'full_fact_set': 0.1125} |
| et_B1_context_sketch | B1_context_sketch | 0.9105 | 0.1782 | 0.0398 | 0.0039 | 0.0885 | {'summary': 0.07, 'decomposition': 0.775, 'full_fact_set': 0.155} |
| rf_B1_fact_meta | B1_fact_meta | 0.9102 | 0.2182 | 0.0401 | 0.0035 | 0.081 | {'summary': 0.0725, 'decomposition': 0.7225, 'fact_snippets': 0.0075, 'full_fact_set': 0.1975} |
| hgb_B0_sketch | B0_sketch | 0.9093 | 0.139 | 0.041 | 0.0026 | 0.06 | {'summary': 0.095, 'decomposition': 0.77, 'fact_snippets': 0.0275, 'full_fact_set': 0.1075} |

## Answer-type oracle headroom

| answer | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| False | 207 | fixed_decomposition | 0.9074 | 0.9553 | 0.048 |
| True | 193 | fixed_decomposition | 0.9059 | 0.9449 | 0.039 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.003916861256832027,
  "ci95": [
    0.0005309166953813948,
    0.007219175851027527
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(hgb_B0_sketch)": 2,
    "best_legal_router(et_B1_fact_meta)": 2,
    "best_legal_router(hgb_B1_fact_meta)": 1,
    "best_legal_router(rf_B0_sketch)": 1,
    "best_legal_router(hgb_B1_context_sketch)": 2,
    "best_legal_router(rf_B1_fact_meta)": 1,
    "best_legal_router(rf_B1_context_sketch)": 1
  },
  "repeats": 10
}
```

## Protocol B legal/illegal trace

```json
{
  "visible_fields": {
    "query_id": "265dd54c248f8b048851",
    "question": "Do the anchors on Rede Globo speak Chinese?",
    "term": "Rede Globo",
    "description": "Brazilian commercial television network",
    "decomposition": [
      "What country broadcasts Rede Globo?",
      "What is the official language of #1?",
      "Is #2 Chinese?"
    ],
    "declared_views": [
      "summary",
      "decomposition",
      "fact_snippets",
      "full_fact_set"
    ],
    "released_state": "question, term, description, decomposition text, and coarse fact-pool metadata; support facts, answer, and unpaid fact/full scores are hidden"
  },
  "legal_action": {
    "submitted_jsonl": {
      "query_id": "265dd54c248f8b048851",
      "cell_id": "strategyqa_265dd54c248f8b048851",
      "tier": "B1_reasoning",
      "cost_menu": "strategyqa-reasoning-v0-op",
      "ranked_views": [
        "decomposition",
        "fact_snippets",
        "full_fact_set"
      ],
      "route": "decomposition"
    },
    "charged_cost": 0.04,
    "scored_utility": 0.9968
  },
  "hidden_evaluator_fields": {
    "facts": [
      "Rede Globo is a Brazilian television network.",
      "The official language of Brazil is Portuguese."
    ],
    "answer": false
  },
  "illegal_variant": {
    "submitted_jsonl": {
      "query_id": "265dd54c248f8b048851",
      "cell_id": "strategyqa_265dd54c248f8b048851",
      "tier": "B1_reasoning",
      "cost_menu": "strategyqa-reasoning-v0-op",
      "route": "fact_snippets",
      "facts": [
        "Rede Globo is a Brazilian television network.",
        "The official language of Brazil is Portuguese."
      ],
      "answer": false
    },
    "rejection_reason": "facts and answer are evaluator-only fields for this StrategyQA reasoning slice."
  },
  "scope_note": "This is exploratory reasoning-evidence acquisition, not answer generation or an official leaderboard slice."
}
```