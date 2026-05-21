# StrategyQA Reasoning Evidence Audit

Exploratory Protocol B reasoning-evidence slice over StrategyQA. A method sees the question, term, description, and decomposition sketch, then may buy fact snippets or a full fact set before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: candidate_reasoning_slice
- Rows: 100 ({'train': 60, 'dev': 20, 'test': 20})
- Split: `train`; top-k: 4; lambda: 0.08; candidate pool: 24
- Oracle view share: {'summary': 0.65, 'full_fact_set': 0.1, 'decomposition': 0.25}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.9012 | 0.9012 | 0.8675 | 0.6577 | 0 | 0.0759 | -0.1156 | -0.0079 | {'summary': 1.0} |
| fixed_decomposition | 0.909 | 0.9122 | 0.89 | 0.698 | 0.04 | 0.068 | 0 | 0 | {'decomposition': 1.0} |
| fixed_fact_snippets | 0.8933 | 0.9109 | 0.89 | 0.698 | 0.22 | 0.0838 | -0.2315 | -0.0158 | {'fact_snippets': 1.0} |
| fixed_full_fact_set | 0.8885 | 0.9645 | 0.9525 | 0.7438 | 0.95 | 0.0886 | -0.3018 | -0.0205 | {'full_fact_set': 1.0} |
| threshold_gate(summary,decomposition) | 0.9293 | 0.9317 | 0.9025 | 0.6938 | 0.03 | 0.0477 | 0.2986 | 0.0203 | {'summary': 0.25, 'decomposition': 0.75} |
| best_legal_router(hgb_B0_sketch) | 0.8694 | 0.8733 | 0.8425 | 0.6494 | 0.049 | 0.1076 | -0.5821 | -0.0396 | {'summary': 0.45, 'decomposition': 0.4, 'fact_snippets': 0.15} |
| best_legal_adaptive | 0.9293 | 0.9317 | 0.9025 | 0.6938 | 0.03 | 0.0477 | 0.2986 | 0.0203 | {'summary': 0.25, 'decomposition': 0.75} |
| oracle_route | 0.9771 | 0.9855 | 0.9775 | 0.7605 | 0.105 | 0 | 1 | 0.068 | {'summary': 0.65, 'decomposition': 0.25, 'full_fact_set': 0.1} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| summary_only | 0.9012 | 0.9012 | 0.8675 | 0.6577 | 0 | 0.0759 | -0.0079 |
| decomposition_only | 0.909 | 0.9122 | 0.89 | 0.698 | 0.04 | 0.068 | 0 |
| real_fact_snippets | 0.8933 | 0.9109 | 0.89 | 0.698 | 0.22 | 0.0838 | -0.0158 |
| full_fact_set | 0.8885 | 0.9645 | 0.9525 | 0.7438 | 0.95 | 0.0886 | -0.0205 |
| shuffled_facts | 0.0552 | 0.0728 | 0.085 | 0.0694 | 0.22 | 0.9219 | -0.8538 |
| cross_question_facts | 0.083 | 0.1006 | 0.1267 | 0.1004 | 0.22 | 0.8941 | -0.8261 |
| same_count_random | 0.0936 | 0.1112 | 0.1375 | 0.1054 | 0.22 | 0.8834 | -0.8154 |
| decomposition_shuffled | 0.6876 | 0.7052 | 0.6967 | 0.5333 | 0.22 | 0.2895 | -0.2215 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B0_sketch | B0_sketch | 0.9553 | 0.1915 | 0.0218 | 0.0463 | 0.6803 | {'summary': 0.3, 'decomposition': 0.4, 'fact_snippets': 0.15, 'full_fact_set': 0.15} |
| hgb_B1_context_sketch | B1_context_sketch | 0.9261 | 0.07 | 0.0509 | 0.0171 | 0.2516 | {'summary': 0.15, 'decomposition': 0.65, 'fact_snippets': 0.2} |
| hgb_B1_fact_meta | B1_fact_meta | 0.9171 | 0.0985 | 0.06 | 0.0081 | 0.1185 | {'summary': 0.35, 'decomposition': 0.45, 'fact_snippets': 0.15, 'full_fact_set': 0.05} |
| et_B0_sketch | B0_sketch | 0.9119 | 0.129 | 0.0651 | 0.0029 | 0.0428 | {'summary': 0.05, 'decomposition': 0.85, 'full_fact_set': 0.1} |
| et_B1_context_sketch | B1_context_sketch | 0.9067 | 0.1745 | 0.0704 | -0.0024 | -0.0346 | {'summary': 0.05, 'decomposition': 0.8, 'full_fact_set': 0.15} |
| et_B1_fact_meta | B1_fact_meta | 0.9067 | 0.1745 | 0.0704 | -0.0024 | -0.0346 | {'summary': 0.05, 'decomposition': 0.8, 'full_fact_set': 0.15} |
| rf_B1_fact_meta | B1_fact_meta | 0.8963 | 0.077 | 0.0808 | -0.0127 | -0.187 | {'summary': 0.2, 'decomposition': 0.55, 'fact_snippets': 0.25} |
| rf_B1_context_sketch | B1_context_sketch | 0.8963 | 0.077 | 0.0808 | -0.0127 | -0.187 | {'summary': 0.2, 'decomposition': 0.55, 'fact_snippets': 0.25} |

## Answer-type oracle headroom

| answer | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| False | 13 | fixed_decomposition | 0.9814 | 0.9934 | 0.0121 |
| True | 7 | fixed_summary | 0.8448 | 0.9467 | 0.1019 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": -0.0005025326213546899,
  "ci95": [
    -0.008559526187312583,
    0.007943308216451778
  ],
  "positive_share": 0.3333333333333333,
  "selected_policies": {
    "threshold_gate(summary,decomposition)": 3
  },
  "repeats": 3
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