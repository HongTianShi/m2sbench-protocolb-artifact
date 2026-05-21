# FEVER Claim-Evidence Handle Audit

Exploratory Protocol B claim-evidence slice over FEVER official JSONL. A method sees a claim and cheap title/entity overlap handles, then may buy page-title or evidence-sentence handles before evaluator-held evidence handles are scored. The official JSONL used here does not include wiki sentence text or full page text, so this is a handle-acquisition audit rather than a full FEVER retrieval benchmark.

- Status: candidate_claim_verification_slice
- Rows: 100 ({'train': 60, 'dev': 20, 'test': 20})
- Top-k: 5; lambda: 0.08; candidate pool: 24
- Oracle view share: {'summary': 0.95, 'full_evidence_page': 0.05}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | title_recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.9883 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.0000 | 0.0072 | 0.0000 | 0.0000 | {'summary': 1.0} |
| fixed_title_page | 0.9843 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.0500 | 0.0112 | -0.5536 | -0.0040 | {'title_page': 1.0} |
| fixed_evidence_sentence | 0.9707 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.2200 | 0.0248 | -2.4359 | -0.0176 | {'evidence_sentence': 1.0} |
| fixed_full_evidence_page | 0.9408 | 0.9984 | 0.9339 | 0.4348 | 0.9267 | 0.7200 | 0.0547 | -6.5733 | -0.0475 | {'full_evidence_page': 1.0} |
| threshold_gate(summary,full_evidence_page) | 0.9854 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.0360 | 0.0101 | -0.3986 | -0.0029 | {'summary': 0.95, 'full_evidence_page': 0.05} |
| best_legal_router(rf_B0_claim) | 0.9869 | 0.9984 | 0.9339 | 0.4348 | 0.9267 | 0.1440 | 0.0086 | -0.1958 | -0.0014 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| best_legal_adaptive | 0.9869 | 0.9984 | 0.9339 | 0.4348 | 0.9267 | 0.1440 | 0.0086 | -0.1958 | -0.0014 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| oracle_route | 0.9955 | 0.9984 | 0.9339 | 0.4348 | 0.9267 | 0.0360 | 0.0000 | 1.0000 | 0.0072 | {'summary': 0.95, 'full_evidence_page': 0.05} |

## Randomization and shortcut controls

| control | utility | raw_ndcg | support_recall | support_f1 | title_recall | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claim_only | 0.9883 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.0000 | 0.0072 | 0.0000 |
| real_title_page | 0.9843 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.0500 | 0.0112 | -0.0040 |
| real_evidence_sentence | 0.9707 | 0.9883 | 0.9173 | 0.4223 | 0.9017 | 0.2200 | 0.0248 | -0.0176 |
| full_evidence_page | 0.9408 | 0.9984 | 0.9339 | 0.4348 | 0.9267 | 0.7200 | 0.0547 | -0.0475 |
| shuffled_evidence_sentence | 0.1781 | 0.1957 | 0.2177 | 0.1200 | 0.3367 | 0.2200 | 0.8174 | -0.8102 |
| wrong_evidence_sentence | 0.1464 | 0.1640 | 0.2154 | 0.0942 | 0.2850 | 0.2200 | 0.8491 | -0.8418 |
| same_count_random | 0.1884 | 0.2060 | 0.2204 | 0.1089 | 0.2617 | 0.2200 | 0.8071 | -0.7999 |
| title_frequency_matched_random | 0.2091 | 0.2131 | 0.1839 | 0.1521 | 0.1683 | 0.0500 | 0.7864 | -0.7792 |
| high_frequency_page | 0.2091 | 0.2131 | 0.1839 | 0.1521 | 0.1683 | 0.0500 | 0.7864 | -0.7792 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B0_claim | B0_claim | 0.9869 | 0.1440 | 0.0086 | -0.0014 | -0.1958 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| et_B0_claim | B0_claim | 0.9869 | 0.1440 | 0.0086 | -0.0014 | -0.1958 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| rf_B1_title_meta | B1_title_meta | 0.9869 | 0.1440 | 0.0086 | -0.0014 | -0.1958 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| et_B1_title_meta | B1_title_meta | 0.9869 | 0.1440 | 0.0086 | -0.0014 | -0.1958 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| rf_B1_sentence_handle | B1_sentence_handle | 0.9869 | 0.1440 | 0.0086 | -0.0014 | -0.1958 | {'summary': 0.8, 'full_evidence_page': 0.2} |
| et_B1_sentence_handle | B1_sentence_handle | 0.9840 | 0.1800 | 0.0115 | -0.0043 | -0.5944 | {'summary': 0.75, 'full_evidence_page': 0.25} |
| hgb_B0_claim | B0_claim | 0.9811 | 0.2160 | 0.0144 | -0.0072 | -0.9930 | {'summary': 0.7, 'full_evidence_page': 0.3} |
| hgb_B1_sentence_handle | B1_sentence_handle | 0.9667 | 0.3960 | 0.0288 | -0.0216 | -2.9860 | {'summary': 0.45, 'full_evidence_page': 0.55} |

## Label-split oracle headroom

| label | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| REFUTES | 4 | fixed_summary | 1.0000 | 1.0000 | 0.0000 |
| SUPPORTS | 16 | fixed_summary | 0.9853 | 0.9944 | 0.0090 |

## Repeated split stability

```json
{
  "ci95": [
    -0.030871449534814327,
    0.033378526412828805
  ],
  "learned_minus_best_fixed_mean": -0.00042068385614807413,
  "positive_share": 0.5,
  "repeats": 10,
  "selected_policies": {
    "best_legal_router(et_B0_claim)": 2,
    "best_legal_router(hgb_B1_sentence_handle)": 1,
    "best_legal_router(rf_B0_claim)": 3,
    "threshold_gate(summary,full_evidence_page)": 4
  }
}
```

## Protocol B legal/illegal trace

Legal route example:

```json
{
  "query_id": "fever_75397",
  "cell_id": "claim_evidence_pool",
  "tier": "B0",
  "cost_menu": "fever-claim-v1-op",
  "ranked_views": [
    "title_page",
    "evidence_sentence",
    "full_evidence_page"
  ],
  "route": "title_page"
}
```

Invalid route example:

```json
{
  "query_id": "fever_75397",
  "cell_id": "claim_evidence_pool",
  "tier": "B0",
  "cost_menu": "fever-claim-v1-op",
  "route": "evidence_sentence",
  "gold_evidence_handle": "Nikolaj_Coster-Waldau#7"
}
```

Reason: `gold_evidence_handle` is evaluator-only before evidence is bought.

Runtime: 18.95s
