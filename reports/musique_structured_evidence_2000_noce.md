# MuSiQue Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence replication over MuSiQue paragraphs. A method sees the question, decomposition-question text, paragraph-title sketch, and coarse metadata, then may buy paragraph, multi-paragraph support-set, full-context, or optional CE evidence before evaluator-held support labels are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.55, 'paragraph': 0.2575, 'support_set': 0.1, 'decomp_title': 0.0875, 'full_context': 0.005}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.5484 | 0.5484 | 0.5346 | 0.3806 | 0.5521 | 0 | 0.1286 | -0.0265 | -0.0033 | {'summary': 1.0} |
| fixed_decomp_title | 0.5517 | 0.5549 | 0.5356 | 0.3814 | 0.554 | 0.04 | 0.1252 | 0 | 0 | {'decomp_title': 1.0} |
| fixed_paragraph | 0.5484 | 0.5612 | 0.5579 | 0.3986 | 0.591 | 0.16 | 0.1285 | -0.0264 | -0.0033 | {'paragraph': 1.0} |
| fixed_support_set | 0.5235 | 0.5475 | 0.5392 | 0.3848 | 0.574 | 0.3 | 0.1534 | -0.2247 | -0.0281 | {'support_set': 1.0} |
| fixed_full_context | 0.4944 | 0.5384 | 0.531 | 0.3791 | 0.5627 | 0.55 | 0.1825 | -0.4574 | -0.0573 | {'full_context': 1.0} |
| threshold_gate(summary,paragraph) | 0.5446 | 0.5567 | 0.5515 | 0.3934 | 0.5831 | 0.1512 | 0.1323 | -0.0565 | -0.0071 | {'summary': 0.055, 'paragraph': 0.945} |
| best_legal_router(hgb_B0_title_decomp) | 0.572 | 0.5807 | 0.5717 | 0.4086 | 0.6002 | 0.1087 | 0.1049 | 0.1621 | 0.0203 | {'summary': 0.4175, 'paragraph': 0.525, 'support_set': 0.0275, 'full_context': 0.03} |
| best_legal_adaptive | 0.572 | 0.5807 | 0.5717 | 0.4086 | 0.6002 | 0.1087 | 0.1049 | 0.1621 | 0.0203 | {'summary': 0.4175, 'paragraph': 0.525, 'support_set': 0.0275, 'full_context': 0.03} |
| oracle_route | 0.6769 | 0.6831 | 0.6698 | 0.4793 | 0.6846 | 0.0775 | 0 | 1 | 0.1252 | {'summary': 0.55, 'decomp_title': 0.0875, 'paragraph': 0.2575, 'support_set': 0.1, 'full_context': 0.005} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.5484 | 0.5484 | 0.5346 | 0.3806 | 0.5521 | 0 | 0.1286 | -0.0033 |
| decomposition_title | 0.5517 | 0.5549 | 0.5356 | 0.3814 | 0.554 | 0.04 | 0.1252 | 0 |
| real_paragraph | 0.5484 | 0.5612 | 0.5579 | 0.3986 | 0.591 | 0.16 | 0.1285 | -0.0033 |
| real_support_set | 0.5235 | 0.5475 | 0.5392 | 0.3848 | 0.574 | 0.3 | 0.1534 | -0.0281 |
| shuffled_paragraph | 0.159 | 0.1718 | 0.2152 | 0.1556 | 0.249 | 0.16 | 0.518 | -0.3927 |
| same_title_shuffled_mapping | 0.1646 | 0.1774 | 0.2102 | 0.1524 | 0.2523 | 0.16 | 0.5123 | -0.3871 |
| random_paragraph | 0.1442 | 0.157 | 0.2015 | 0.1447 | 0.2469 | 0.16 | 0.5327 | -0.4075 |
| same_count_random | 0.1396 | 0.1524 | 0.1935 | 0.1408 | 0.2325 | 0.16 | 0.5373 | -0.412 |
| decomposition_shuffled | 0.4376 | 0.4616 | 0.4906 | 0.3509 | 0.5246 | 0.3 | 0.2394 | -0.1141 |
| full_context | 0.4944 | 0.5384 | 0.531 | 0.3791 | 0.5627 | 0.55 | 0.1825 | -0.0573 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.5838 | 0.0875 | 0.0931 | 0.0322 | 0.2567 | {'summary': 0.345, 'decomp_title': 0.2225, 'paragraph': 0.365, 'support_set': 0.0675} |
| hgb_B1_paragraph_meta | B1_paragraph_meta | 0.574 | 0.1007 | 0.1029 | 0.0223 | 0.1781 | {'summary': 0.4725, 'paragraph': 0.4425, 'support_set': 0.0675, 'full_context': 0.0175} |
| et_B1_paragraph_meta | B1_paragraph_meta | 0.5734 | 0.1104 | 0.1035 | 0.0217 | 0.1733 | {'summary': 0.2925, 'decomp_title': 0.2025, 'paragraph': 0.3825, 'support_set': 0.105, 'full_context': 0.0175} |
| rf_B1_context_sketch | B1_context_sketch | 0.5732 | 0.0919 | 0.1037 | 0.0215 | 0.172 | {'summary': 0.315, 'decomp_title': 0.225, 'paragraph': 0.4025, 'support_set': 0.0525, 'full_context': 0.005} |
| hgb_B0_title_decomp | B0_title_decomp | 0.572 | 0.1087 | 0.1049 | 0.0203 | 0.1621 | {'summary': 0.4175, 'paragraph': 0.525, 'support_set': 0.0275, 'full_context': 0.03} |
| hgb_B1_context_sketch | B1_context_sketch | 0.5713 | 0.0963 | 0.1056 | 0.0196 | 0.1565 | {'summary': 0.505, 'paragraph': 0.4175, 'support_set': 0.0525, 'full_context': 0.025} |
| rf_B1_decomp_meta | B1_decomp_meta | 0.5712 | 0.0903 | 0.1057 | 0.0195 | 0.156 | {'summary': 0.31, 'decomp_title': 0.2225, 'paragraph': 0.42, 'support_set': 0.0475} |
| rf_B0_title_decomp | B0_title_decomp | 0.5702 | 0.1 | 0.1067 | 0.0185 | 0.1481 | {'summary': 0.27, 'decomp_title': 0.2625, 'paragraph': 0.38, 'support_set': 0.0775, 'full_context': 0.01} |

## Hop-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| 2hop | 292 | fixed_summary | 0.5903 | 0.7112 | 0.1209 |
| 3hop1 | 69 | fixed_paragraph | 0.4648 | 0.5863 | 0.1215 |
| 3hop2 | 12 | fixed_paragraph | 0.4781 | 0.6283 | 0.1502 |
| 4hop1 | 19 | fixed_paragraph | 0.4637 | 0.5523 | 0.0886 |
| 4hop2 | 1 | fixed_summary | 0.3904 | 0.3904 | 0 |
| 4hop3 | 7 | fixed_decomp_title | 0.5022 | 0.6034 | 0.1012 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.004693365738357158,
  "ci95": [
    -0.00746045776853746,
    0.018393949914521673
  ],
  "positive_share": 0.6,
  "selected_policies": {
    "best_legal_router(rf_B1_context_sketch)": 2,
    "best_legal_router(et_B1_context_sketch)": 1,
    "best_legal_router(rf_B1_decomp_meta)": 2,
    "best_legal_router(rf_B1_paragraph_meta)": 1,
    "best_legal_router(et_B1_decomp_meta)": 1,
    "best_legal_router(hgb_B0_title_decomp)": 1,
    "best_legal_router(hgb_B1_paragraph_meta)": 1,
    "threshold_gate(summary,paragraph)": 1
  },
  "repeats": 10
}
```

## Protocol B legal/illegal trace

```json
{
  "visible_fields": {
    "query_id": "2hop__269805_135710",
    "question": "What is the country where Nissedal is located named after?",
    "decomposition_questions": [
      "Nissedal >> country",
      "The #1 was named for whom?"
    ],
    "declared_views": [
      "summary",
      "decomp_title",
      "paragraph",
      "support_set",
      "full_context"
    ],
    "visible_titles": [
      "Where Dead Voices Gather",
      "The Hireling Shepherd",
      "The Book of Proper Names",
      "Milton F. Pavlic"
    ],
    "released_state": "question, decomposition-question text, paragraph-title sketch, and coarse paragraph metadata; support labels, answers, and unpaid CE/full scores are hidden"
  },
  "legal_action": {
    "submitted_jsonl": {
      "query_id": "2hop__269805_135710",
      "cell_id": "musique_2hop_269805_135710",
      "tier": "B1_structured",
      "cost_menu": "musique-structured-v0-op",
      "ranked_views": [
        "support_set",
        "decomp_title",
        "paragraph",
        "full_context"
      ],
      "route": "support_set"
    },
    "charged_cost": 0.3,
    "scored_utility": 0.6694264036172708
  },
  "hidden_evaluator_fields": {
    "support_idx": [
      "10",
      "6"
    ],
    "support_titles": [
      "norway",
      "tveitsund"
    ],
    "answer": "north",
    "decomposition_answers": [
      "Norway",
      "north"
    ],
    "decomposition_support_idx": [
      10,
      6
    ]
  },
  "illegal_variant": {
    "submitted_jsonl": {
      "query_id": "2hop__269805_135710",
      "cell_id": "musique_2hop_269805_135710",
      "tier": "B1_structured",
      "cost_menu": "musique-structured-v0-op",
      "route": "support_set",
      "is_supporting": [
        "10",
        "6"
      ],
      "answer": "north",
      "decomposition_answers": [
        "Norway",
        "north"
      ]
    },
    "rejection_reason": "support labels, answers, and decomposition answers are evaluator-only fields."
  },
  "scope_note": "This is structured evidence acquisition over MuSiQue paragraphs, not answer generation."
}
```
