# MuSiQue Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence replication over MuSiQue paragraphs. A method sees the question, decomposition-question text, paragraph-title sketch, and coarse metadata, then may buy paragraph, multi-paragraph support-set, full-context, or optional CE evidence before evaluator-held support labels are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 100 ({'train': 60, 'dev': 20, 'test': 20})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.6, 'paragraph': 0.35, 'decomp_title': 0.05}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.623 | 0.623 | 0.6583 | 0.4452 | 0.6583 | 0 | 0.103 | -0.0855 | -0.0081 | {'summary': 1.0} |
| fixed_decomp_title | 0.6311 | 0.6343 | 0.6583 | 0.4452 | 0.6583 | 0.04 | 0.0949 | 0 | 0 | {'decomp_title': 1.0} |
| fixed_paragraph | 0.5525 | 0.5653 | 0.6292 | 0.4268 | 0.6542 | 0.16 | 0.1735 | -0.8282 | -0.0786 | {'paragraph': 1.0} |
| fixed_support_set | 0.4423 | 0.4663 | 0.5042 | 0.3435 | 0.5042 | 0.3 | 0.2837 | -1.9898 | -0.1888 | {'support_set': 1.0} |
| fixed_full_context | 0.4091 | 0.4531 | 0.4792 | 0.3268 | 0.4792 | 0.55 | 0.3169 | -2.3398 | -0.222 | {'full_context': 1.0} |
| threshold_gate(summary,support_set) | 0.518 | 0.5312 | 0.5833 | 0.3952 | 0.5833 | 0.165 | 0.208 | -1.1919 | -0.1131 | {'summary': 0.45, 'support_set': 0.55} |
| best_legal_router(et_B1_paragraph_meta) | 0.5879 | 0.5924 | 0.6417 | 0.431 | 0.6417 | 0.056 | 0.1381 | -0.4555 | -0.0432 | {'summary': 0.35, 'decomp_title': 0.4, 'paragraph': 0.25} |
| best_legal_adaptive | 0.5879 | 0.5924 | 0.6417 | 0.431 | 0.6417 | 0.056 | 0.1381 | -0.4555 | -0.0432 | {'summary': 0.35, 'decomp_title': 0.4, 'paragraph': 0.25} |
| oracle_route | 0.726 | 0.7306 | 0.7708 | 0.5244 | 0.7708 | 0.058 | 0 | 1 | 0.0949 | {'summary': 0.6, 'decomp_title': 0.05, 'paragraph': 0.35} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.623 | 0.623 | 0.6583 | 0.4452 | 0.6583 | 0 | 0.103 | -0.0081 |
| decomposition_title | 0.6311 | 0.6343 | 0.6583 | 0.4452 | 0.6583 | 0.04 | 0.0949 | 0 |
| real_paragraph | 0.5525 | 0.5653 | 0.6292 | 0.4268 | 0.6542 | 0.16 | 0.1735 | -0.0786 |
| real_support_set | 0.4423 | 0.4663 | 0.5042 | 0.3435 | 0.5042 | 0.3 | 0.2837 | -0.1888 |
| shuffled_paragraph | 0.1877 | 0.2005 | 0.2458 | 0.1744 | 0.2958 | 0.16 | 0.5383 | -0.4434 |
| same_title_shuffled_mapping | 0.1551 | 0.1679 | 0.1917 | 0.1393 | 0.1917 | 0.16 | 0.5709 | -0.476 |
| random_paragraph | 0.1955 | 0.2083 | 0.2583 | 0.1869 | 0.2583 | 0.16 | 0.5305 | -0.4356 |
| same_count_random | 0.1941 | 0.2069 | 0.2667 | 0.181 | 0.2667 | 0.16 | 0.5319 | -0.437 |
| decomposition_shuffled | 0.4684 | 0.4924 | 0.5667 | 0.381 | 0.5667 | 0.3 | 0.2576 | -0.1628 |
| full_context | 0.4091 | 0.4531 | 0.4792 | 0.3268 | 0.4792 | 0.55 | 0.3169 | -0.222 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| et_B1_context_sketch | B1_context_sketch | 0.648 | 0.048 | 0.078 | 0.0168 | 0.1775 | {'summary': 0.4, 'decomp_title': 0.4, 'paragraph': 0.2} |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.6425 | 0.052 | 0.0835 | 0.0113 | 0.1196 | {'summary': 0.45, 'decomp_title': 0.3, 'paragraph': 0.25} |
| rf_B0_title_decomp | B0_title_decomp | 0.6296 | 0.04 | 0.0964 | -0.0016 | -0.0164 | {'summary': 0.6, 'decomp_title': 0.2, 'paragraph': 0.2} |
| hgb_B1_context_sketch | B1_context_sketch | 0.6291 | 0.058 | 0.0969 | -0.002 | -0.0211 | {'summary': 0.45, 'decomp_title': 0.25, 'paragraph': 0.3} |
| ridge_B0_title_decomp | B0_title_decomp | 0.623 | 0 | 0.103 | -0.0081 | -0.0855 | {'summary': 1.0} |
| ridge_B1_paragraph_meta | B1_paragraph_meta | 0.623 | 0 | 0.103 | -0.0081 | -0.0855 | {'summary': 1.0} |
| ridge_B1_decomp_meta | B1_decomp_meta | 0.623 | 0 | 0.103 | -0.0081 | -0.0855 | {'summary': 1.0} |
| ridge_B1_context_sketch | B1_context_sketch | 0.623 | 0 | 0.103 | -0.0081 | -0.0855 | {'summary': 1.0} |

## Hop-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| 2hop | 18 | fixed_decomp_title | 0.6773 | 0.7723 | 0.0949 |
| 3hop1 | 1 | fixed_summary | 0.4367 | 0.4367 | 0 |
| 4hop2 | 1 | fixed_paragraph | 0.1824 | 0.1824 | 0 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": -0.01789074981995183,
  "ci95": [
    -0.03857478986652286,
    0.0076907386763158826
  ],
  "positive_share": 0.3333333333333333,
  "selected_policies": {
    "threshold_gate(summary,support_set)": 1,
    "threshold_gate(summary,paragraph)": 1,
    "best_legal_router(hgb_B0_title_decomp)": 1
  },
  "repeats": 3
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
