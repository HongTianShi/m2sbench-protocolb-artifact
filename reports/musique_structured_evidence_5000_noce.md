# MuSiQue Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence replication over MuSiQue paragraphs. A method sees the question, decomposition-question text, paragraph-title sketch, and coarse metadata, then may buy paragraph, multi-paragraph support-set, full-context, or optional CE evidence before evaluator-held support labels are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 5000 ({'train': 3000, 'dev': 1000, 'test': 1000})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.515, 'paragraph': 0.315, 'support_set': 0.096, 'decomp_title': 0.069, 'full_context': 0.005}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.5314 | 0.5314 | 0.5268 | 0.3799 | 0.5425 | 0 | 0.1502 | -0.2469 | -0.0297 | {'summary': 1.0} |
| fixed_decomp_title | 0.5284 | 0.5316 | 0.5245 | 0.3781 | 0.5403 | 0.04 | 0.1532 | -0.2724 | -0.0328 | {'decomp_title': 1.0} |
| fixed_paragraph | 0.5612 | 0.574 | 0.5654 | 0.4063 | 0.5949 | 0.16 | 0.1204 | 0 | 0 | {'paragraph': 1.0} |
| fixed_support_set | 0.5237 | 0.5477 | 0.5414 | 0.3897 | 0.5703 | 0.3 | 0.1578 | -0.3107 | -0.0374 | {'support_set': 1.0} |
| fixed_full_context | 0.4947 | 0.5387 | 0.5313 | 0.382 | 0.5593 | 0.55 | 0.1869 | -0.5517 | -0.0664 | {'full_context': 1.0} |
| threshold_gate(summary,paragraph) | 0.5588 | 0.5709 | 0.5629 | 0.4046 | 0.5919 | 0.1512 | 0.1228 | -0.0194 | -0.0023 | {'summary': 0.055, 'paragraph': 0.945} |
| best_legal_router(et_B1_paragraph_meta) | 0.5678 | 0.5769 | 0.5623 | 0.4046 | 0.5843 | 0.1138 | 0.1138 | 0.0552 | 0.0067 | {'summary': 0.269, 'decomp_title': 0.196, 'paragraph': 0.416, 'support_set': 0.104, 'full_context': 0.015} |
| best_legal_adaptive | 0.5678 | 0.5769 | 0.5623 | 0.4046 | 0.5843 | 0.1138 | 0.1138 | 0.0552 | 0.0067 | {'summary': 0.269, 'decomp_title': 0.196, 'paragraph': 0.416, 'support_set': 0.104, 'full_context': 0.015} |
| oracle_route | 0.6816 | 0.6884 | 0.6797 | 0.4897 | 0.6928 | 0.0847 | 0 | 1 | 0.1204 | {'summary': 0.515, 'decomp_title': 0.069, 'paragraph': 0.315, 'support_set': 0.096, 'full_context': 0.005} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.5314 | 0.5314 | 0.5268 | 0.3799 | 0.5425 | 0 | 0.1502 | -0.0297 |
| decomposition_title | 0.5284 | 0.5316 | 0.5245 | 0.3781 | 0.5403 | 0.04 | 0.1532 | -0.0328 |
| real_paragraph | 0.5612 | 0.574 | 0.5654 | 0.4063 | 0.5949 | 0.16 | 0.1204 | 0 |
| real_support_set | 0.5237 | 0.5477 | 0.5414 | 0.3897 | 0.5703 | 0.3 | 0.1578 | -0.0374 |
| shuffled_paragraph | 0.1472 | 0.16 | 0.1923 | 0.1421 | 0.223 | 0.16 | 0.5344 | -0.4139 |
| same_title_shuffled_mapping | 0.1656 | 0.1784 | 0.211 | 0.1554 | 0.2482 | 0.16 | 0.516 | -0.3956 |
| random_paragraph | 0.1563 | 0.1691 | 0.2061 | 0.1508 | 0.2462 | 0.16 | 0.5253 | -0.4049 |
| same_count_random | 0.1512 | 0.164 | 0.203 | 0.1492 | 0.2397 | 0.16 | 0.5304 | -0.4099 |
| decomposition_shuffled | 0.4666 | 0.4906 | 0.5118 | 0.3671 | 0.5413 | 0.3 | 0.215 | -0.0946 |
| full_context | 0.4947 | 0.5387 | 0.5313 | 0.382 | 0.5593 | 0.55 | 0.1869 | -0.0664 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| et_B1_context_sketch | B1_context_sketch | 0.5805 | 0.1117 | 0.1011 | 0.0194 | 0.1609 | {'summary': 0.26, 'decomp_title': 0.219, 'paragraph': 0.41, 'support_set': 0.095, 'full_context': 0.016} |
| hgb_B1_decomp_meta | B1_decomp_meta | 0.574 | 0.0975 | 0.1076 | 0.0129 | 0.1067 | {'summary': 0.402, 'decomp_title': 0.121, 'paragraph': 0.387, 'support_set': 0.075, 'full_context': 0.015} |
| et_B0_title_decomp | B0_title_decomp | 0.5727 | 0.1174 | 0.1089 | 0.0115 | 0.0958 | {'summary': 0.265, 'decomp_title': 0.188, 'paragraph': 0.416, 'support_set': 0.115, 'full_context': 0.016} |
| et_B1_decomp_meta | B1_decomp_meta | 0.5707 | 0.1123 | 0.1109 | 0.0096 | 0.0794 | {'summary': 0.265, 'decomp_title': 0.213, 'paragraph': 0.408, 'support_set': 0.097, 'full_context': 0.017} |
| rf_B1_context_sketch | B1_context_sketch | 0.5697 | 0.1068 | 0.1119 | 0.0085 | 0.071 | {'summary': 0.301, 'decomp_title': 0.204, 'paragraph': 0.383, 'support_set': 0.097, 'full_context': 0.015} |
| et_B1_paragraph_meta | B1_paragraph_meta | 0.5678 | 0.1138 | 0.1138 | 0.0067 | 0.0552 | {'summary': 0.269, 'decomp_title': 0.196, 'paragraph': 0.416, 'support_set': 0.104, 'full_context': 0.015} |
| hgb_B0_title_decomp | B0_title_decomp | 0.5673 | 0.0912 | 0.1142 | 0.0062 | 0.0514 | {'summary': 0.312, 'decomp_title': 0.222, 'paragraph': 0.43, 'support_set': 0.025, 'full_context': 0.011} |
| hgb_B1_context_sketch | B1_context_sketch | 0.5666 | 0.087 | 0.1149 | 0.0055 | 0.0456 | {'summary': 0.527, 'paragraph': 0.417, 'support_set': 0.042, 'full_context': 0.014} |

## Hop-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| 2hop | 697 | fixed_paragraph | 0.5988 | 0.7157 | 0.1169 |
| 3hop1 | 209 | fixed_paragraph | 0.4677 | 0.5979 | 0.1302 |
| 3hop2 | 36 | fixed_paragraph | 0.5391 | 0.7007 | 0.1616 |
| 4hop1 | 30 | fixed_paragraph | 0.445 | 0.5411 | 0.0961 |
| 4hop2 | 7 | fixed_summary | 0.3861 | 0.4418 | 0.0557 |
| 4hop3 | 21 | fixed_paragraph | 0.5295 | 0.6313 | 0.1018 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.0094622635883714,
  "ci95": [
    0.0024552818941627834,
    0.018242103187364477
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(et_B1_context_sketch)": 3,
    "best_legal_router(hgb_B1_context_sketch)": 1,
    "best_legal_router(rf_B1_decomp_meta)": 1,
    "best_legal_router(hgb_B1_decomp_meta)": 2,
    "best_legal_router(et_B1_decomp_meta)": 2,
    "best_legal_router(et_B1_paragraph_meta)": 1
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
