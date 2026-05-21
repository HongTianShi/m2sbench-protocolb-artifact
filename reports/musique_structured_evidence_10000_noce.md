# MuSiQue Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence replication over MuSiQue paragraphs. A method sees the question, decomposition-question text, paragraph-title sketch, and coarse metadata, then may buy paragraph, multi-paragraph support-set, full-context, or optional CE evidence before evaluator-held support labels are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 10000 ({'train': 6000, 'dev': 2000, 'test': 2000})
- Upstream split: `train`; internal route split: 6000/2000/2000 train/dev/test; top-k: 4; lambda: 0.08
- Oracle view share: {'paragraph': 0.303, 'summary': 0.529, 'decomp_title': 0.074, 'support_set': 0.09, 'full_context': 0.004}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.5306 | 0.5306 | 0.5275 | 0.3773 | 0.5444 | 0 | 0.152 | -0.2693 | -0.0323 | {'summary': 1.0} |
| fixed_decomp_title | 0.5306 | 0.5338 | 0.5249 | 0.3757 | 0.5418 | 0.04 | 0.1521 | -0.2697 | -0.0323 | {'decomp_title': 1.0} |
| fixed_paragraph | 0.5629 | 0.5757 | 0.5718 | 0.4094 | 0.6007 | 0.16 | 0.1198 | 0 | 0 | {'paragraph': 1.0} |
| fixed_support_set | 0.5233 | 0.5473 | 0.5405 | 0.387 | 0.5697 | 0.3 | 0.1594 | -0.3307 | -0.0396 | {'support_set': 1.0} |
| fixed_full_context | 0.4936 | 0.5376 | 0.5326 | 0.3809 | 0.5615 | 0.55 | 0.189 | -0.5784 | -0.0693 | {'full_context': 1.0} |
| threshold_gate(summary,paragraph) | 0.5611 | 0.5732 | 0.5694 | 0.4075 | 0.5984 | 0.1515 | 0.1216 | -0.0151 | -0.0018 | {'summary': 0.053, 'paragraph': 0.947} |
| best_legal_router(et_B1_context_sketch) | 0.5819 | 0.5916 | 0.5827 | 0.4171 | 0.6074 | 0.1209 | 0.1007 | 0.1588 | 0.019 | {'summary': 0.214, 'decomp_title': 0.203, 'paragraph': 0.4635, 'support_set': 0.1085, 'full_context': 0.011} |
| best_legal_adaptive | 0.5819 | 0.5916 | 0.5827 | 0.4171 | 0.6074 | 0.1209 | 0.1007 | 0.1588 | 0.019 | {'summary': 0.214, 'decomp_title': 0.203, 'paragraph': 0.4635, 'support_set': 0.1085, 'full_context': 0.011} |
| oracle_route | 0.6826 | 0.6891 | 0.6757 | 0.4846 | 0.6919 | 0.0806 | 0 | 1 | 0.1198 | {'summary': 0.529, 'decomp_title': 0.074, 'paragraph': 0.303, 'support_set': 0.09, 'full_context': 0.004} |

## Randomization and decomposition controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_title_recall | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.5306 | 0.5306 | 0.5275 | 0.3773 | 0.5444 | 0 | 0.152 | -0.0323 |
| decomposition_title | 0.5306 | 0.5338 | 0.5249 | 0.3757 | 0.5418 | 0.04 | 0.1521 | -0.0323 |
| real_paragraph | 0.5629 | 0.5757 | 0.5718 | 0.4094 | 0.6007 | 0.16 | 0.1198 | 0 |
| real_support_set | 0.5233 | 0.5473 | 0.5405 | 0.387 | 0.5697 | 0.3 | 0.1594 | -0.0396 |
| shuffled_paragraph | 0.1466 | 0.1594 | 0.1946 | 0.1428 | 0.2288 | 0.16 | 0.5361 | -0.4163 |
| same_title_shuffled_mapping | 0.1564 | 0.1692 | 0.2058 | 0.1501 | 0.2445 | 0.16 | 0.5262 | -0.4065 |
| random_paragraph | 0.1511 | 0.1639 | 0.1995 | 0.145 | 0.2392 | 0.16 | 0.5315 | -0.4118 |
| same_count_random | 0.1526 | 0.1654 | 0.2017 | 0.1466 | 0.2402 | 0.16 | 0.5301 | -0.4103 |
| decomposition_shuffled | 0.4669 | 0.4909 | 0.5139 | 0.3683 | 0.5447 | 0.3 | 0.2157 | -0.096 |
| full_context | 0.4936 | 0.5376 | 0.5326 | 0.3809 | 0.5615 | 0.55 | 0.189 | -0.0693 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B1_decomp_meta | B1_decomp_meta | 0.5851 | 0.1183 | 0.0976 | 0.0222 | 0.1851 | {'summary': 0.2245, 'decomp_title': 0.2215, 'paragraph': 0.444, 'support_set': 0.0885, 'full_context': 0.0215} |
| rf_B1_context_sketch | B1_context_sketch | 0.5843 | 0.1165 | 0.0983 | 0.0214 | 0.1789 | {'summary': 0.2215, 'decomp_title': 0.229, 'paragraph': 0.445, 'support_set': 0.0855, 'full_context': 0.019} |
| et_B1_decomp_meta | B1_decomp_meta | 0.5837 | 0.1209 | 0.099 | 0.0208 | 0.1737 | {'summary': 0.205, 'decomp_title': 0.21, 'paragraph': 0.4725, 'support_set': 0.1, 'full_context': 0.0125} |
| et_B1_context_sketch | B1_context_sketch | 0.5819 | 0.1209 | 0.1007 | 0.019 | 0.1588 | {'summary': 0.214, 'decomp_title': 0.203, 'paragraph': 0.4635, 'support_set': 0.1085, 'full_context': 0.011} |
| et_B1_paragraph_meta | B1_paragraph_meta | 0.5789 | 0.1235 | 0.1038 | 0.016 | 0.1334 | {'summary': 0.2115, 'decomp_title': 0.1815, 'paragraph': 0.4875, 'support_set': 0.11, 'full_context': 0.0095} |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.5782 | 0.1207 | 0.1044 | 0.0153 | 0.1281 | {'summary': 0.2265, 'decomp_title': 0.199, 'paragraph': 0.468, 'support_set': 0.083, 'full_context': 0.0235} |
| hgb_B1_decomp_meta | B1_decomp_meta | 0.5779 | 0.1017 | 0.1047 | 0.015 | 0.1255 | {'summary': 0.398, 'paragraph': 0.58, 'support_set': 0.013, 'full_context': 0.009} |
| hgb_B1_context_sketch | B1_context_sketch | 0.576 | 0.1186 | 0.1066 | 0.0131 | 0.1096 | {'summary': 0.3015, 'paragraph': 0.6745, 'support_set': 0.01, 'full_context': 0.014} |

## Hop-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| 2hop | 1443 | fixed_paragraph | 0.5931 | 0.715 | 0.1219 |
| 3hop1 | 367 | fixed_paragraph | 0.4716 | 0.5835 | 0.1119 |
| 3hop2 | 72 | fixed_paragraph | 0.5428 | 0.6847 | 0.1418 |
| 4hop1 | 57 | fixed_paragraph | 0.4892 | 0.5981 | 0.1089 |
| 4hop2 | 10 | fixed_summary | 0.3973 | 0.4717 | 0.0744 |
| 4hop3 | 51 | fixed_support_set | 0.5192 | 0.6134 | 0.0942 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.015294552519766524,
  "ci95": [
    0.01162034763819452,
    0.018564970196528374
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(et_B1_context_sketch)": 6,
    "best_legal_router(et_B1_decomp_meta)": 4
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

