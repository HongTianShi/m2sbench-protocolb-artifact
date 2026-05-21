# HotpotQA Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence slice over HotpotQA distractor contexts. A method sees the question and title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'sentence': 0.1125, 'summary': 0.36, 'paragraph': 0.3325, 'ce': 0.1075, 'title_shortlist': 0.0325, 'full_context': 0.055}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.665 | 0.665 | 0.6775 | 0.4525 | 0.2905 | 0 | 0.2478 | -1.1134 | -0.1305 | {'summary': 1.0} |
| fixed_title_shortlist | 0.6734 | 0.6766 | 0.6787 | 0.4533 | 0.2913 | 0.04 | 0.2394 | -1.0417 | -0.1221 | {'title_shortlist': 1.0} |
| fixed_paragraph | 0.7956 | 0.8084 | 0.8337 | 0.5567 | 0.3699 | 0.16 | 0.1173 | 0 | 0 | {'paragraph': 1.0} |
| fixed_sentence | 0.771 | 0.7934 | 0.8137 | 0.5433 | 0.4481 | 0.28 | 0.1418 | -0.2093 | -0.0245 | {'sentence': 1.0} |
| fixed_full_context | 0.7373 | 0.7813 | 0.8025 | 0.5358 | 0.3495 | 0.55 | 0.1756 | -0.4972 | -0.0583 | {'full_context': 1.0} |
| fixed_ce | 0.7408 | 0.8168 | 0.8187 | 0.5467 | 0.3555 | 0.95 | 0.172 | -0.467 | -0.0548 | {'ce': 1.0} |
| threshold_gate(summary,sentence) | 0.7584 | 0.7796 | 0.7987 | 0.5333 | 0.4319 | 0.2653 | 0.1544 | -0.3172 | -0.0372 | {'summary': 0.0525, 'sentence': 0.9475} |
| best_legal_router(rf_B1_sentence_meta) | 0.7862 | 0.8088 | 0.815 | 0.5442 | 0.3821 | 0.2829 | 0.1266 | -0.0799 | -0.0094 | {'summary': 0.1675, 'title_shortlist': 0.08, 'paragraph': 0.3475, 'sentence': 0.1875, 'full_context': 0.0875, 'ce': 0.13} |
| best_legal_adaptive | 0.7862 | 0.8088 | 0.815 | 0.5442 | 0.3821 | 0.2829 | 0.1266 | -0.0799 | -0.0094 | {'summary': 0.1675, 'title_shortlist': 0.08, 'paragraph': 0.3475, 'sentence': 0.1875, 'full_context': 0.0875, 'ce': 0.13} |
| oracle_route | 0.9128 | 0.9303 | 0.94 | 0.6275 | 0.4285 | 0.2184 | 0 | 1 | 0.1173 | {'summary': 0.36, 'title_shortlist': 0.0325, 'paragraph': 0.3325, 'sentence': 0.1125, 'full_context': 0.055, 'ce': 0.1075} |

## Randomization controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.665 | 0.665 | 0.6775 | 0.4525 | 0.2905 | 0 | 0.2478 | -0.1305 |
| title_shortlist | 0.6734 | 0.6766 | 0.6787 | 0.4533 | 0.2913 | 0.04 | 0.2394 | -0.1221 |
| real_paragraph | 0.7956 | 0.8084 | 0.8337 | 0.5567 | 0.3699 | 0.16 | 0.1173 | 0 |
| real_sentence | 0.771 | 0.7934 | 0.8137 | 0.5433 | 0.4481 | 0.28 | 0.1418 | -0.0245 |
| shuffled_title | 0.3005 | 0.3005 | 0.3837 | 0.2567 | 0.1755 | 0 | 0.6123 | -0.4951 |
| random_paragraph | 0.2994 | 0.3122 | 0.4012 | 0.2683 | 0.1726 | 0.16 | 0.6134 | -0.4962 |
| random_sentence | 0.2719 | 0.2943 | 0.3837 | 0.2567 | 0.1725 | 0.28 | 0.6409 | -0.5236 |
| full_context | 0.7373 | 0.7813 | 0.8025 | 0.5358 | 0.3495 | 0.55 | 0.1756 | -0.0583 |
| ce | 0.7408 | 0.8168 | 0.8187 | 0.5467 | 0.3555 | 0.95 | 0.172 | -0.0548 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B1_context_sketch | B1_context_sketch | 0.8071 | 0.3259 | 0.1057 | 0.0116 | 0.0985 | {'summary': 0.185, 'title_shortlist': 0.065, 'paragraph': 0.2325, 'sentence': 0.265, 'full_context': 0.07, 'ce': 0.1825} |
| et_B1_context_sketch | B1_context_sketch | 0.806 | 0.3002 | 0.1068 | 0.0104 | 0.0888 | {'summary': 0.155, 'title_shortlist': 0.0925, 'paragraph': 0.235, 'sentence': 0.3175, 'full_context': 0.05, 'ce': 0.15} |
| hgb_B1_context_sketch | B1_context_sketch | 0.8045 | 0.3031 | 0.1083 | 0.0089 | 0.076 | {'summary': 0.195, 'title_shortlist': 0.035, 'paragraph': 0.1025, 'sentence': 0.4475, 'full_context': 0.1225, 'ce': 0.0975} |
| hgb_B1_sentence_meta | B1_sentence_meta | 0.7993 | 0.2575 | 0.1136 | 0.0037 | 0.0316 | {'summary': 0.12, 'title_shortlist': 0.105, 'paragraph': 0.315, 'sentence': 0.3225, 'full_context': 0.045, 'ce': 0.0925} |
| hgb_B0_title | B0_title | 0.7977 | 0.245 | 0.1151 | 0.0022 | 0.0184 | {'summary': 0.2375, 'title_shortlist': 0.02, 'paragraph': 0.3375, 'sentence': 0.265, 'full_context': 0.0425, 'ce': 0.0975} |
| hgb_B1_paragraph_meta | B1_paragraph_meta | 0.7964 | 0.2397 | 0.1164 | 0.0008 | 0.0072 | {'summary': 0.2225, 'title_shortlist': 0.0275, 'paragraph': 0.355, 'sentence': 0.25, 'full_context': 0.065, 'ce': 0.08} |
| rf_B0_title | B0_title | 0.7926 | 0.3165 | 0.1202 | -0.003 | -0.0252 | {'summary': 0.19, 'title_shortlist': 0.0875, 'paragraph': 0.2575, 'sentence': 0.1775, 'full_context': 0.1275, 'ce': 0.16} |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.7917 | 0.2889 | 0.1212 | -0.0039 | -0.0332 | {'summary': 0.195, 'title_shortlist': 0.065, 'paragraph': 0.3025, 'sentence': 0.21, 'full_context': 0.0925, 'ce': 0.135} |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge | 332 | fixed_paragraph | 0.7791 | 0.8961 | 0.117 |
| comparison | 68 | fixed_title_shortlist | 0.9846 | 0.9942 | 0.0096 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.0180464053953044,
  "ci95": [
    0.009841039972444874,
    0.024912053298094913
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(et_B1_context_sketch)": 4,
    "best_legal_router(hgb_B1_paragraph_meta)": 1,
    "best_legal_router(hgb_B1_context_sketch)": 3,
    "best_legal_router(rf_B1_context_sketch)": 1,
    "best_legal_router(hgb_B1_sentence_meta)": 1
  },
  "repeats": 10
}
```

## Protocol B legal/illegal trace

```json
{
  "visible_fields": {
    "query_id": "5a8d7341554299441c6b9fe5",
    "question": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
    "declared_views": [
      "summary",
      "title_shortlist",
      "paragraph",
      "sentence",
      "full_context",
      "ce"
    ],
    "visible_titles": [
      "Allie Goertz",
      "List of The Simpsons guest stars",
      "List of The Simpsons video games",
      "The Simpsons: An Uncensored, Unauthorized History"
    ],
    "released_state": "question plus context-title sketch; support facts, answers, and unpaid sentence/CE scores are hidden"
  },
  "legal_action": {
    "submitted_jsonl": {
      "query_id": "5a8d7341554299441c6b9fe5",
      "cell_id": "hotpot_5a8d7341554299441c6b9fe5",
      "tier": "B1_structured",
      "cost_menu": "hotpot-structured-v0-op",
      "ranked_views": [
        "paragraph",
        "title_shortlist",
        "sentence",
        "full_context"
      ],
      "route": "paragraph"
    },
    "charged_cost": 0.16,
    "scored_utility": 0.9872
  },
  "hidden_evaluator_fields": {
    "support_titles": [
      "allie goertz",
      "milhouse van houten"
    ],
    "supporting_facts": [
      [
        "allie goertz",
        0
      ],
      [
        "allie goertz",
        1
      ],
      [
        "allie goertz",
        2
      ],
      [
        "milhouse van houten",
        0
      ]
    ],
    "answer": "President Richard Nixon"
  },
  "illegal_variant": {
    "submitted_jsonl": {
      "query_id": "5a8d7341554299441c6b9fe5",
      "cell_id": "hotpot_5a8d7341554299441c6b9fe5",
      "tier": "B1_structured",
      "cost_menu": "hotpot-structured-v0-op",
      "route": "sentence",
      "supporting_facts": [
        [
          "allie goertz",
          0
        ],
        [
          "allie goertz",
          1
        ],
        [
          "allie goertz",
          2
        ],
        [
          "milhouse van houten",
          0
        ]
      ],
      "answer": "President Richard Nixon"
    },
    "rejection_reason": "supporting_facts and answers are evaluator-only fields."
  },
  "scope_note": "This is structured evidence acquisition over HotpotQA distractor contexts, not answer generation."
}
```
