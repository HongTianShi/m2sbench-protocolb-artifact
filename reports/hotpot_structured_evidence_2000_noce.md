# HotpotQA Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence slice over HotpotQA distractor contexts. A method sees the question and title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'sentence': 0.115, 'summary': 0.4075, 'paragraph': 0.37, 'title_shortlist': 0.0425, 'full_context': 0.065}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.665 | 0.665 | 0.6775 | 0.4525 | 0.2905 | 0 | 0.2297 | -1.3168 | -0.1305 | {'summary': 1.0} |
| fixed_title_shortlist | 0.6734 | 0.6766 | 0.6787 | 0.4533 | 0.2913 | 0.04 | 0.2213 | -1.232 | -0.1221 | {'title_shortlist': 1.0} |
| fixed_paragraph | 0.7956 | 0.8084 | 0.8337 | 0.5567 | 0.3699 | 0.16 | 0.0991 | 0 | 0 | {'paragraph': 1.0} |
| fixed_sentence | 0.771 | 0.7934 | 0.8137 | 0.5433 | 0.4481 | 0.28 | 0.1237 | -0.2476 | -0.0245 | {'sentence': 1.0} |
| fixed_full_context | 0.7373 | 0.7813 | 0.8025 | 0.5358 | 0.3495 | 0.55 | 0.1574 | -0.5881 | -0.0583 | {'full_context': 1.0} |
| threshold_gate(summary,sentence) | 0.7584 | 0.7796 | 0.7987 | 0.5333 | 0.4319 | 0.2653 | 0.1363 | -0.3751 | -0.0372 | {'summary': 0.0525, 'sentence': 0.9475} |
| best_legal_router(hgb_B1_sentence_meta) | 0.8039 | 0.8186 | 0.8325 | 0.5558 | 0.4137 | 0.1833 | 0.0908 | 0.084 | 0.0083 | {'summary': 0.12, 'title_shortlist': 0.105, 'paragraph': 0.4, 'sentence': 0.3375, 'full_context': 0.0375} |
| best_legal_adaptive | 0.8039 | 0.8186 | 0.8325 | 0.5558 | 0.4137 | 0.1833 | 0.0908 | 0.084 | 0.0083 | {'summary': 0.12, 'title_shortlist': 0.105, 'paragraph': 0.4, 'sentence': 0.3375, 'full_context': 0.0375} |
| oracle_route | 0.8947 | 0.905 | 0.9125 | 0.6092 | 0.4165 | 0.1289 | 0 | 1 | 0.0991 | {'summary': 0.4075, 'title_shortlist': 0.0425, 'paragraph': 0.37, 'sentence': 0.115, 'full_context': 0.065} |

## Randomization controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.665 | 0.665 | 0.6775 | 0.4525 | 0.2905 | 0 | 0.2297 | -0.1305 |
| title_shortlist | 0.6734 | 0.6766 | 0.6787 | 0.4533 | 0.2913 | 0.04 | 0.2213 | -0.1221 |
| real_paragraph | 0.7956 | 0.8084 | 0.8337 | 0.5567 | 0.3699 | 0.16 | 0.0991 | 0 |
| real_sentence | 0.771 | 0.7934 | 0.8137 | 0.5433 | 0.4481 | 0.28 | 0.1237 | -0.0245 |
| shuffled_title | 0.3005 | 0.3005 | 0.3837 | 0.2567 | 0.1755 | 0 | 0.5942 | -0.4951 |
| random_paragraph | 0.2994 | 0.3122 | 0.4012 | 0.2683 | 0.1726 | 0.16 | 0.5953 | -0.4962 |
| random_sentence | 0.2719 | 0.2943 | 0.3837 | 0.2567 | 0.1725 | 0.28 | 0.6228 | -0.5236 |
| full_context | 0.7373 | 0.7813 | 0.8025 | 0.5358 | 0.3495 | 0.55 | 0.1574 | -0.0583 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B1_context_sketch | B1_context_sketch | 0.8053 | 0.2022 | 0.0894 | 0.0097 | 0.0983 | {'summary': 0.2, 'title_shortlist': 0.075, 'paragraph': 0.2625, 'sentence': 0.36, 'full_context': 0.1025} |
| et_B1_context_sketch | B1_context_sketch | 0.8053 | 0.2059 | 0.0894 | 0.0097 | 0.0978 | {'summary': 0.1525, 'title_shortlist': 0.105, 'paragraph': 0.2375, 'sentence': 0.4225, 'full_context': 0.0825} |
| hgb_B1_sentence_meta | B1_sentence_meta | 0.8039 | 0.1833 | 0.0908 | 0.0083 | 0.084 | {'summary': 0.12, 'title_shortlist': 0.105, 'paragraph': 0.4, 'sentence': 0.3375, 'full_context': 0.0375} |
| hgb_B1_paragraph_meta | B1_paragraph_meta | 0.8037 | 0.1833 | 0.091 | 0.0082 | 0.0824 | {'summary': 0.215, 'title_shortlist': 0.0225, 'paragraph': 0.3325, 'sentence': 0.3975, 'full_context': 0.0325} |
| hgb_B1_context_sketch | B1_context_sketch | 0.8031 | 0.218 | 0.0916 | 0.0075 | 0.0761 | {'summary': 0.0925, 'title_shortlist': 0.145, 'paragraph': 0.1575, 'sentence': 0.54, 'full_context': 0.065} |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.7969 | 0.2034 | 0.0978 | 0.0014 | 0.0137 | {'summary': 0.1775, 'title_shortlist': 0.0875, 'paragraph': 0.37, 'sentence': 0.2225, 'full_context': 0.1425} |
| rf_B1_sentence_meta | B1_sentence_meta | 0.7957 | 0.2047 | 0.099 | 0.0002 | 0.0017 | {'summary': 0.18, 'title_shortlist': 0.08, 'paragraph': 0.3625, 'sentence': 0.2375, 'full_context': 0.14} |
| et_B1_paragraph_meta | B1_paragraph_meta | 0.7956 | 0.2316 | 0.0991 | 0.0001 | 0.0008 | {'summary': 0.105, 'title_shortlist': 0.105, 'paragraph': 0.325, 'sentence': 0.2975, 'full_context': 0.1675} |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge | 332 | fixed_paragraph | 0.7791 | 0.8743 | 0.0952 |
| comparison | 68 | fixed_title_shortlist | 0.9846 | 0.9941 | 0.0096 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.013581019897280156,
  "ci95": [
    0.0032436550419641663,
    0.02762054618158514
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(rf_B1_context_sketch)": 1,
    "best_legal_router(hgb_B0_title)": 2,
    "best_legal_router(hgb_B1_context_sketch)": 4,
    "best_legal_router(et_B1_context_sketch)": 2,
    "best_legal_router(et_B0_title)": 1
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
      "full_context"
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
