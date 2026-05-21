# HotpotQA Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence slice over HotpotQA distractor contexts. A method sees the question and title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 100 ({'train': 60, 'dev': 20, 'test': 20})
- Upstream split: `train`; internal route split recorded in Rows; top-k: 4; lambda: 0.08
- Oracle view share: {'paragraph': 0.3, 'sentence': 0.2, 'summary': 0.45, 'title_shortlist': 0.05}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7026 | 0.7026 | 0.7 | 0.4667 | 0.2675 | 0 | 0.1687 | -1.2488 | -0.0937 | {'summary': 1.0} |
| fixed_title_shortlist | 0.7074 | 0.7106 | 0.7 | 0.4667 | 0.2675 | 0.04 | 0.1638 | -1.1844 | -0.0888 | {'title_shortlist': 1.0} |
| fixed_paragraph | 0.7546 | 0.7674 | 0.8 | 0.5333 | 0.3206 | 0.16 | 0.1166 | -0.5547 | -0.0416 | {'paragraph': 1.0} |
| fixed_sentence | 0.7962 | 0.8186 | 0.8 | 0.5333 | 0.4246 | 0.28 | 0.075 | 0 | 0 | {'sentence': 1.0} |
| fixed_full_context | 0.7279 | 0.7719 | 0.725 | 0.4833 | 0.2817 | 0.55 | 0.1433 | -0.9109 | -0.0683 | {'full_context': 1.0} |
| threshold_gate(summary,sentence) | 0.7842 | 0.8033 | 0.8 | 0.5333 | 0.4579 | 0.238 | 0.087 | -0.1596 | -0.012 | {'summary': 0.15, 'sentence': 0.85} |
| best_legal_router(hgb_B0_title) | 0.8044 | 0.8238 | 0.8 | 0.5333 | 0.4079 | 0.2425 | 0.0668 | 0.109 | 0.0082 | {'title_shortlist': 0.15, 'paragraph': 0.35, 'sentence': 0.35, 'full_context': 0.15} |
| best_legal_adaptive | 0.8044 | 0.8238 | 0.8 | 0.5333 | 0.4079 | 0.2425 | 0.0668 | 0.109 | 0.0082 | {'title_shortlist': 0.15, 'paragraph': 0.35, 'sentence': 0.35, 'full_context': 0.15} |
| oracle_route | 0.8712 | 0.8797 | 0.875 | 0.5833 | 0.3651 | 0.106 | 0 | 1 | 0.075 | {'summary': 0.45, 'title_shortlist': 0.05, 'paragraph': 0.3, 'sentence': 0.2} |

## Randomization controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7026 | 0.7026 | 0.7 | 0.4667 | 0.2675 | 0 | 0.1687 | -0.0937 |
| title_shortlist | 0.7074 | 0.7106 | 0.7 | 0.4667 | 0.2675 | 0.04 | 0.1638 | -0.0888 |
| real_paragraph | 0.7546 | 0.7674 | 0.8 | 0.5333 | 0.3206 | 0.16 | 0.1166 | -0.0416 |
| real_sentence | 0.7962 | 0.8186 | 0.8 | 0.5333 | 0.4246 | 0.28 | 0.075 | 0 |
| shuffled_title | 0.245 | 0.245 | 0.3 | 0.2 | 0.1278 | 0 | 0.6262 | -0.5512 |
| random_paragraph | 0.3362 | 0.349 | 0.375 | 0.25 | 0.1532 | 0.16 | 0.535 | -0.46 |
| random_sentence | 0.3929 | 0.4153 | 0.475 | 0.3167 | 0.2063 | 0.28 | 0.4783 | -0.4033 |
| full_context | 0.7279 | 0.7719 | 0.725 | 0.4833 | 0.2817 | 0.55 | 0.1433 | -0.0683 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rf_B1_sentence_meta | B1_sentence_meta | 0.8243 | 0.2355 | 0.0469 | 0.0281 | 0.3744 | {'summary': 0.1, 'paragraph': 0.25, 'sentence': 0.6, 'full_context': 0.05} |
| et_B1_sentence_meta | B1_sentence_meta | 0.8138 | 0.2895 | 0.0574 | 0.0176 | 0.2349 | {'summary': 0.1, 'paragraph': 0.25, 'sentence': 0.4, 'full_context': 0.25} |
| hgb_B1_sentence_meta | B1_sentence_meta | 0.8099 | 0.2235 | 0.0613 | 0.0137 | 0.1828 | {'summary': 0.1, 'title_shortlist': 0.05, 'paragraph': 0.25, 'sentence': 0.55, 'full_context': 0.05} |
| hgb_B1_paragraph_meta | B1_paragraph_meta | 0.8089 | 0.2365 | 0.0623 | 0.0127 | 0.1689 | {'title_shortlist': 0.2, 'paragraph': 0.3, 'sentence': 0.35, 'full_context': 0.15} |
| hgb_B0_title | B0_title | 0.8044 | 0.2425 | 0.0668 | 0.0082 | 0.109 | {'title_shortlist': 0.15, 'paragraph': 0.35, 'sentence': 0.35, 'full_context': 0.15} |
| hgb_B1_context_sketch | B1_context_sketch | 0.8017 | 0.2765 | 0.0695 | 0.0055 | 0.0727 | {'summary': 0.05, 'title_shortlist': 0.1, 'paragraph': 0.05, 'sentence': 0.65, 'full_context': 0.15} |
| et_B1_context_sketch | B1_context_sketch | 0.8012 | 0.282 | 0.07 | 0.005 | 0.0669 | {'summary': 0.1, 'title_shortlist': 0.05, 'paragraph': 0.1, 'sentence': 0.55, 'full_context': 0.2} |
| rf_B1_paragraph_meta | B1_paragraph_meta | 0.8007 | 0.2115 | 0.0705 | 0.0045 | 0.0602 | {'summary': 0.1, 'paragraph': 0.45, 'sentence': 0.4, 'full_context': 0.05} |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge | 17 | fixed_sentence | 0.7689 | 0.8492 | 0.0803 |
| comparison | 3 | fixed_summary | 0.9732 | 0.9957 | 0.0225 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.024849620633022373,
  "ci95": [
    0.022657989375291808,
    0.02704125189075294
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(hgb_B1_sentence_meta)": 1,
    "best_legal_router(rf_B1_paragraph_meta)": 1
  },
  "repeats": 2
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
