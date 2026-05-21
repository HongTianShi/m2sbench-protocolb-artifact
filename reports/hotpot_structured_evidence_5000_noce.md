# HotpotQA Structured Evidence Acquisition Audit

Candidate Protocol B structured-evidence slice over HotpotQA distractor contexts. A method sees the question and title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

- Status: supporting_generalization_check
- Rows: 5000 ({'train': 3000, 'dev': 1000, 'test': 1000})
- Upstream split: `train`; internal route split: 3000/1000/1000 train/dev/test; top-k: 4; lambda: 0.08
- Oracle view share: {'sentence': 0.135, 'paragraph': 0.339, 'full_context': 0.086, 'summary': 0.407, 'title_shortlist': 0.033}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.6647 | 0.6647 | 0.6825 | 0.4565 | 0.3052 | 0 | 0.2377 | -1.0346 | -0.1209 | {'summary': 1.0} |
| fixed_title_shortlist | 0.6725 | 0.6757 | 0.6855 | 0.4585 | 0.3068 | 0.04 | 0.2299 | -0.9678 | -0.1131 | {'title_shortlist': 1.0} |
| fixed_paragraph | 0.7855 | 0.7983 | 0.823 | 0.5501 | 0.3704 | 0.16 | 0.1168 | 0 | 0 | {'paragraph': 1.0} |
| fixed_sentence | 0.7834 | 0.8058 | 0.824 | 0.5508 | 0.4713 | 0.28 | 0.119 | -0.0186 | -0.0022 | {'sentence': 1.0} |
| fixed_full_context | 0.7546 | 0.7986 | 0.818 | 0.5468 | 0.3561 | 0.55 | 0.1478 | -0.265 | -0.031 | {'full_context': 1.0} |
| threshold_gate(summary,paragraph) | 0.7752 | 0.7872 | 0.81 | 0.5415 | 0.3638 | 0.1496 | 0.1272 | -0.0885 | -0.0103 | {'summary': 0.065, 'paragraph': 0.935} |
| best_legal_router(et_B1_paragraph_meta) | 0.8057 | 0.825 | 0.838 | 0.5601 | 0.406 | 0.2416 | 0.0966 | 0.1727 | 0.0202 | {'summary': 0.101, 'title_shortlist': 0.11, 'paragraph': 0.335, 'sentence': 0.245, 'full_context': 0.209} |
| best_legal_adaptive | 0.8057 | 0.825 | 0.838 | 0.5601 | 0.406 | 0.2416 | 0.0966 | 0.1727 | 0.0202 | {'summary': 0.101, 'title_shortlist': 0.11, 'paragraph': 0.335, 'sentence': 0.245, 'full_context': 0.209} |
| oracle_route | 0.9024 | 0.9136 | 0.9245 | 0.6178 | 0.4418 | 0.1407 | 0 | 1 | 0.1168 | {'summary': 0.407, 'title_shortlist': 0.033, 'paragraph': 0.339, 'sentence': 0.135, 'full_context': 0.086} |

## Randomization controls

| control | utility | raw_ndcg | support_recall | support_f1 | support_sentence_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.6647 | 0.6647 | 0.6825 | 0.4565 | 0.3052 | 0 | 0.2377 | -0.1209 |
| title_shortlist | 0.6725 | 0.6757 | 0.6855 | 0.4585 | 0.3068 | 0.04 | 0.2299 | -0.1131 |
| real_paragraph | 0.7855 | 0.7983 | 0.823 | 0.5501 | 0.3704 | 0.16 | 0.1168 | 0 |
| real_sentence | 0.7834 | 0.8058 | 0.824 | 0.5508 | 0.4713 | 0.28 | 0.119 | -0.0022 |
| shuffled_title | 0.3058 | 0.3058 | 0.391 | 0.2621 | 0.1709 | 0 | 0.5966 | -0.4797 |
| random_paragraph | 0.3194 | 0.3322 | 0.419 | 0.2808 | 0.1832 | 0.16 | 0.5829 | -0.4661 |
| random_sentence | 0.2957 | 0.3181 | 0.4115 | 0.2758 | 0.1839 | 0.28 | 0.6067 | -0.4898 |
| full_context | 0.7546 | 0.7986 | 0.818 | 0.5468 | 0.3561 | 0.55 | 0.1478 | -0.031 |

## Feature-tier learner readout

| policy | feature_tier | utility | cost | regret | diff_vs_best_fixed | gap_closed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| et_B1_context_sketch | B1_context_sketch | 0.8166 | 0.2117 | 0.0857 | 0.0311 | 0.2661 | {'summary': 0.141, 'title_shortlist': 0.116, 'paragraph': 0.301, 'sentence': 0.312, 'full_context': 0.13} |
| hgb_B0_title | B0_title | 0.8115 | 0.2074 | 0.0908 | 0.026 | 0.2226 | {'summary': 0.165, 'title_shortlist': 0.055, 'paragraph': 0.247, 'sentence': 0.472, 'full_context': 0.061} |
| hgb_B1_context_sketch | B1_context_sketch | 0.8115 | 0.2172 | 0.0909 | 0.0259 | 0.222 | {'summary': 0.2, 'title_shortlist': 0.03, 'paragraph': 0.096, 'sentence': 0.63, 'full_context': 0.044} |
| rf_B1_context_sketch | B1_context_sketch | 0.8095 | 0.1983 | 0.0928 | 0.024 | 0.2054 | {'summary': 0.196, 'title_shortlist': 0.065, 'paragraph': 0.352, 'sentence': 0.272, 'full_context': 0.115} |
| hgb_B1_sentence_meta | B1_sentence_meta | 0.8085 | 0.204 | 0.0938 | 0.023 | 0.1967 | {'summary': 0.222, 'paragraph': 0.248, 'sentence': 0.471, 'full_context': 0.059} |
| hgb_B1_paragraph_meta | B1_paragraph_meta | 0.8067 | 0.2044 | 0.0957 | 0.0211 | 0.181 | {'summary': 0.184, 'title_shortlist': 0.033, 'paragraph': 0.252, 'sentence': 0.479, 'full_context': 0.052} |
| et_B1_paragraph_meta | B1_paragraph_meta | 0.8057 | 0.2416 | 0.0966 | 0.0202 | 0.1727 | {'summary': 0.101, 'title_shortlist': 0.11, 'paragraph': 0.335, 'sentence': 0.245, 'full_context': 0.209} |
| et_B0_title | B0_title | 0.8044 | 0.2443 | 0.098 | 0.0189 | 0.1614 | {'summary': 0.115, 'title_shortlist': 0.103, 'paragraph': 0.314, 'sentence': 0.25, 'full_context': 0.218} |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge | 806 | fixed_sentence | 0.7756 | 0.8827 | 0.1071 |
| comparison | 194 | fixed_title_shortlist | 0.9693 | 0.9839 | 0.0146 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.02257996531841693,
  "ci95": [
    0.011799607450076341,
    0.029470211828680253
  ],
  "positive_share": 1.0,
  "selected_policies": {
    "best_legal_router(hgb_B1_context_sketch)": 6,
    "best_legal_router(et_B1_context_sketch)": 2,
    "best_legal_router(et_B0_title)": 1,
    "best_legal_router(rf_B1_context_sketch)": 1
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

