# 2Wiki Hyperlink-Corpus Stress Audit

Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.

- Rows: 1000 ({'train': 600, 'dev': 200, 'test': 200})
- Expansion caps: 1-hop=16, 2-hop=16
- Top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.595, 'provided_context': 0.14, 'hyperlink_1hop': 0.16, 'hyperlink_2hop': 0.095, 'full_local_pool': 0.01}

## Corpus expansion/profile

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "seed_pass": {
    "pass": "seed_context_titles",
    "needed": 5812,
    "found": 5812,
    "seconds": 225.495,
    "lines_scanned": 5987436,
    "bytes_scanned": 7009187318,
    "scan_mb_per_s": 31.084
  },
  "first_hop_pass": {
    "pass": "selected_1hop_titles",
    "needed": 9880,
    "found": 8302,
    "seconds": 230.494,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 30.422
  },
  "second_hop_pass": {
    "pass": "selected_2hop_titles",
    "needed": 6284,
    "found": 5607,
    "seconds": 227.718,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 30.793
  },
  "selected_1hop_unique": 9880,
  "selected_2hop_unique": 6284,
  "doc_map_size": 18720,
  "mean_1hop_per_query": 15.924,
  "mean_2hop_per_query": 16.0
}
```

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7627 | 0.7627 | 0.7087 | 0 | 0.0949 | 0 | 0 | {'summary': 1.0} |
| fixed_provided_context | 0.7218 | 0.7282 | 0.7125 | 0.08 | 0.1358 | -0.4304 | -0.0408 | {'provided_context': 1.0} |
| fixed_hyperlink_1hop | 0.7625 | 0.7785 | 0.7887 | 0.2 | 0.095 | -0.0013 | -0.0001 | {'hyperlink_1hop': 1.0} |
| fixed_hyperlink_2hop | 0.7458 | 0.773 | 0.7963 | 0.34 | 0.1118 | -0.178 | -0.0169 | {'hyperlink_2hop': 1.0} |
| fixed_full_local_pool | 0.7107 | 0.7547 | 0.7887 | 0.55 | 0.1469 | -0.5472 | -0.0519 | {'full_local_pool': 1.0} |
| threshold_gate(neg_entropy,hyperlink_2hop) | 0.7785 | 0.7955 | 0.7913 | 0.2125 | 0.079 | 0.1674 | 0.0159 | {'summary': 0.375, 'hyperlink_2hop': 0.625} |
| best_legal_router(et_hyper) | 0.7921 | 0.8012 | 0.7762 | 0.1137 | 0.0655 | 0.31 | 0.0294 | {'summary': 0.49, 'provided_context': 0.095, 'hyperlink_1hop': 0.25, 'hyperlink_2hop': 0.165} |
| best_legal_adaptive | 0.7921 | 0.8012 | 0.7762 | 0.1137 | 0.0655 | 0.31 | 0.0294 | {'summary': 0.49, 'provided_context': 0.095, 'hyperlink_1hop': 0.25, 'hyperlink_2hop': 0.165} |
| oracle_route | 0.8576 | 0.8641 | 0.8575 | 0.081 | 0 | 1 | 0.0949 | {'summary': 0.595, 'provided_context': 0.14, 'hyperlink_1hop': 0.16, 'hyperlink_2hop': 0.095, 'full_local_pool': 0.01} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7627 | 0.7627 | 0.7087 | 0.5088 | 0.3606 | 0 | 0.0949 | 0 |
| provided_context | 0.7218 | 0.7282 | 0.7125 | 0.5117 | 0.3662 | 0.08 | 0.1358 | -0.0408 |
| real_1hop | 0.7625 | 0.7785 | 0.7887 | 0.5663 | 0.4083 | 0.2 | 0.095 | -0.0001 |
| shuffled_1hop | 0.2966 | 0.3126 | 0.375 | 0.2767 | 0.2093 | 0.2 | 0.5609 | -0.466 |
| degree_random_1hop | 0.7234 | 0.7394 | 0.7188 | 0.5179 | 0.3713 | 0.2 | 0.1341 | -0.0392 |
| real_2hop | 0.7458 | 0.773 | 0.7963 | 0.5729 | 0.4149 | 0.34 | 0.1118 | -0.0169 |
| random_2hop | 0.7055 | 0.7327 | 0.7163 | 0.5154 | 0.3682 | 0.34 | 0.1521 | -0.0572 |
| full_local_pool | 0.7107 | 0.7547 | 0.7887 | 0.5679 | 0.4095 | 0.55 | 0.1469 | -0.0519 |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge_comparison | 41 | fixed_hyperlink_2hop | 0.6574 | 0.7436 | 0.0863 |
| comparison | 64 | fixed_summary | 0.9845 | 0.9936 | 0.0091 |
| compositional | 91 | fixed_hyperlink_1hop | 0.7231 | 0.8155 | 0.0924 |
| inference | 4 | fixed_hyperlink_1hop | 0.7906 | 0.805 | 0.0144 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.019576892856128257,
  "ci95": [
    0.0008111338456204763,
    0.031050607778929457
  ],
  "positive_share": 0.9,
  "repeats": 10
}
```

## Protocol B legal/illegal trace

```json
{
  "visible_fields": {
    "query_id": "61df8a820bdc11eba7f7acde48001122",
    "question": "When is the composer of film Sruthilayalu 's birthday?",
    "declared_views": [
      "summary",
      "provided_context",
      "hyperlink_1hop",
      "hyperlink_2hop",
      "full_local_pool"
    ],
    "visible_titles": [
      "Sruthilayalu",
      "Theodred II (Bishop of Elmham)",
      "Abe Meyer",
      "Alonso Mudarra"
    ],
    "released_corpus_view": "title sketch plus declared hyperlink-expansion budget; support labels are hidden"
  },
  "legal_action": {
    "submitted_jsonl": {
      "query_id": "61df8a820bdc11eba7f7acde48001122",
      "cell_id": "2wiki_hyper_61df8a820bdc11eba7f7acde48001122",
      "tier": "B1_hyperlink",
      "cost_menu": "2wiki-hyper-v1-op",
      "ranked_views": [
        "hyperlink_1hop",
        "provided_context",
        "hyperlink_2hop"
      ],
      "route": "hyperlink_1hop"
    },
    "charged_cost": 0.2,
    "scored_utility": 0.984
  },
  "hidden_evaluator_fields": {
    "support_titles": [
      "k. v. mahadevan",
      "sruthilayalu"
    ],
    "evidence_triples": [
      [
        "Sruthilayalu",
        "composer",
        "K. V. Mahadevan"
      ],
      [
        "K. V. Mahadevan",
        "date of birth",
        "14 March 1918"
      ]
    ],
    "answer": "14 March 1918"
  },
  "illegal_variant": {
    "submitted_jsonl": {
      "query_id": "61df8a820bdc11eba7f7acde48001122",
      "cell_id": "2wiki_hyper_61df8a820bdc11eba7f7acde48001122",
      "tier": "B1_hyperlink",
      "cost_menu": "2wiki-hyper-v1-op",
      "route": "hyperlink_2hop",
      "support_titles": [
        "k. v. mahadevan",
        "sruthilayalu"
      ],
      "answer": "14 March 1918"
    },
    "rejection_reason": "support titles and answers are evaluator-only fields, even when hyperlink expansion is legal."
  },
  "scope_note": "This is a hyperlink-corpus evidence-acquisition trace, not a generated-answer RAG transcript."
}
```