# 2Wiki Hyperlink-Corpus Stress Audit

Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.

- Rows: 2000 ({'train': 1200, 'dev': 400, 'test': 400})
- Expansion caps: 1-hop=16, 2-hop=16
- Top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.6125, 'hyperlink_1hop': 0.1775, 'full_local_pool': 0.01, 'hyperlink_2hop': 0.0525, 'provided_context': 0.1475}

## Corpus expansion/profile

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "seed_pass": {
    "pass": "seed_context_titles",
    "needed": 10900,
    "found": 10900,
    "seconds": 198.427,
    "lines_scanned": 5989261,
    "bytes_scanned": 7011530548,
    "scan_mb_per_s": 35.336
  },
  "first_hop_pass": {
    "pass": "selected_1hop_titles",
    "needed": 17706,
    "found": 14836,
    "seconds": 199.66,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 35.121
  },
  "second_hop_pass": {
    "pass": "selected_2hop_titles",
    "needed": 10272,
    "found": 9117,
    "seconds": 202.051,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 34.705
  },
  "selected_1hop_unique": 17706,
  "selected_2hop_unique": 10272,
  "doc_map_size": 32745,
  "mean_1hop_per_query": 15.9125,
  "mean_2hop_per_query": 16.0
}
```

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7772 | 0.7772 | 0.7331 | 0 | 0.0946 | -0.05 | -0.0045 | {'summary': 1.0} |
| fixed_provided_context | 0.7346 | 0.741 | 0.7475 | 0.08 | 0.1372 | -0.5226 | -0.0471 | {'provided_context': 1.0} |
| fixed_hyperlink_1hop | 0.7817 | 0.7977 | 0.8081 | 0.2 | 0.0901 | 0 | 0 | {'hyperlink_1hop': 1.0} |
| fixed_hyperlink_2hop | 0.7735 | 0.8007 | 0.8175 | 0.34 | 0.0984 | -0.0915 | -0.0082 | {'hyperlink_2hop': 1.0} |
| fixed_full_local_pool | 0.7371 | 0.7811 | 0.8106 | 0.55 | 0.1348 | -0.4951 | -0.0446 | {'full_local_pool': 1.0} |
| threshold_gate(neg_entropy,hyperlink_1hop) | 0.8126 | 0.8221 | 0.8081 | 0.1195 | 0.0593 | 0.3421 | 0.0308 | {'summary': 0.4025, 'hyperlink_1hop': 0.5975} |
| best_legal_router(rf_hyper) | 0.8071 | 0.8156 | 0.8037 | 0.1067 | 0.0648 | 0.2809 | 0.0253 | {'summary': 0.5325, 'provided_context': 0.075, 'hyperlink_1hop': 0.245, 'hyperlink_2hop': 0.14, 'full_local_pool': 0.0075} |
| best_legal_adaptive | 0.8071 | 0.8156 | 0.8037 | 0.1067 | 0.0648 | 0.2809 | 0.0253 | {'summary': 0.5325, 'provided_context': 0.075, 'hyperlink_1hop': 0.245, 'hyperlink_2hop': 0.14, 'full_local_pool': 0.0075} |
| oracle_route | 0.8719 | 0.8775 | 0.8688 | 0.0706 | 0 | 1 | 0.0901 | {'summary': 0.6125, 'provided_context': 0.1475, 'hyperlink_1hop': 0.1775, 'hyperlink_2hop': 0.0525, 'full_local_pool': 0.01} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7772 | 0.7772 | 0.7331 | 0.5265 | 0.3673 | 0 | 0.0946 | -0.0045 |
| provided_context | 0.7346 | 0.741 | 0.7475 | 0.5417 | 0.3878 | 0.08 | 0.1372 | -0.0471 |
| real_1hop | 0.7817 | 0.7977 | 0.8081 | 0.5831 | 0.4189 | 0.2 | 0.0901 | 0 |
| shuffled_1hop | 0.3013 | 0.3173 | 0.3819 | 0.2819 | 0.2078 | 0.2 | 0.5706 | -0.4804 |
| degree_random_1hop | 0.7224 | 0.7384 | 0.7194 | 0.519 | 0.3686 | 0.2 | 0.1495 | -0.0594 |
| real_2hop | 0.7735 | 0.8007 | 0.8175 | 0.5904 | 0.4236 | 0.34 | 0.0984 | -0.0082 |
| random_2hop | 0.7284 | 0.7556 | 0.7394 | 0.5335 | 0.3811 | 0.34 | 0.1435 | -0.0534 |
| full_local_pool | 0.7371 | 0.7811 | 0.8106 | 0.5856 | 0.4191 | 0.55 | 0.1348 | -0.0446 |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge_comparison | 83 | fixed_hyperlink_1hop | 0.6994 | 0.7826 | 0.0832 |
| comparison | 127 | fixed_summary | 0.985 | 0.9928 | 0.0078 |
| compositional | 185 | fixed_hyperlink_1hop | 0.7442 | 0.8296 | 0.0853 |
| inference | 5 | fixed_summary | 0.8453 | 0.8453 | 0 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.02185070602881669,
  "ci95": [
    0.015003085723234852,
    0.03126581547018795
  ],
  "positive_share": 1.0,
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
