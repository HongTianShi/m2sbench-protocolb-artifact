# 2Wiki Hyperlink-Corpus Stress Audit

Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.

- Rows: 5000 ({'train': 3000, 'dev': 1000, 'test': 1000})
- Expansion caps: 1-hop=16, 2-hop=16
- Top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.555, 'hyperlink_1hop': 0.174, 'hyperlink_2hop': 0.081, 'provided_context': 0.18, 'full_local_pool': 0.01}

## Corpus expansion/profile

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "seed_pass": {
    "pass": "seed_context_titles",
    "needed": 24795,
    "found": 24795,
    "seconds": 207.959,
    "lines_scanned": 5989261,
    "bytes_scanned": 7011530548,
    "scan_mb_per_s": 33.716
  },
  "first_hop_pass": {
    "pass": "selected_1hop_titles",
    "needed": 36624,
    "found": 30434,
    "seconds": 206.066,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 34.029
  },
  "second_hop_pass": {
    "pass": "selected_2hop_titles",
    "needed": 19311,
    "found": 17013,
    "seconds": 211.975,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 33.08
  },
  "selected_1hop_unique": 36624,
  "selected_2hop_unique": 19311,
  "doc_map_size": 66691,
  "mean_1hop_per_query": 15.895,
  "mean_2hop_per_query": 16.0
}
```

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7689 | 0.7689 | 0.7192 | 0 | 0.1122 | -0.2633 | -0.0234 | {'summary': 1.0} |
| fixed_provided_context | 0.7476 | 0.754 | 0.749 | 0.08 | 0.1335 | -0.5031 | -0.0447 | {'provided_context': 1.0} |
| fixed_hyperlink_1hop | 0.7922 | 0.8082 | 0.8135 | 0.2 | 0.0888 | 0 | 0 | {'hyperlink_1hop': 1.0} |
| fixed_hyperlink_2hop | 0.7855 | 0.8127 | 0.822 | 0.34 | 0.0956 | -0.0762 | -0.0068 | {'hyperlink_2hop': 1.0} |
| fixed_full_local_pool | 0.7459 | 0.7899 | 0.8145 | 0.55 | 0.1351 | -0.5213 | -0.0463 | {'full_local_pool': 1.0} |
| threshold_gate(neg_entropy,hyperlink_1hop) | 0.8083 | 0.8181 | 0.8057 | 0.1224 | 0.0728 | 0.1807 | 0.0161 | {'summary': 0.388, 'hyperlink_1hop': 0.612} |
| best_legal_router(et_hyper) | 0.8175 | 0.8264 | 0.8093 | 0.1113 | 0.0635 | 0.2847 | 0.0253 | {'summary': 0.517, 'provided_context': 0.051, 'hyperlink_1hop': 0.285, 'hyperlink_2hop': 0.146, 'full_local_pool': 0.001} |
| best_legal_adaptive | 0.8175 | 0.8264 | 0.8093 | 0.1113 | 0.0635 | 0.2847 | 0.0253 | {'summary': 0.517, 'provided_context': 0.051, 'hyperlink_1hop': 0.285, 'hyperlink_2hop': 0.146, 'full_local_pool': 0.001} |
| oracle_route | 0.8811 | 0.8877 | 0.8788 | 0.0822 | 0 | 1 | 0.0888 | {'summary': 0.555, 'provided_context': 0.18, 'hyperlink_1hop': 0.174, 'hyperlink_2hop': 0.081, 'full_local_pool': 0.01} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7689 | 0.7689 | 0.7192 | 0.5199 | 0.3612 | 0 | 0.1122 | -0.0234 |
| provided_context | 0.7476 | 0.754 | 0.749 | 0.5463 | 0.3867 | 0.08 | 0.1335 | -0.0447 |
| real_1hop | 0.7922 | 0.8082 | 0.8135 | 0.5907 | 0.4208 | 0.2 | 0.0888 | 0 |
| shuffled_1hop | 0.3341 | 0.3501 | 0.418 | 0.3102 | 0.2267 | 0.2 | 0.547 | -0.4582 |
| degree_random_1hop | 0.7259 | 0.7419 | 0.724 | 0.5253 | 0.3715 | 0.2 | 0.1551 | -0.0663 |
| real_2hop | 0.7855 | 0.8127 | 0.822 | 0.596 | 0.4241 | 0.34 | 0.0956 | -0.0068 |
| random_2hop | 0.7181 | 0.7453 | 0.7355 | 0.533 | 0.3775 | 0.34 | 0.163 | -0.0742 |
| full_local_pool | 0.7459 | 0.7899 | 0.8145 | 0.5905 | 0.4207 | 0.55 | 0.1351 | -0.0463 |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge_comparison | 219 | fixed_hyperlink_1hop | 0.7144 | 0.7935 | 0.0791 |
| comparison | 287 | fixed_summary | 0.9858 | 0.9952 | 0.0094 |
| compositional | 472 | fixed_hyperlink_1hop | 0.7627 | 0.8517 | 0.089 |
| inference | 22 | fixed_summary | 0.8366 | 0.8946 | 0.058 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.028608992756260658,
  "ci95": [
    0.024710132610070015,
    0.03563770142734867
  ],
  "positive_share": 1.0,
  "repeats": 20
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