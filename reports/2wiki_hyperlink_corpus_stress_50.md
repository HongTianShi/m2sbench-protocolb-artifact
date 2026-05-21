# 2Wiki Hyperlink-Corpus Stress Audit

Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.

- Rows: 50 ({'train': 30, 'dev': 10, 'test': 10})
- Expansion caps: 1-hop=16, 2-hop=16
- Top-k: 4; lambda: 0.08
- Oracle view share: {'hyperlink_1hop': 0.3, 'provided_context': 0.1, 'full_local_pool': 0.1, 'summary': 0.4, 'hyperlink_2hop': 0.1}

## Corpus expansion/profile

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "seed_pass": {
    "pass": "seed_context_titles",
    "needed": 380,
    "found": 380,
    "seconds": 220.498,
    "lines_scanned": 5975774,
    "bytes_scanned": 6994900760,
    "scan_mb_per_s": 31.723
  },
  "first_hop_pass": {
    "pass": "selected_1hop_titles",
    "needed": 680,
    "found": 583,
    "seconds": 215.972,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 32.468
  },
  "second_hop_pass": {
    "pass": "selected_2hop_titles",
    "needed": 578,
    "found": 511,
    "seconds": 222.262,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 31.549
  },
  "selected_1hop_unique": 680,
  "selected_2hop_unique": 578,
  "doc_map_size": 1439,
  "mean_1hop_per_query": 16.0,
  "mean_2hop_per_query": 16.0
}
```

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7726 | 0.7726 | 0.7 | 0 | 0.1504 | -0.9928 | -0.0749 | {'summary': 1.0} |
| fixed_provided_context | 0.7137 | 0.7201 | 0.675 | 0.08 | 0.2093 | -1.7736 | -0.1339 | {'provided_context': 1.0} |
| fixed_hyperlink_1hop | 0.8475 | 0.8635 | 0.925 | 0.2 | 0.0755 | 0 | 0 | {'hyperlink_1hop': 1.0} |
| fixed_hyperlink_2hop | 0.8237 | 0.8509 | 0.875 | 0.34 | 0.0993 | -0.3162 | -0.0239 | {'hyperlink_2hop': 1.0} |
| fixed_full_local_pool | 0.7968 | 0.8408 | 0.9 | 0.55 | 0.1262 | -0.6723 | -0.0507 | {'full_local_pool': 1.0} |
| threshold_gate(pool_count,hyperlink_2hop) | 0.7737 | 0.7792 | 0.725 | 0.068 | 0.1492 | -0.9775 | -0.0738 | {'summary': 0.8, 'hyperlink_2hop': 0.2} |
| best_legal_router(et_hyper) | 0.7803 | 0.7942 | 0.825 | 0.174 | 0.1427 | -0.8913 | -0.0673 | {'summary': 0.2, 'hyperlink_1hop': 0.7, 'hyperlink_2hop': 0.1} |
| best_legal_adaptive | 0.7737 | 0.7792 | 0.725 | 0.068 | 0.1492 | -0.9775 | -0.0738 | {'summary': 0.8, 'hyperlink_2hop': 0.2} |
| oracle_route | 0.923 | 0.9355 | 0.95 | 0.157 | 0 | 1 | 0.0755 | {'summary': 0.4, 'provided_context': 0.1, 'hyperlink_1hop': 0.3, 'hyperlink_2hop': 0.1, 'full_local_pool': 0.1} |

## Anti-token and path controls

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.7726 | 0.7726 | 0.7 | 0.5 | 0.3514 | 0 | 0.1504 | -0.0749 |
| provided_context | 0.7137 | 0.7201 | 0.675 | 0.4917 | 0.3464 | 0.08 | 0.2093 | -0.1339 |
| real_1hop | 0.8475 | 0.8635 | 0.925 | 0.6583 | 0.4857 | 0.2 | 0.0755 | 0 |
| shuffled_1hop | 0.3269 | 0.3429 | 0.45 | 0.35 | 0.2843 | 0.2 | 0.5961 | -0.5206 |
| degree_random_1hop | 0.741 | 0.757 | 0.7 | 0.5 | 0.3514 | 0.2 | 0.182 | -0.1066 |
| real_2hop | 0.8237 | 0.8509 | 0.875 | 0.625 | 0.4607 | 0.34 | 0.0993 | -0.0239 |
| random_2hop | 0.7271 | 0.7543 | 0.7 | 0.5 | 0.3514 | 0.34 | 0.1958 | -0.1204 |
| full_local_pool | 0.7968 | 0.8408 | 0.9 | 0.65 | 0.4807 | 0.55 | 0.1262 | -0.0507 |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge_comparison | 2 | fixed_provided_context | 0.7143 | 0.7541 | 0.0397 |
| comparison | 4 | fixed_summary | 1 | 1 | 0 |
| compositional | 3 | fixed_hyperlink_2hop | 0.9051 | 0.9126 | 0.0075 |
| inference | 1 | fixed_hyperlink_1hop | 0.984 | 0.984 | 0 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": -0.08452934129145889,
  "ci95": [
    -0.1430703423033602,
    -0.02598834027955759
  ],
  "positive_share": 0.0,
  "repeats": 2
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