# 2Wiki Hyperlink-Corpus Stress Audit

Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.

Boundary: this is not an end-to-end RAG or answer-generation benchmark. It
scores which evidence view is worth buying before support-title scoring.
Generated answers, free-form rationales, and agent trajectories are not scored
unless a future slice declares them as structured route fields.

- Rows: 10000 ({'train': 6000, 'dev': 2000, 'test': 2000})
- Expansion caps: 1-hop=16, 2-hop=16
- Top-k: 4; lambda: 0.08
- Oracle view share: {'summary': 0.563, 'hyperlink_1hop': 0.1705, 'provided_context': 0.169, 'hyperlink_2hop': 0.085, 'full_local_pool': 0.0125}

Metric aliases: the menus use `support_f1` for the report column sometimes
called support-fact F1 in prose. Public data provenance, cache expectations,
control definitions, and legal feature-tier definitions are in
`docs/2wiki_data_card.md`.

## Corpus expansion/profile

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "seed_pass": {
    "pass": "seed_context_titles",
    "needed": 45406,
    "found": 45406,
    "seconds": 184.586,
    "lines_scanned": 5989520,
    "bytes_scanned": 7011847584,
    "scan_mb_per_s": 37.987
  },
  "first_hop_pass": {
    "pass": "selected_1hop_titles",
    "needed": 62209,
    "found": 51172,
    "seconds": 183.991,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 38.112
  },
  "second_hop_pass": {
    "pass": "selected_2hop_titles",
    "needed": 30409,
    "found": 26502,
    "seconds": 221.829,
    "lines_scanned": 5989847,
    "bytes_scanned": 7012188729,
    "scan_mb_per_s": 31.611
  },
  "selected_1hop_unique": 62209,
  "selected_2hop_unique": 30409,
  "doc_map_size": 111863,
  "mean_1hop_per_query": 15.8957,
  "mean_2hop_per_query": 16.0
}
```

## Fixed and adaptive policies

| policy | utility | raw_ndcg | recall | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.774 | 0.774 | 0.7255 | 0 | 0.1125 | -0.1783 | -0.017 | {'summary': 1.0} |
| fixed_provided_context | 0.7478 | 0.7542 | 0.7589 | 0.08 | 0.1386 | -0.4526 | -0.0432 | {'provided_context': 1.0} |
| fixed_hyperlink_1hop | 0.791 | 0.807 | 0.8195 | 0.2 | 0.0954 | 0 | 0 | {'hyperlink_1hop': 1.0} |
| fixed_hyperlink_2hop | 0.7901 | 0.8173 | 0.8326 | 0.34 | 0.0963 | -0.0095 | -0.0009 | {'hyperlink_2hop': 1.0} |
| fixed_full_local_pool | 0.7493 | 0.7933 | 0.8255 | 0.55 | 0.1371 | -0.4365 | -0.0417 | {'full_local_pool': 1.0} |
| threshold_gate(neg_entropy,hyperlink_1hop) | 0.81 | 0.8192 | 0.8085 | 0.115 | 0.0765 | 0.1988 | 0.019 | {'summary': 0.425, 'hyperlink_1hop': 0.575} |
| best_legal_router(et_hyper) | 0.8188 | 0.8279 | 0.8163 | 0.1131 | 0.0676 | 0.2917 | 0.0278 | {'summary': 0.5125, 'provided_context': 0.076, 'hyperlink_1hop': 0.2555, 'hyperlink_2hop': 0.1425, 'full_local_pool': 0.0135} |
| best_legal_adaptive | 0.8188 | 0.8279 | 0.8163 | 0.1131 | 0.0676 | 0.2917 | 0.0278 | {'summary': 0.5125, 'provided_context': 0.076, 'hyperlink_1hop': 0.2555, 'hyperlink_2hop': 0.1425, 'full_local_pool': 0.0135} |
| oracle_route | 0.8865 | 0.8931 | 0.8876 | 0.0834 | 0 | 1 | 0.0954 | {'summary': 0.563, 'provided_context': 0.169, 'hyperlink_1hop': 0.1705, 'hyperlink_2hop': 0.085, 'full_local_pool': 0.0125} |

## Anti-token and path controls

Title-only and degree-matched random controls are intentionally strong because
visible titles and entity priors already narrow many 2Wiki rows. The audit asks
whether real relation/path evidence still adds value beyond those shortcuts.

| control | utility | raw_ndcg | support_recall | support_f1 | triple_endpoint_f1 | cost | regret | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_only | 0.774 | 0.774 | 0.7255 | 0.5213 | 0.3664 | 0 | 0.1125 | -0.017 |
| provided_context | 0.7478 | 0.7542 | 0.7589 | 0.5501 | 0.3938 | 0.08 | 0.1386 | -0.0432 |
| real_1hop | 0.791 | 0.807 | 0.8195 | 0.5916 | 0.4253 | 0.2 | 0.0954 | 0 |
| shuffled_1hop | 0.3259 | 0.3419 | 0.4106 | 0.3021 | 0.2197 | 0.2 | 0.5605 | -0.4651 |
| degree_random_1hop | 0.7365 | 0.7525 | 0.7372 | 0.5311 | 0.3787 | 0.2 | 0.15 | -0.0545 |
| real_2hop | 0.7901 | 0.8173 | 0.8326 | 0.601 | 0.432 | 0.34 | 0.0963 | -0.0009 |
| random_2hop | 0.7195 | 0.7467 | 0.7358 | 0.5312 | 0.379 | 0.34 | 0.1669 | -0.0715 |
| full_local_pool | 0.7493 | 0.7933 | 0.8255 | 0.5959 | 0.4291 | 0.55 | 0.1371 | -0.0417 |

## Question-type oracle headroom

| type | n | best_fixed | fixed_utility | oracle_utility | oracle_gap |
| --- | --- | --- | --- | --- | --- |
| bridge_comparison | 413 | fixed_hyperlink_1hop | 0.7083 | 0.7939 | 0.0856 |
| comparison | 614 | fixed_summary | 0.9917 | 0.9959 | 0.0042 |
| compositional | 925 | fixed_hyperlink_2hop | 0.7588 | 0.8555 | 0.0967 |
| inference | 48 | fixed_summary | 0.8156 | 0.8791 | 0.0636 |

## Repeated split stability

```json
{
  "learned_minus_best_fixed_mean": 0.029625693815554376,
  "ci95": [
    0.02557919883610413,
    0.0336129854494959
  ],
  "positive_share": 1.0,
  "repeats": 10
}
```

## Protocol B legal/illegal trace

The hidden-field values below are placeholders. The artifact records this as a contract trace, not as solver-visible training data.

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
      "<evaluator-held support title>",
      "<evaluator-held support title>"
    ],
    "evidence_triples": [
      [
        "<head>",
        "<relation>",
        "<tail>"
      ],
      [
        "<head>",
        "<relation>",
        "<tail>"
      ]
    ],
    "answer": "<evaluator-held answer>"
  },
  "illegal_variant": {
    "submitted_jsonl": {
      "query_id": "61df8a820bdc11eba7f7acde48001122",
      "cell_id": "2wiki_hyper_61df8a820bdc11eba7f7acde48001122",
      "tier": "B1_hyperlink",
      "cost_menu": "2wiki-hyper-v1-op",
      "route": "hyperlink_2hop",
      "support_titles": [
        "<forbidden support title>",
        "<forbidden support title>"
      ],
      "answer": "<forbidden answer>"
    },
    "rejection_reason": "support titles and answers are evaluator-only fields, even when hyperlink expansion is legal."
  },
  "scope_note": "This is a hyperlink-corpus evidence-acquisition trace, not a generated-answer RAG transcript."
}
```
