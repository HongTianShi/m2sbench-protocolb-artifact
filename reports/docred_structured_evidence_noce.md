# DocRED Structured Evidence Acquisition Audit

Protocol B slice over DocRED relation evidence annotations. A method sees a document/relation/entity-pair sketch and may buy declared evidence views before evaluator-held evidence sentence ids are scored. This is evidence acquisition, not relation extraction training.

- Queries: 30,000; candidate sentences: 32,313; split: {'train': 18000, 'dev': 6000, 'test': 6000}
- Top-k: 5; lambda: 0.08
- Views: ['summary', 'co_mention', 'evidence_sentences', 'full_context']
- View costs: {'summary': 0.0, 'co_mention': 0.12, 'evidence_sentences': 0.22, 'full_context': 0.55}
- Oracle view share on hidden test: {'co_mention': 0.4915, 'summary': 0.39466666666666667, 'evidence_sentences': 0.07366666666666667, 'full_context': 0.04016666666666667}

## Fixed and adaptive policies

| policy | utility | raw_ndcg | evidence_f1 | cost | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_summary | 0.7208 | 0.7208 | 0.3951 | 0 | 0.2455 | -5.8253 | -0.2095 | {'summary': 1.0} |
| fixed_co_mention | 0.9304 | 0.94 | 0.4682 | 0.12 | 0.036 | 0 | 0 | {'co_mention': 1.0} |
| fixed_evidence_sentences | 0.8901 | 0.9077 | 0.4649 | 0.22 | 0.0762 | -1.119 | -0.0402 | {'evidence_sentences': 1.0} |
| fixed_full_context | 0.8164 | 0.8604 | 0.4557 | 0.55 | 0.1499 | -3.1688 | -0.114 | {'full_context': 1.0} |
| ridge_B1_entity | 0.9304 | 0.94 | 0.4682 | 0.12 | 0.036 | 0 | 0 | {'co_mention': 1.0} |
| oracle_route | 0.9663 | 0.9741 | 0.4778 | 0.0973 | 0 | 1 | 0.036 | {'co_mention': 0.4915, 'summary': 0.3947, 'evidence_sentences': 0.0737, 'full_context': 0.0402} |

## Legal feature-tier learner ablation

Rows use only released manifest summaries: candidate counts, entity/co-mention score summaries, candidate-sentence score summaries, and context-length metadata. Evidence sentence ids, relation labels used for scoring, unpaid full/CE scores, and oracle views remain evaluator-held.

| feature_tier | policy | utility | regret | gap_closed | diff_vs_best_fixed | view_share |
| --- | --- | --- | --- | --- | --- | --- |
| B0_summary | hgb_B0_summary | 0.9302 | 0.0361 | -0.0047 | -0.0002 | {'co_mention': 0.9993, 'evidence_sentences': 0.0007} |
| B1_entity | ridge_B1_entity | 0.9304 | 0.036 | 0 | 0 | {'co_mention': 1.0} |
| B1_sentence | ridge_B1_sentence | 0.9304 | 0.036 | 0 | 0 | {'co_mention': 1.0} |
| B1_context | ridge_B1_context | 0.9304 | 0.036 | 0 | 0 | {'co_mention': 1.0} |

## Real evidence views versus controls

| control | utility | raw_ndcg | evidence_f1 | hit | cost | diff_vs_best_control |
| --- | --- | --- | --- | --- | --- | --- |
| title_position | 0.7208 | 0.7208 | 0.3951 | 0.9062 | 0 | -0.2095 |
| real_co_mention | 0.9304 | 0.94 | 0.4682 | 0.9975 | 0.12 | 0 |
| shuffled_co_mention | 0.4474 | 0.457 | 0.3238 | 0.7965 | 0.12 | -0.483 |
| degree_random_co_mention | 0.5007 | 0.5103 | 0.3628 | 0.8585 | 0.12 | -0.4297 |
| same_doc_random | 0.4354 | 0.453 | 0.3215 | 0.7893 | 0.22 | -0.495 |
| evidence_sentences | 0.8901 | 0.9077 | 0.4649 | 0.998 | 0.22 | -0.0402 |
| full_context | 0.8164 | 0.8604 | 0.4557 | 0.9915 | 0.55 | -0.114 |

## Repeated split stability

```json
{
  "mean_learned_minus_best_fixed": 0.0,
  "ci95": [
    0.0,
    0.0
  ],
  "positive_share": 0.0,
  "selected_policies": {
    "ridge_B0_summary": 9,
    "ridge_B1_entity": 1
  }
}
```

## Reading

DocRED is a second structured-evidence family: methods buy entity/co-mention, candidate-sentence, full-context, or optional CE evidence before evaluator-held evidence-sentence qrels are scored.
