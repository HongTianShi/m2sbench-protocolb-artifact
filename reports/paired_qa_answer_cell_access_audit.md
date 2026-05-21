# Paired-QA Answer-Cell Access Audit

This optional audit instantiates the access-routing contract on paired question-answer data. It is not part of the primary semantic IVF-PQ leaderboard. Its purpose is to check whether the same evidence-access question appears when qrels are exact paired answers rather than label-derived relevance surrogates.

## Setup

- Data slice: 30,000 public question-answer pairs staged locally.
- Query: the question text.
- Candidate set: answers inside the IVF-style cell containing the paired gold answer.
- Relevance: the paired answer is the single evaluator-held relevant item.
- Vectorization: character n-gram hashing, TF-IDF weighting, 128-dimensional SVD embeddings.
- Views:
  - summary: cell-only order with no item-level evidence;
  - PQ: compressed 32-dimensional evidence;
  - full: 128-dimensional answer evidence.
- Metric: NDCG@10 before cost and utility after the default access menu, with lambda = 0.08 and costs summary/PQ/full = 0/0.20/0.58.

## Results

| view | NDCG@10 | utility | regret | hit@10 |
|---|---:|---:|---:|---:|
| summary | 0.029 | 0.029 | 0.321 | 0.064 |
| PQ | 0.164 | 0.148 | 0.201 | 0.293 |
| full | 0.334 | 0.287 | 0.063 | 0.513 |

Cell size median/p90/max: 172 / 276 / 339.

Per-query access diagnostics:

- Non-full views are cost-adjusted best on 60.9% of queries.
- PQ dominates full before cost on 59.7% of queries.
- Aggregate break-even thresholds: summary to PQ lambda* = 0.676; PQ to full lambda* = 0.446.

## Reading

This slice is evidence-essential in aggregate: full answer evidence gives the best mean utility under the default menu. At the query level, however, many answer cells still do not justify full evidence, so the access decision remains heterogeneous. This complements the main IVF-PQ text audit, where compressed evidence wins on average, by showing the opposite boundary case under exact paired-answer qrels.
