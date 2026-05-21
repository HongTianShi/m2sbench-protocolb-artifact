# DocRED Evidence Access Audit

This optional audit instantiates the Protocol B access contract as an evidence-sentence ranking task. It complements the IVF-PQ label-surrogate slice by using exact DocRED evidence annotations as qrels.

## Setup

- Dataset: DocRED train/dev documents with relation evidence annotations.
- Query: document title plus a head entity, tail entity, and relation description.
- Candidate set: sentences in the same document.
- Qrels: evaluator-held evidence sentence indices supplied by DocRED.
- Views:
  - summary: entity/co-mention lexical summary with no dense sentence evidence;
  - PQ dense: the first 64 dimensions of a dense sentence embedding;
  - full dense: full sentence-transformers/all-MiniLM-L6-v2 sentence evidence.
- Metric: NDCG@5 before cost and utility after the default access menu, with lambda = 0.08 and costs summary/PQ/full = 0.0/0.2/0.58.

## Results

Scored 30,000 relation queries over 32,313 candidate sentences. Median/p90 candidate sentences per query: 8.0/12.0. Runtime: 125.7s.

| view | NDCG@5 | utility | regret | hit@5 | cost | mean_us |
| --- | --- | --- | --- | --- | --- | --- |
| summary | 0.941 | 0.941 | 0.024 | 0.997 | 0 | 2.235 |
| pq_dense | 0.829 | 0.813 | 0.152 | 0.985 | 0.2 | 3.714 |
| full_dense | 0.861 | 0.814 | 0.15 | 0.991 | 0.58 | 4.403 |

Subset diagnostics:

| scope | queries | summary_ndcg | pq_ndcg | full_ndcg | best | non_full_best |
| --- | --- | --- | --- | --- | --- | --- |
| all | 30000 | 0.941 | 0.829 | 0.861 | summary | 0.967 |
| summary_imperfect | 6761 | 0.738 | 0.731 | 0.766 | summary | 0.852 |
| summary_hard | 418 | 0.319 | 0.579 | 0.607 | pq_dense | 0.72 |

Per-query access diagnostics:

- Non-full views are cost-adjusted best on 96.7% of queries.
- PQ dense dominates full dense before cost on 73.7% of queries.
- Best-view shares: summary 88.8%, PQ dense 7.9%, full dense 3.3%.
- Aggregate break-even thresholds: summary to PQ lambda* = -0.561; PQ to full lambda* = 0.085.

## Reading

This is a RAG-style evidence access boundary case with exact qrels. Cheap entity/co-mention evidence is often sufficient because many DocRED evidence sentences explicitly mention the queried entities. The hard subset records where this summary breaks down and dense sentence evidence becomes a targeted escalation view. The result supports the same access-routing contract without relying on label-derived pseudo-qrels.
