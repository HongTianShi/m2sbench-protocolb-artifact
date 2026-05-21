# 2Wiki Hyperlink-Corpus Systems Profile

Derived from `reports/2wiki_hyperlink_corpus_stress_2000.json`. The profile records cold streaming/materialization of the 7GB hyperlink paragraph corpus for the structured evidence-acquisition slice. It is not an answer-generation or GraphRAG serving benchmark.

| item | value | readout |
| --- | --- | --- |
| Cold corpus materialization | 21.036 GB in 600.138 s | 35.052 MB/s; 0.3001 s/query amortized |
| Local evidence map | 32745 docs | 10900 context titles; 17706 1-hop; 10272 2-hop |
| Best fixed vs adaptive | 0.782 -> 0.807 | oracle 0.872; repeated delta 0.0219 |
| Graph controls | real 1-hop 0.782 | shuffled 0.301; degree-random 0.722; random 2-hop 0.728 |

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "n_rows": 2000,
  "passes": 3,
  "total_bytes_streamed": 21035908006,
  "total_gb_streamed": 21.036,
  "total_lines_scanned": 17968955,
  "total_scan_seconds": 600.138,
  "mean_scan_mb_per_s": 35.052,
  "cold_materialization_s_per_query": 0.3001,
  "cold_streamed_mb_per_query": 10.518,
  "context_titles": 10900,
  "selected_1hop_unique": 17706,
  "selected_2hop_unique": 10272,
  "doc_map_size": 32745,
  "mean_1hop_per_query": 15.912,
  "mean_2hop_per_query": 16.0,
  "found_1hop": 14836,
  "found_2hop": 9117
}
```