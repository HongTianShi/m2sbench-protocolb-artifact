# 2Wiki Hyperlink-Corpus Systems Profile

Derived from `reports/2wiki_hyperlink_corpus_stress_50.json`. The profile records cold streaming/materialization of the 7GB hyperlink paragraph corpus for the structured evidence-acquisition slice. It is not an answer-generation or GraphRAG serving benchmark.

| item | value | readout |
| --- | --- | --- |
| Cold corpus materialization | 21.019 GB in 658.732 s | 31.909 MB/s; 13.1746 s/query amortized |
| Local evidence map | 1439 docs | 380 context titles; 680 1-hop; 578 2-hop |
| Best fixed vs adaptive | 0.848 -> 0.774 | oracle 0.923; repeated delta -0.0845 |
| Graph controls | real 1-hop 0.848 | shuffled 0.327; degree-random 0.741; random 2-hop 0.727 |

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "n_rows": 50,
  "passes": 3,
  "total_bytes_streamed": 21019278218,
  "total_gb_streamed": 21.019,
  "total_lines_scanned": 17955468,
  "total_scan_seconds": 658.732,
  "mean_scan_mb_per_s": 31.909,
  "cold_materialization_s_per_query": 13.1746,
  "cold_streamed_mb_per_query": 420.386,
  "context_titles": 380,
  "selected_1hop_unique": 680,
  "selected_2hop_unique": 578,
  "doc_map_size": 1439,
  "mean_1hop_per_query": 16.0,
  "mean_2hop_per_query": 16.0,
  "found_1hop": 583,
  "found_2hop": 511
}
```