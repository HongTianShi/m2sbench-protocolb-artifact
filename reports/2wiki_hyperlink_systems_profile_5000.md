# 2Wiki Hyperlink-Corpus Systems Profile

Derived from `reports/2wiki_hyperlink_corpus_stress_5000.json`. The profile records cold streaming/materialization of the 7GB hyperlink paragraph corpus for the structured evidence-acquisition slice. It is not an answer-generation or GraphRAG serving benchmark.

| item | value | readout |
| --- | --- | --- |
| Cold corpus materialization | 21.036 GB in 626.0 s | 33.604 MB/s; 0.1252 s/query amortized |
| Local evidence map | 66691 docs | 24795 context titles; 36624 1-hop; 19311 2-hop |
| Best fixed vs adaptive | 0.792 -> 0.818 | oracle 0.881; repeated delta 0.0286 |
| Graph controls | real 1-hop 0.792 | shuffled 0.334; degree-random 0.726; random 2-hop 0.718 |

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "n_rows": 5000,
  "passes": 3,
  "total_bytes_streamed": 21035908006,
  "total_gb_streamed": 21.036,
  "total_lines_scanned": 17968955,
  "total_scan_seconds": 626.0,
  "mean_scan_mb_per_s": 33.604,
  "cold_materialization_s_per_query": 0.1252,
  "cold_streamed_mb_per_query": 4.207,
  "context_titles": 24795,
  "selected_1hop_unique": 36624,
  "selected_2hop_unique": 19311,
  "doc_map_size": 66691,
  "mean_1hop_per_query": 15.895,
  "mean_2hop_per_query": 16.0,
  "found_1hop": 30434,
  "found_2hop": 17013
}
```