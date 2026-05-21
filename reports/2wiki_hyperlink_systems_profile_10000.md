# 2Wiki Hyperlink-Corpus Systems Profile

Derived from `reports/2wiki_hyperlink_corpus_stress_10000.json`. The profile records cold streaming/materialization of the 7GB hyperlink paragraph corpus for the structured evidence-acquisition slice. It is not an answer-generation, GraphRAG serving, `C_mem`, or `C_lat` profile. The only official 2Wiki scoring profile is the declared `C_op` menu; these streaming fields are audit metadata.

Profiler context: this row was produced as a batch materialization audit on the
same local WSL2/Ubuntu + Python 3.10 + RTX 4070 Laptop GPU environment used for
the overnight mainline run. The `.059 s/query` number is amortized over the
10k-row corpus-expansion job and is not a per-query serving latency. Clean
release archives keep this compact profile and exclude raw cache maps/logs.

| item | value | readout |
| --- | --- | --- |
| Cold corpus materialization | 21.036 GB in 590.406 s | 35.63 MB/s; 0.059 s/query amortized |
| Local evidence map | 111863 docs | 45406 context titles; 62209 1-hop; 30409 2-hop |
| Best fixed vs adaptive | 0.791 -> 0.819 | oracle 0.886; repeated delta 0.0296 |
| Graph controls | real 1-hop 0.791 | shuffled 0.326; degree-random 0.736; random 2-hop 0.720 |

```json
{
  "corpus": "<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
  "n_rows": 10000,
  "passes": 3,
  "total_bytes_streamed": 21036225042,
  "total_gb_streamed": 21.036,
  "total_lines_scanned": 17969214,
  "total_scan_seconds": 590.406,
  "mean_scan_mb_per_s": 35.63,
  "cold_materialization_s_per_query": 0.059,
  "cold_streamed_mb_per_query": 2.104,
  "context_titles": 45406,
  "selected_1hop_unique": 62209,
  "selected_2hop_unique": 30409,
  "doc_map_size": 111863,
  "mean_1hop_per_query": 15.896,
  "mean_2hop_per_query": 16.0,
  "found_1hop": 51172,
  "found_2hop": 26502
}
```
