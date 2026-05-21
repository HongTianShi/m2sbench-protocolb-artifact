# Dense View Taxonomy

The dense semantic menu intentionally flattens several adjacent retrieval
decisions into one Protocol B candidate-view list. Reports may use the full flat
menu or restricted submenus.

| View family | Views | Interpretation | Typical report use |
| --- | --- | --- | --- |
| Released summary | `summary` | Free compact retrieval state. | All dense reports. |
| Compressed representations | `binary`, `pq`, `int8` | Storage/latency/quality choices before full dense materialization. | Compression ladder; joint dense menu. |
| ANN-depth controls | `hnsw16`, `hnsw64` | Search-depth/deeper-access decisions. | HNSW restricted solver; joint dense menu. |
| Full vector evidence | `full` | Full dense first-stage evidence. | Standard-qrel anchor; joint menu. |
| Expensive reranking | `ce` | Cross-encoder/reranker-as-paid-view. | CE purchase reports use incremental CE cost; dense ladder reports use cumulative CE cost. |

Restricted-solver reports compare QPP/cascade/ANN-depth/reranker-style policies
inside their natural submenus. Joint-menu reports put all declared dense views
under one hidden-qrel, paid-view, cost-regret contract.
