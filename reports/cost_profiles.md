# Cost Profile Report Pointer

Canonical frozen reports:

- `reports/unified_systems_profile_audit.md`
- `reports/unified_systems_profile_audit.json`
- `reports/semantic_access_cost_profile.md`
- `reports/semantic_access_cost_profile.json`
- `reports/2wiki_hyperlink_systems_profile_10000.md`
- `reports/2wiki_hyperlink_systems_profile_10000.json`

Consistency checks retained outside the main paper-facing row:

- `reports/2wiki_hyperlink_systems_profile_5000.md`
- `reports/2wiki_hyperlink_systems_profile_5000.json`
- `reports/2wiki_hyperlink_systems_profile_2000.md`
- `reports/2wiki_hyperlink_systems_profile_2000.json`

Purpose: supports the paper's cost-profile derivation. `C_op` is the frozen leaderboard profile, while `C_mem` and `C_lat` are secondary materialization/latency rescoring profiles.

Cost constants provenance:

- `C_op` values are coarse, versioned prices assigned by the view-dependency
  DAG before scoring for leaderboard comparability. They are not tuned from
  retrieval outcomes and are not fitted from the local WSL2/RTX 4070 profiler.
- Cheap summaries have zero declared purchase cost; compressed/ANN views pay
  incremental access prices; full dense pays the cumulative dense-view price.
- CE uses two accounting contexts: `.45` is the incremental CE purchase cost
  after a first-stage candidate pool already exists, while `1.03` is the
  cumulative ladder cost when CE is treated as the richest text-reranking view.
- `C_mem` combines per-query bytes touched with resident index/model footprint
  as a conservative sensitivity profile; it is not a production amortization
  claim.
- `C_lat` is normalized p95 latency under the fixed profiler configuration,
  with CPU HNSW and GPU dense/CE operators, so PQ/full/HNSW reversals are
  implementation- and profile-dependent rather than universal cost claims.
