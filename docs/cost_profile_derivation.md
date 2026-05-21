# Declared Cost Profile Card

M2S-Bench uses declared scoring profiles, not universal deployment prices.
The official leaderboard profile `C_op` is a frozen operation-unit convention.
`C_mem` and `C_lat` are secondary dense rescoring profiles derived from the
unified systems audit fields and are used to show profile sensitivity.

Scoring uses `U = A - lambda C`, with `lambda = 0.08` for the headline profiles.

Machine-readable companion: `docs/cost_profile_derivation.json`. That file
records the declared dependency parents, normalized operation components, and
generated cost-menu ids used by the validator-facing menus.
The lightweight verifier `scripts/verify_systems_profile.py` recomputes
`materialized_bytes`, `C_mem`, `C_lat`, `qps_from_p50`, and fixed-profile
winners from the retained systems-profile JSON.

The `C_op` values are not learned from the outcome tables. They are a frozen
leaderboard convention chosen to satisfy an intended access-ordering rule:
released summaries are free; compressed signs and PQ are cheaper than dense
materialization; shallow ANN depth is cheaper than deeper ANN/full vectors; and
CE is charged either as an incremental reranker purchase or as the cumulative
richest ladder view, depending on the declared report context.

The JSON cost menu is scorer truth. For audit provenance, the HNSW rows also
retain unrounded source values (`0.1375` for shallow HNSW and `0.1125` for the
increment from HNSW16 to HNSW64); the public scorer uses the rounded menu
values shown below.

## Dense Semantic Profiles

| View | C_op declared cost | Rationale | Secondary profile source |
| --- | ---: | --- | --- |
| `summary` | 0.000 | Free released compact state. | none |
| `binary` | 0.080 | Cheap compressed sign view. | `reports/unified_systems_profile_audit.md` |
| `pq` | 0.200 | IVF-PQ/compressed vector access. | `reports/unified_systems_profile_audit.md` |
| `int8` | 0.300 | Post-training int8 dense view. | `reports/unified_systems_profile_audit.md` |
| `hnsw16` | 0.138 | Shallow ANN-depth view; rounded official C_op value. | `reports/unified_systems_profile_audit.md` |
| `hnsw64` | 0.250 | Deeper ANN view after shallow graph access; incremental official value is 0.113 after rounding. | `reports/unified_systems_profile_audit.md` |
| `full` | 0.580 | Full dense vector/materialized first-stage evidence. | `reports/unified_systems_profile_audit.md` |
| `ce` | 1.030 | Cumulative CE ladder view when cross-encoder reranking is treated as richest evidence. | `reports/unified_systems_profile_audit.md` |

CE also has an incremental purchase interpretation: `0.45` is the CE-only
increment after a first-stage pool exists; `1.03` is the cumulative ladder cost
when CE is scored as a full text-reranking view. The paper uses whichever
accounting context the corresponding table declares. CE-purchase reports mark
the active context as incremental; dense ladder/joint-menu reports use the
cumulative `dense-semantic-v1-op` menu.

## Secondary Dense Profiles

`menus/dense_semantic.C_mem.cost_menu.json` and
`menus/dense_semantic.C_lat.cost_menu.json` set `declared_cost` to the active
normalized rescoring cost for that profile.

They retain the original operation-unit ladder fields for traceability:

- `incremental_cost` and `cumulative_cost` are the original C_op ladder values.
- `declared_cost` is the active C_mem or C_lat cost used when that menu is the
  scorer input.
- `qps` records sustained median QPS from the source systems audit, not
  `QPS* = 1e6 / p50_us`.

This is why local latency can make full vectors appear cheaper than PQ while
the official operation-unit profile still charges full vectors more: Protocol B
requires a declared profile and stores enough audit fields for rescoring.

## 2Wiki Structured Profiles

The 2Wiki public-dev and paper reports use `C_op` only.

| Slice | View | C_op cost | Rationale |
| --- | --- | ---: | --- |
| `2wiki_structured_evidence` | `summary` | 0.000 | Released title/entity sketch. |
| `2wiki_structured_evidence` | `one_hop` | 0.160 | One-hop/context-neighborhood evidence. |
| `2wiki_structured_evidence` | `two_hop` | 0.260 | Two-hop/path evidence. |
| `2wiki_structured_evidence` | `full_context` | 0.550 | Full local context evidence. |
| `2wiki_structured_evidence` | `ce` | 0.950 | Cumulative CE/reranking view over structured candidate context. |
| `2wiki_hyperlink_stress` | `summary` | 0.000 | Released title/entity sketch. |
| `2wiki_hyperlink_stress` | `provided_context` | 0.080 | Dataset-provided local context access. |
| `2wiki_hyperlink_stress` | `hyperlink_1hop` | 0.200 | 1-hop hyperlink expansion. |
| `2wiki_hyperlink_stress` | `hyperlink_2hop` | 0.340 | 2-hop hyperlink/path expansion. |
| `2wiki_hyperlink_stress` | `full_local_pool` | 0.550 | Full local paragraph pool. |

The 2Wiki hyperlink systems report records corpus streaming/materialization
fields, but no official `C_mem` or `C_lat` rescoring menu is declared for
2Wiki. Those fields are audit metadata, not deployment-cost truth.

## Profiler Metadata

The local systems profile is a calibration audit, not a deployment-cost claim.
The frozen profiler was run on WSL2/Ubuntu with Python 3.10, PyTorch
2.5.1+cu121, FAISS-GPU 1.11.0, and an RTX 4070 Laptop GPU. The host machine
reports an Intel Core i7-14700HX CPU, 20 physical cores / 28 logical threads,
about 16 GB RAM, and a YMTC YMSS2ED08D25MC fixed disk. Dense/CE timing rows are
batch/profile measurements from `reports/unified_systems_profile_audit.*`;
2Wiki hyperlink rows are cold streaming/materialization measurements from
`reports/2wiki_hyperlink_systems_profile_10000.*`.
