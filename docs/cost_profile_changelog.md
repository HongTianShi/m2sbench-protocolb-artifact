# Cost Profile Changelog And Accounting Notes

Protocol B scores routes under a declared profile. The profiles below are
leaderboard or rescoring conventions, not deployment-cost claims.

## Dense semantic profiles

| Profile | Menu file | Role |
| --- | --- | --- |
| `C_op` | `menus/dense_semantic.cost_menu.json` | Frozen operation-unit profile used by the paper-facing dense leaderboard rows. |
| `C_mem` | `menus/dense_semantic.C_mem.cost_menu.json` | Secondary materialization/bytes rescoring profile derived from `reports/unified_systems_profile_audit.json`. |
| `C_lat` | `menus/dense_semantic.C_lat.cost_menu.json` | Secondary p95-latency rescoring profile derived from the same profiler report. |

The official utility rule is `U = A - lambda*C`; regret is computed against the
best evaluator-held route under the same declared profile. The default dense
lambda is `.08`, and lambda-sweep reports are retained for sensitivity checks.

## CE accounting

The dense menu records two CE costs:

- `ce.incremental_cost = .45`: buying the CE reranker after a first-stage pool
  already exists.
- `ce.cumulative_cost = 1.03`: treating CE as the richest text-reranking view in
  the full compression ladder.

Reports that study selective CE purchase use the incremental accounting context.
Reports that place CE in the compression ladder use the cumulative context. Both
values are stored in the cost menus so future scorers can select the intended
profile explicitly.

## Systems profile caveat

`C_mem` and `C_lat` are diagnostic rescoring profiles from one profiler
configuration. They can choose different fixed winners from `C_op`; this
disagreement is expected and is why the artifact stores raw quality, declared
cost, latency, bytes, QPS, and profile ids separately.

For per-view values and provenance, see `docs/cost_profile_derivation.md`.
