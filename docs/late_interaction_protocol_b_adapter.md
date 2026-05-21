# Late-Interaction / PLAID-Style Protocol B Adapter Note

This is a non-scored adapter note. It is included to show how a production
late-interaction retrieval stack, such as a ColBERTv2/PLAID-style pipeline,
would be declared under the same Protocol B route schema. The current paper
does not report a ColBERTv2 or PLAID experiment and does not claim a new
late-interaction retrieval engine.

## Protocol B interpretation

Late-interaction systems often combine several stages: candidate generation,
compressed or centroid-based approximations, pruning, exact token-level MaxSim
scoring, and sometimes a downstream cross-encoder or LLM reranker. Protocol B
can represent these stages as candidate evidence views. A solver sees only the
declared method-visible state, chooses one route or a compiled route, and the
evaluator charges the declared view cost before applying hidden qrels.

## Example view menu

| View id | Role | Method-visible before purchase | Paid/evaluator-held before purchase |
| --- | --- | --- | --- |
| `summary` | Cheap state | Query id, cheap lexical or dense-pool statistics, declared menu and costs. | Qrels, final relevance labels, exact token-level scores. |
| `lexical_or_dense_pool` | First-stage pool | Pool size, approximate score statistics, optional public candidate ids. | Full per-token interactions and reranker scores. |
| `centroid_interaction` | PLAID-style approximate late-interaction evidence | Centroid ids, centroid-score summaries, pruning budget metadata. | Exact token-level MaxSim scores and full unpruned ranking. |
| `pruned_late_interaction` | Pruned late-interaction evidence | Declared pruning budget, retained-centroid or retained-token summary. | Exact unpruned ColBERT-style scores unless this view is bought. |
| `exact_late_interaction` | Full ColBERT-style evidence | View id, cost, scorer id, and any declared low-cost summary. | Exact token-level interaction scores before purchase. |
| `ce_or_llm_rerank` | Optional expensive reranker | Declared model/scorer id and incremental cost after a pool exists. | CE/LLM scores before purchase. |

## Legal route examples

The same route JSONL object is used as in the dense and 2Wiki fixtures:

```json
{"query_id":"q17","cell_id":"c04","tier":"B0","cost_menu":"late_interaction_v1","route":"centroid_interaction","ranked_views":["centroid_interaction","pruned_late_interaction","exact_late_interaction"]}
```

The following would be invalid before purchasing the corresponding view:

```json
{"query_id":"q17","cell_id":"c04","tier":"B0","cost_menu":"late_interaction_v1","route":"exact_late_interaction","exact_maxsim_margin":0.31}
```

The invalid field is an unpaid exact late-interaction score. It is
evaluator-held until `exact_late_interaction` is bought.

## Accounting choices

Two accounting modes are possible and should be declared in the cost menu:

- Incremental stage cost: each stage is charged as the extra work after all
  parent stages already exist.
- Compiled route cost: a route such as `plaid_search` is charged as one
  bundled implementation that may include candidate generation, centroid
  pruning, and exact late interaction internally.

Both are legal Protocol B declarations as long as the cost profile is frozen
before scoring and the paid fields remain hidden until the matching route is
bought.

## Scope boundary

This note connects Protocol B to production late-interaction systems, but it is
not an official leaderboard slice in this release. A future scored slice would
need a frozen corpus, declared ColBERT/PLAID indexes, method-visible manifests,
view and cost menus, scorer ids, valid/invalid route fixtures, and public-dev
plus private-test splits under the same hidden-qrel policy.
