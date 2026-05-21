# Menu Integrity Report

This static report records cross-file menu checks that plain JSON Schema does
not express by itself. The command-line validator enforces the same route/menu
membership checks when a submission is scored.

| View menu | Accepted cost menus | Row-level declared views | Cost-key coverage |
| --- | --- | --- | --- |
| `menus/dense_semantic.view_menu.json` | `dense-semantic-v1-op`, `dense-semantic-v1-mem`, `dense-semantic-v1-lat` | Enforced from `manifest_row.declared_views` when a manifest is supplied. | All declared dense views have matching `view_costs` entries in each accepted dense cost menu. |
| `menus/2wiki_structured.view_menu.json` | `2wiki-structured-v1-op` | Enforced from `manifest_row.declared_views` when a manifest is supplied. | All declared 2Wiki structured views have matching `view_costs` entries. |
| `menus/2wiki_hyperlink.view_menu.json` | `2wiki-hyper-v1-op` | Enforced from `manifest_row.declared_views` when a manifest is supplied. | All declared hyperlink-stress views have matching `view_costs` entries. |

Hidden/evaluator-only aliases are intentionally duplicated as
`hidden_evaluator_fields` and `evaluator_only_fields` in view menus so both the
paper terminology and older validator wording resolve to the same forbidden
field set. The validator also recursively scans nested diagnostics for those
keys, not only top-level route fields.

Secondary dense profiles use the same view menu and set `declared_cost` as the
active scoring field. `incremental_cost` and `cumulative_cost` are retained as
trace fields so reviewers can see the original `C_op` ladder after rescoring
under `C_mem` or `C_lat`.
