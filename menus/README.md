# Official Protocol B Menus

This directory contains the paper-facing menu declarations used by the
Protocol B route validator. A reviewer can inspect these files without running
heavy reproduction:

- `dense_semantic.view_menu.json` and `dense_semantic.cost_menu.json` declare
  the dense semantic access menu used by the FiQA joint dense audit.
- `dense_semantic.C_mem.cost_menu.json` and
  `dense_semantic.C_lat.cost_menu.json` are secondary dense rescoring profiles
  derived from `reports/unified_systems_profile_audit.json`; they are
  diagnostic profile declarations, not deployment-cost claims.
- `2wiki_structured.view_menu.json` and `2wiki_structured.cost_menu.json`
  declare the provided-context 2Wiki structured evidence slice.
- `2wiki_hyperlink.view_menu.json` and `2wiki_hyperlink.cost_menu.json`
  declare the 7GB hyperlink-corpus stress slice.

HotpotQA and MuSiQue are supporting structured-evidence checks rather than
official menu-backed leaderboard slices. Their machine-readable status cards
are in `manifests/supporting_structured_checks.json`.

The JSONL submission schema stays the same across slices. New evidence views
are added by declaring a view menu, a cost menu, and a frozen scorer id, then
validating route rows with `scripts/validate_protocol_b_route.py`.

Each official view entry is intentionally declarative:

- `visible_fields`: fields a method may use before buying the view.
- `paid_fields`: fields materialized only after the view is bought.
- `forbidden_before_purchase`: fields rejected if they appear in a submitted
  row before purchase, including support facts, qrels, answers, unpaid CE/full
  scores, and oracle utilities.
- `parents`: dependency views used by cumulative-cost profiles.

The schema accepts both `hidden_evaluator_fields` and the backward-compatible
alias `evaluator_only_fields`; new menu files should include both with matching
values.

For cost provenance, see `docs/cost_profile_derivation.md`. In secondary dense
profiles, `declared_cost` is the active C_mem or C_lat cost used by that menu;
`incremental_cost` and `cumulative_cost` retain the original C_op ladder values
for traceability.
