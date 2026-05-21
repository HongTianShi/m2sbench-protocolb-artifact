# Protocol B Submission Contract

Protocol B evaluates an evidence-purchase decision. A solver first sees a released state and a declared menu of evidence views. It then chooses which view to buy before evaluator-held qrels, support facts, targets, or unpaid reranker scores are applied.

## Required JSONL Fields

Each submitted row must contain:

- `query_id`: opaque query id from the released manifest.
- `cell_id`: opaque cell or candidate-pool id.
- `tier`: visibility tier, such as `B0`, `B1`, `B2`, or a slice-specific tier such as `B1_hyperlink`.
- `cost_menu`: declared cost-menu/profile version.
- One primary action: `route`, or `ranked_views` when `route` is absent.
  `view_set` and `abstain` are schema extensions and are accepted only when
  the slice menu declares a non-oracle set or abstention rule.

The main leaderboard scores one qrel-blind primary action. If `route` is present, the evaluator charges and scores that view. If `route` is absent and `ranked_views` is present, the first valid ranked view is used. If both `route` and `ranked_views` are present, `route` is the scored action and `ranked_views` is a diagnostic ranking/tie-break field. `view_set` is never resolved by a hidden oracle choice; current official menus reject it for leaderboard scoring unless a future menu declares a deterministic set rule and set cost. `abstain` is likewise rejected unless the menu declares an abstention cost and utility.

## Official Menu Files

The paper-facing Protocol B slices are declared by machine-readable menu files:

- `menus/dense_semantic.view_menu.json` with `menus/dense_semantic.cost_menu.json`
- `menus/2wiki_structured.view_menu.json` with `menus/2wiki_structured.cost_menu.json`
- `menus/2wiki_hyperlink.view_menu.json` with `menus/2wiki_hyperlink.cost_menu.json`

These files list declared views, allowed visibility tiers, hidden evaluator fields, frozen scorer ids, costs, and CE accounting context. The root-level `submission_schema.json` and controlled-cell demo files remain only for the small public demo; the `schema/` and `menus/` directories are the paper-facing contract.

For a compact review map, see `docs/official_benchmark_contract.md`. It marks
dense semantic access and 2Wiki structured evidence acquisition as validated
core slices, marks 2Wiki 10k hyperlink expansion as the canonical structured
stress audit, and separates IVF-PQ scale stress, controlled diagnostics, and
portability adapters from the main validation claim.

Each official view-menu row declares:

- `visible_fields`: fields available before purchase.
- `paid_fields`: fields materialized only when the view is bought.
- `forbidden_before_purchase`: fields rejected if they appear in a submitted row
  before the matching view is bought.
- `parents`: dependency views for cumulative cost profiles.

The menu-level `hidden_evaluator_fields` is the canonical name for fields held
by the evaluator. `evaluator_only_fields` is retained as a backward-compatible
alias and has the same values in official menus.

The lightweight `public_dev/` manifests store released sketches compactly under
`visible_features` with a `declared_views` list. The corresponding
`public_dev_visible_fields` entry in each view menu records this fixture shape;
the full semantic names for paid and evaluator-held fields remain in the view
rows.

## Lightweight Reviewer Validation

The route validator is intentionally standard-library only and does not run the heavy experiments:

```bash
python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_dense.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json \
  --manifest manifests/dense_semantic.dev_manifest.jsonl

python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_2wiki_structured.jsonl \
  --view-menu menus/2wiki_structured.view_menu.json \
  --cost-menu menus/2wiki_structured.cost_menu.json \
  --manifest manifests/2wiki_structured.dev_manifest.jsonl

python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_valid_2wiki_hyperlink.jsonl \
  --view-menu menus/2wiki_hyperlink.view_menu.json \
  --cost-menu menus/2wiki_hyperlink.cost_menu.json \
  --manifest manifests/2wiki_hyperlink.dev_manifest.jsonl
```

The validator checks required fields, exact one-primary-action semantics, declared view membership, view-menu/cost-menu internal consistency, cost-menu id consistency, visibility-tier membership, duplicate views, optional manifest membership, and recursive leakage keys inside `diagnostics`.

## Public-Dev Scoring

Future methods can also exercise the paper-facing public-dev scoring path
without rerunning the heavy audits. The scorer consumes arbitrary valid route
JSONL over a released public-dev manifest and joins evaluator-held reference
rows only inside the scoring script:

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_dense_semantic_router.jsonl \
  --manifest public_dev/dense_semantic.manifest.jsonl \
  --reference public_dev/evaluator_only/dense_semantic.reference.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json
```

Analogous packs are provided for `2wiki_structured`. A smaller
`2wiki_hyperlink` pack is included as a structured-stress review fixture; the
paper-facing 10k hyperlink result remains the frozen report under `reports/`.
The scorer writes per-row raw quality, charged cost, utility, regret, oracle
headroom, and a leaderboard row under the system temp directory unless
`--out-dir` or explicit output paths are supplied. Private-test operation uses
the same route schema and menu ids while withholding the reference file and
returning only aggregate scores. Per-row oracle fields emitted by this script
are public-dev diagnostics only and are never returned for private-test rows.

## Valid Row

```json
{
  "query_id": "q17",
  "cell_id": "c04",
  "tier": "B0",
  "cost_menu": "dense-semantic-v1-op",
  "ranked_views": ["int8", "full"],
  "route": "int8"
}
```

## Invalid Row

```json
{
  "query_id": "q17",
  "cell_id": "c04",
  "tier": "B0",
  "cost_menu": "dense-semantic-v1-op",
  "route": "ce",
  "ce_score_margin": 0.31
}
```

This row is invalid because `ce_score_margin` is evaluator-only before the CE view is bought.

The same validator rejects the example:

```bash
python scripts/validate_protocol_b_route.py \
  submission_template/protocol_b_invalid_leak.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json
```

## Visibility Tiers

- `B0`: released summary and declared view/cost menu only.
- `B1`: declared low-cost context, such as shallow agreement, cache statistics, same-cell history, title/entity sketches, or a released context graph.
- `B2`: explicitly purchased structural evidence.

Hidden qrels, support facts, labels, source metadata, full vectors, CE scores, hidden targets, and undeclared logs are evaluator-only until the matching view is bought.

## Private-Test Policy

Private-test rows reuse the same route schema plus declared view-menu and cost-menu ids, but evaluator-held qrels, support facts, source labels, CE scores, or refreshed seeds are withheld. Feedback is aggregate-only and capped by leaderboard/menu version. A valid solver must be deterministic under the submitted JSONL and must not use hidden-test feedback for post-hoc reranking.

Operational policy for a leaderboard version:

- This release's public-dev manifests expose no qrels, answers, support facts,
  CE scores, full-view scores, or oracle utilities. Future debug-only
  public-dev releases may explicitly expose public labels, but such fields must
  be declared outside the official hidden-test route contract.
- A solver version is the tuple of code/config/checkpoint plus the submitted JSONL. Changing any component starts a new submission.
- Submission feedback is aggregate-only. Per-row hidden-test labels, CE scores, support facts, and oracle views are never returned.
- Submission caps are versioned with the leaderboard; this review policy allows at most three private-test submissions per solver and menu version after public-dev tuning. Repeated probing beyond the cap is invalid.
- Maintainers may refresh seeds, qrels, or corpus slices while preserving the route schema, view-menu ids, cost-menu ids, and scorer contracts.

## External Tools

LLM/tool-using solvers are allowed as implementation routes over method-visible fields and explicitly purchased views. Web search, cached qrels, memorized hidden labels, unpaid CE/full scores, and undeclared external logs are forbidden.

Tool outputs must be serialized only through the submitted route fields. A tool may summarize released title sketches or purchased context, but it may not import `evaluator_only/`, support-fact labels, public-dev answers for hidden-test rows, or undeclared local corpora.

## Schemas

- `schema/route_schema.json`
- `schema/view_menu_schema.json`
- `schema/cost_menu_schema.json`
- `menus/*.view_menu.json`
- `menus/*.cost_menu.json`
- `scripts/validate_protocol_b_route.py`
- `scripts/score_protocol_b_public_dev.py`
- `public_dev/*.manifest.jsonl`
- `public_dev/evaluator_only/*.reference.jsonl`

The older root-level `submission_schema.json` is retained for the small controlled-cell demo. The `schema/` directory is the paper-facing Protocol B route contract used by the dense and 2Wiki evidence-access reports.
