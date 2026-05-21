# Protocol B Public-Dev Submission Packs

This directory is the lightweight public-development evaluator path for the
paper-facing Protocol B contract. It is separate from the small controlled-cell
demo at the repository root.

Each official pack has:

- `<slice>.manifest.jsonl`: method-visible rows with `query_id`, `cell_id`,
  `tier`, `cost_menu`, declared views, and cheap visible feature sketches.
- `evaluator_only/<slice>.reference.jsonl`: evaluator-held public-dev
  references used by the scorer. Solver code must not read these files;
  private-test scoring withholds the analogous reference rows.
- `submission_template/protocol_b_public_dev_<slice>_*.jsonl`: example route
  submissions that can be scored immediately.

The public-dev packs are intentionally small deterministic review fixtures.
They demonstrate the exact submit/validate/score/leaderboard loop used by the
contract; they are not the large paper measurements. Dense and provided-context
2Wiki mirror the validated core slices, while `2wiki_hyperlink` is a
structured-stress fixture. The larger paper metrics are the frozen reports under
`reports/`, and private leaderboard rows can reuse the same schema with withheld
references or refreshed seeds/corpora.

Important: this directory is a local scoring fixture, not a participant training
set for paper claims. `*.manifest.jsonl` files are method-visible inputs.
`evaluator_only/*.reference.jsonl` files are evaluator-held public-dev
answers/scores included only so reviewers can run the scorer locally. Do not
train solvers on reference rows, per-row oracle diagnostics, or generated
scorer outputs.

Feature meanings are centralized in
`docs/public_dev_feature_dictionary.md`. The evaluator-held reference files are
documented in `public_dev/DO_NOT_TRAIN_ON_REFERENCE.md`; they are shipped only
for local scoring and should not be used to fit solvers.

## Dense public-dev scorer

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_dense_semantic_router.jsonl \
  --manifest public_dev/dense_semantic.manifest.jsonl \
  --reference public_dev/evaluator_only/dense_semantic.reference.jsonl \
  --view-menu menus/dense_semantic.view_menu.json \
  --cost-menu menus/dense_semantic.cost_menu.json
```

## 2Wiki structured public-dev scorer

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl \
  --manifest public_dev/2wiki_structured.manifest.jsonl \
  --reference public_dev/evaluator_only/2wiki_structured.reference.jsonl \
  --view-menu menus/2wiki_structured.view_menu.json \
  --cost-menu menus/2wiki_structured.cost_menu.json
```

## 2Wiki hyperlink public-dev scorer

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl \
  --manifest public_dev/2wiki_hyperlink.manifest.jsonl \
  --reference public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl \
  --view-menu menus/2wiki_hyperlink.view_menu.json \
  --cost-menu menus/2wiki_hyperlink.cost_menu.json
```

The scorer writes per-row metrics, a JSON summary, and a one-row leaderboard
under the system temp directory unless `--out-dir` or explicit output paths are
supplied. Per-row oracle fields are public-dev diagnostics only; private-test
scoring returns aggregate leaderboard rows without evaluator-held labels,
support facts, paid scores, or oracle views.
