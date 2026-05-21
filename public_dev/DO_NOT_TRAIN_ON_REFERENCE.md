# Do Not Train On Public-Dev References

`public_dev/evaluator_only/*.reference.jsonl` files are evaluator-held public-dev references.
They are shipped only so reviewers can run a local submit/score loop without a
private server.

For a legal Protocol B solver, use only:

- `public_dev/*.manifest.jsonl`
- declared `menus/*.view_menu.json`
- declared `menus/*.cost_menu.json`
- public documentation under `docs/`

Do not train on or import:

- `public_dev/evaluator_only/*.reference.jsonl`
- support facts, answers, qrels, CE/full scores, oracle utilities, or best-view
  labels from frozen reports
- `evaluator_only/`

Private-test scoring withholds the analogous reference rows and returns only
aggregate leaderboard feedback. Public-dev references are for scorer
demonstration and reviewer audit, not for fitting route policies.
