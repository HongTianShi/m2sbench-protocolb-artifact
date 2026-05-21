# Evaluator-Held Public-Dev References

These `*.reference.jsonl` files are included only so reviewers can run the
local public-dev scorer without a private server. They contain evaluator-held
view scores and oracle diagnostics for small deterministic review fixtures.

Solver code should read the sibling `public_dev/*.manifest.jsonl` files, menus,
and documentation, but should not train on or import these reference rows. In a
private-test leaderboard, analogous reference rows are withheld and only
aggregate leaderboard feedback is returned.

The file `2wiki_synthetic_raw_label_toy.json` is a one-row illustrative example
showing how support facts/triples/answers conceptually induce view-score
fixtures. It is synthetic, not a public-dev training row, and is not consumed by
the scorer.
