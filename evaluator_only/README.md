# Evaluator-Only Reference

`hidden_reference.jsonl` is a public-development scoring fixture for the small
controlled-cell demo. It contains canonical targets and oracle-view metadata used
by the public evaluator, and solver code must not import or read this directory.
The runner passes only `challenge_cells.jsonl` records to `solve(cell)`.

For the paper-facing Protocol B dense and 2Wiki slices, evaluator-held fields are
declared in `menus/*.view_menu.json`: qrels, support facts, evidence triples,
answers, hidden targets, unpaid full/CE scores, oracle views, and view utilities
are never method-visible before the matching view is purchased. Private-test
versions withhold the same fields under the same route schema.
