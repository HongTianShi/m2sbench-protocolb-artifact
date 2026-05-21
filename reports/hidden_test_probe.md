# Hidden-Test and Anti-Probing Policy Note

This note records the paper-facing hidden-test policy and points to the supporting route-stability reports.

## Policy

Private-test rows reuse the same Protocol B route schema plus declared view-menu and cost-menu ids as public-dev rows. Evaluator-held qrels, support facts, source labels, CE scores, hidden targets, and refreshed seeds remain withheld. Feedback is aggregate-only and capped by leaderboard/menu version; the archival score is computed from one final deterministic JSONL file per solver.

## Supporting Records

- `reports/protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.md`
- `reports/ir_router_significance.md`
- `reports/evidence_cascade_router_audit.md`
- `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`
- `docs/submission_contract.md`
- `schema/route_schema.json`

## Public/Private Probe Table

The full standard-qrel report above stores the source rows. This compact table
is repeated here so the hidden-test policy note is self-contained.

| submission cap | public-dev utility | private-test utility | overfit gap |
| ---: | ---: | ---: | ---: |
| 1 | 0.1321 | 0.1389 | -0.0069 [-0.0282,0.0110] |
| 3 | 0.3465 | 0.3495 | -0.0029 [-0.0318,0.0255] |
| 10 | 0.3542 | 0.3527 | 0.0015 [-0.0303,0.0336] |

Mean rank correlation between public-dev and private-test candidate utilities:
`0.624`.

## Readout

The standard-qrel records include repeated raw 60/20/20 splits and leave-one-dataset-out stress. These are not a guarantee that public-dev overfitting is impossible; they are a governance check showing that hidden-test rows can reuse the same schema while exposing overfitting and transfer failures rather than silently folding them into a single public score.
