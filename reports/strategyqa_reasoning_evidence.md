# StrategyQA Reasoning Evidence Report Pointer

Status: boundary/exploratory check only. This report is not part of the
official validated core and is not used as a main-paper validation row.

Purpose: exploratory reasoning-evidence slice over StrategyQA. A method sees the
question, term, description, and decomposition sketch, then may buy fact
snippets or a full fact set before evaluator-held supporting facts are scored.
This is evidence acquisition, not answer generation.

Canonical report:

- `reports/strategyqa_reasoning_evidence_2000_noce.md`
- `reports/strategyqa_reasoning_evidence_2000_noce.json`

Smoke report:

- `reports/strategyqa_reasoning_evidence_100_noce.md`
- `reports/strategyqa_reasoning_evidence_100_noce.json`

Scope: StrategyQA is not part of the official leaderboard core and is not used
as a main-paper validation row. It is retained as a noisy reasoning-frontier
check because the released decomposition sketch is often already highly
informative.

Canonical readout: on 2k rows, best fixed decomposition evidence reaches .9067
utility, valid adaptive routing reaches .9093, and the oracle reaches .9503.
Repeated adaptive-minus-fixed is +.0039 with 95% interval [.0005,.0072] and all
10 repeats positive. Shuffled/cross-question/random fact controls collapse
(.1249--.1374 utility), but decomposition-only remains strong, so the result is
best read as an exploratory boundary case rather than a core structured-evidence
claim.
