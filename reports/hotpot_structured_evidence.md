# HotpotQA Structured Evidence Acquisition Report Pointer

Canonical reports:

- `reports/hotpot_structured_evidence_5000_noce.md`
- `reports/hotpot_structured_evidence_5000_noce.json`
- `reports/hotpot_structured_evidence_2000_noce.md`
- `reports/hotpot_structured_evidence_2000_noce.json`
- `reports/hotpot_structured_evidence_2000_ce.md`
- `reports/hotpot_structured_evidence_2000_ce.json`

Purpose: supporting structured-family check over HotpotQA distractor contexts. A method sees the question and context-title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.

Menu status: this is a supporting check, not an official menu-backed leaderboard
slice. The machine-readable status card is
`manifests/supporting_structured_checks.json`; no `menus/hotpot_*.json` official
menu is declared in this release.

Readout: the 5k no-CE run is the preferred supporting check. Best fixed paragraph utility is `.7855`, the best legal adaptive router reaches `.8057`, and the oracle reaches `.9024`. Across 10 repeated 60/20/20 splits, learned-minus-best-fixed is `+.0226` with CI `[+.0118,+.0295]` and positive share `1.0`. Real paragraph/sentence evidence strongly beats shuffled-title, random-paragraph, and random-sentence controls. The optional 2k CE run shows that CE has oracle share (`.1075`) but fixed CE overpays and a single displayed split is not a clean adaptive win; it is retained as an expensive-view boundary record rather than a headline claim.

Scope: HotpotQA is not added as a separate official leaderboard slice in the current paper. It is retained to show that the 2Wiki-style structured evidence-purchase contract can be instantiated on a second public multi-hop QA family with hidden supporting facts and explicit randomization controls.
