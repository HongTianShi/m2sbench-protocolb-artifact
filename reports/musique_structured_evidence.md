# MuSiQue Structured Evidence Acquisition Report Pointer

Purpose: supporting structured-family replication over MuSiQue paragraphs. A
method sees the question, decomposition-question text, paragraph-title sketch,
and coarse paragraph metadata, then may buy paragraph, multi-paragraph
support-set, full-context, or optional CE evidence before evaluator-held support
labels are scored. This is evidence acquisition, not answer generation.

Menu status: this is a supporting check, not an official menu-backed leaderboard
slice. The machine-readable status card is
`manifests/supporting_structured_checks.json`; no `menus/musique_*.json`
official menu is declared in this release.

Canonical report:

- `reports/musique_structured_evidence_10000_noce.md`
- `reports/musique_structured_evidence_10000_noce.json`

Consistency reports:

- `reports/musique_structured_evidence_5000_noce.md`
- `reports/musique_structured_evidence_5000_noce.json`
- `reports/musique_structured_evidence_2000_noce.md`
- `reports/musique_structured_evidence_2000_noce.json`
- `reports/musique_structured_evidence_100_noce.md`
- `reports/musique_structured_evidence_100_noce.json`

Scope: MuSiQue is not added as a separate official leaderboard slice in the
current paper. It is retained to check whether the structured evidence-purchase
pattern observed on 2Wiki and HotpotQA repeats on a decomposition-centric
multi-hop QA family with hidden support labels and explicit shuffled/random
controls.

Canonical readout: on 10k rows, best fixed paragraph evidence reaches .5629
utility, valid adaptive routing reaches .5819, and the oracle reaches .6826.
Repeated adaptive-minus-fixed is +.0153 with 95% interval [.0116,.0186] and all
10 repeats positive. Real paragraph evidence (.5629 utility) is far above
shuffled and random paragraph controls (.1466--.1564 utility), while oracle
headroom remains.
