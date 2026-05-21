# FEVER Claim-Evidence Report Pointer

Status: stopped-after-smoke boundary check only. This report is not part of the
official validated core and is not used as a main-paper validation row.

Purpose: exploratory claim-verification evidence-purchase slice over FEVER official JSONL. The artifact uses claim text plus title/sentence-id evidence handles available in the official file; it does not include wiki sentence text or full page text.

Canonical report:

- `reports/fever_claim_evidence_100_noce.md`
- `reports/fever_claim_evidence_100_noce.json`

Scope: FEVER is not part of the official leaderboard core and is not used as a main-paper validation row. It is retained as a final exploratory claim-evidence check because official evidence handles allow a narrow Protocol B readout, while lexical title shortcuts can be strong.

Canonical readout: on 100 verifiable rows, best fixed summary reaches 0.9883 utility, valid adaptive routing reaches 0.9869, and the oracle reaches 0.9955. Repeated adaptive-minus-fixed is -0.0004 with 95% interval [-0.0309,+0.0334] and positive share 0.500. Real title/evidence handles are compared against shuffled, wrong-evidence, random, and high-frequency-page controls; claim-only/title shortcuts remain the main risk signal.

Selected controls: claim-only 0.9883, real evidence sentence 0.9707, shuffled evidence sentence 0.1781, wrong evidence sentence 0.1464, same-count random 0.1884, high-frequency page 0.2091.
