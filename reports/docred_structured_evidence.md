# DocRED Structured Evidence Candidate Check

Status: boundary/candidate check only. This report is not part of the official
validated core and is not used as a main-paper validation row.

Canonical frozen reports:

- `reports/docred_structured_evidence_noce.md`
- `reports/docred_structured_evidence_noce.json`
- `reports/docred_structured_evidence_ce.md`
- `reports/docred_structured_evidence_ce.json`

Purpose: supporting candidate check for a second structured-evidence family over DocRED relation evidence annotations. The slice treats entity/co-mention sketches, candidate evidence sentences, full document context, and optional CE reranking as declared views, while gold evidence sentence ids, relation labels used for scoring, and unpaid full/CE scores remain evaluator-held.

Readout: real entity/co-mention evidence strongly beats shuffled co-mentions, degree-random co-mentions, and same-document random sentence controls. However, the best fixed co-mention view is already very strong (`.9304` utility), and legal B0/B1 routers do not improve over it in either the no-CE or CE menu (`mean learned-minus-best-fixed = 0.0`). The CE view improves neither fixed utility nor legal routing under the declared cost. For that reason this report is retained as a supporting/negative structured-evidence check and is not used as a main-paper validation row.
