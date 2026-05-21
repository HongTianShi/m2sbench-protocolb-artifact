# Reports Directory Map

This directory contains frozen aggregate reports and supporting JSON/CSV records.
For review, start with the files below instead of browsing the full directory.

## Canonical Paper-Facing Reports

| Role | File |
| --- | --- |
| Paper claim to evidence map | `PAPER_EVIDENCE_INDEX.md` |
| Structured evidence core/supporting/boundary map | `structured_evidence_check_map.md` |
| Statistical/repeated-split protocol card | `statistical_protocol.md` |
| Repeated-split audit index and CSV pointers | `repeated_split_audit_index.md` |
| Hash manifest for headline reports and contract files | `REPORT_MANIFEST_SHA256.md` |
| Dense joint menu pointer | `dense_joint_menu.md` |
| Dense joint menu canonical report | `joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md` |
| 2Wiki provided-context structured evidence | `2wiki_structured_evidence_2000.md` |
| 2Wiki 7GB hyperlink stress pointer | `2wiki_7gb_hyperlink_stress.md` |
| 2Wiki 7GB hyperlink stress canonical 10k report | `2wiki_hyperlink_corpus_stress_10000.md` |
| 2Wiki lambda sensitivity | `2wiki_lambda_sensitivity.md` |
| CE purchase / calibration pointer | `ce_purchase.md` |
| CE decision curve audit | `ce_decision_curve_audit.md` |
| Cost-profile pointer | `cost_profiles.md` |
| Unified systems profile audit | `unified_systems_profile_audit.md` |
| Hidden-test / anti-probing policy note | `hidden_test_probe.md` |

## Supporting Structured-Evidence Checks

These are supporting generalization or boundary checks, not additional promoted
official leaderboard cores.

| Status | File |
| --- | --- |
| Supporting HotpotQA check | `hotpot_structured_evidence.md` |
| HotpotQA 5k no-CE robustness | `hotpot_structured_evidence_5000_noce.md` |
| Supporting MuSiQue check | `musique_structured_evidence.md` |
| MuSiQue 10k no-CE robustness | `musique_structured_evidence_10000_noce.md` |
| Boundary DocRED check | `docred_structured_evidence.md` |
| Boundary StrategyQA check | `strategyqa_reasoning_evidence.md` |
| Boundary FEVER check | `fever_claim_evidence.md` |

## Smoke/Internal Consistency Reports

Reports with smaller row counts such as 50, 100, 1000, or 2000 are retained as
smoke, consistency, or intermediate robustness records. The paper-facing
structured stress readout uses the 2Wiki 10k hyperlink report, while 2k/5k
records are artifact consistency checks.

## What Not To Do

Do not train solvers from report JSON/CSV files. Frozen reports contain
evaluator-held scores, oracle rows, controls, or aggregate diagnostics. Legal
route policies should use only method-visible manifests and menu declarations.
