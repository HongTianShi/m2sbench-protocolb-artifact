# Repeated-Split Audit Index

This index makes the interval evidence inspectable without rerunning heavy audits. Some frozen reports ship per-repeat rows; others ship aggregate repeated-split summaries only because local scratch folds were not retained in the clean release.

| Report | Family | Repeats | Delta mean | 95% interval | Positive share | Per-repeat rows shipped | Rebuild script |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| standard-qrel dense 60/20/20 | validated dense stress | 30 | 0.007592746522277594 | [-0.0018159687519073483, 0.013371304422616958] | 0.9333333333333333 | yes | `scripts/run_protocol_b_standard_qrel_602020_audit.py` |
| dense learner-family sweep | supporting dense weak-signal learner sensitivity | 20 | model-specific; see dense_learner_family_sweep_summary.csv |  | model-specific | no; aggregate model summaries shipped | `scripts/run_learner_family_sweep.py --slice dense_semantic --config configs/learner_sweep_dense.yaml --out reports/dense_learner_family_sweep_summary.csv` |
| 2Wiki structured | validated core | 10 | 0.024014684058544133 | [0.018562255184816752, 0.029037799823964854] | 1.0 | no | `scripts/run_2wiki_structured_evidence_audit.py` |
| 2Wiki hyperlink 10k | validated structured stress | 10 | 0.029625693815554376 | [0.02557919883610413, 0.0336129854494959] | 1.0 | no | `scripts/run_2wiki_hyperlink_corpus_stress.py` |
| HotpotQA 5k | supporting generalization | 10 | 0.02257996531841693 | [0.011799607450076341, 0.029470211828680253] | 1.0 | no | `scripts/run_hotpot_structured_evidence_audit.py` |
| MuSiQue 10k | supporting generalization | 10 | 0.015294552519766524 | [0.01162034763819452, 0.018564970196528374] | 1.0 | no | `scripts/run_musique_structured_evidence_audit.py` |
| StrategyQA 2k | exploratory boundary | 10 | 0.003916861256832027 | [0.0005309166953813948, 0.007219175851027527] | 1.0 | no | `scripts/run_strategyqa_reasoning_evidence_audit.py` |
| FEVER 100 | stopped smoke boundary | 10 | -0.00042068385614807413 | [-0.030871449534814327, 0.033378526412828805] | 0.5 | no | `scripts/run_fever_claim_evidence_audit.py` |

Companion CSVs:

- `reports/protocol_b_standard_qrel_602020_repeated_split_rows.csv`: per-repeat rows for the standard-qrel dense 60/20/20 audit.
- `reports/dense_learner_family_sweep_summary.csv`: model-level legal learner sweep summaries, including negative neural/ranker rows.
- `reports/structured_repeat_summary.csv`: compact repeated-split summaries for 2Wiki core/stress plus supporting/boundary structured checks.
