# Structured Evidence Checks: What Is Core, Supporting, Or Boundary?

This map is a reviewer-facing guide to the structured-evidence experiments that
were run after the 2Wiki main slice. The paper keeps the validated core narrow:
dense semantic access plus 2Wiki structured evidence acquisition. HotpotQA and
MuSiQue are retained as supporting generalization checks. DocRED, StrategyQA,
and FEVER are retained as boundary checks that document when a cheap sketch or
fixed view already dominates.

Machine-readable supporting-check status cards are in
`manifests/supporting_structured_checks.json`. Dataset-source/split/control
cards for HotpotQA, MuSiQue, DocRED, StrategyQA, and FEVER are in
`docs/supporting_structured_data_cards.md`.

## Promotion rule used in this artifact

| Status | Rule of thumb | Paper role |
| --- | --- | --- |
| Validated core | Official menu or paper-facing robustness row; real evidence beats controls; legal adaptive routing improves over the best fixed view with positive repeated-split stability; oracle headroom remains. | Main paper table or compact paragraph. |
| Supporting generalization | Same Protocol B readout repeats on a separate public family, but the slice is not promoted to a separate leaderboard core. | One-sentence supporting evidence; full report in artifact. |
| Boundary check | Controls are informative, but the best fixed cheap/co-mention/title view already dominates or legal routing does not improve. | Artifact-only; documents limits and guards against cherry-picking. |

## Structured-evidence map

| Slice | Status | Canonical report(s) | Main readout |
| --- | --- | --- | --- |
| 2Wiki provided-context structured evidence | Validated core | `reports/2wiki_structured_evidence_2000.md`, `reports/2wiki_structured_evidence.md` | Clean structured-evidence slice: title sketches, 1-hop/2-hop/full context, and CE are declared views; support facts and evidence triples are evaluator-held. |
| 2Wiki 7GB hyperlink-corpus stress | Validated robustness row | `reports/2wiki_hyperlink_corpus_stress_10000.md`, `reports/2wiki_7gb_hyperlink_stress.md` | Query-local hyperlink expansion preserves the pattern at 10k rows: best fixed/adaptive/oracle utilities are `.7910/.8188/.8865`, repeated adaptive-minus-fixed is `+.0296` with interval `[+.0256,+.0336]`, and real hyperlink views beat shuffled/random controls. |
| HotpotQA distractor contexts | Supporting generalization | `reports/hotpot_structured_evidence_5000_noce.md`, `reports/hotpot_structured_evidence.md` | A separate multi-hop QA family repeats the pattern without becoming a third validated core: best fixed/adaptive/oracle utilities `.7855/.8057/.9024`, repeated adaptive-minus-fixed `+.0226` with interval `[+.0118,+.0295]`, all repeats positive, and random/shuffled controls trail real evidence. |
| MuSiQue decomposition/paragraph evidence | Supporting generalization | `reports/musique_structured_evidence_10000_noce.md`, `reports/musique_structured_evidence.md` | Decomposition-centric replication: best fixed/adaptive/oracle utilities `.5629/.5819/.6826`, repeated adaptive-minus-fixed `+.0153` with interval `[+.0116,+.0186]`, all repeats positive, and real paragraph evidence beats shuffled/random paragraph controls. |
| DocRED relation evidence | Boundary/candidate check | `reports/docred_structured_evidence.md`, `reports/docred_structured_evidence_noce.md`, `reports/docred_structured_evidence_ce.md` | Exact evidence annotations make this useful, but fixed co-mention evidence is already strong (`.9304` utility) and legal routers do not improve over it (`mean learned-minus-best-fixed = 0.0`). |
| StrategyQA decomposition/facts | Boundary/exploratory check | `reports/strategyqa_reasoning_evidence_2000_noce.md`, `reports/strategyqa_reasoning_evidence.md` | Shuffled/cross-question/random facts collapse, but the released decomposition sketch is already highly informative. Best fixed/adaptive/oracle utilities `.9067/.9093/.9503`; repeated adaptive-minus-fixed `+.0039` with interval `[+.0005,+.0072]`. |
| FEVER official claim/evidence handles | Stopped-after-smoke boundary check | `reports/fever_claim_evidence_100_noce.md`, `reports/fever_claim_evidence.md` | Official JSONL has title/sentence-id evidence handles but not page text. Claim-only title matching already reaches `.9883` utility, adaptive routing is slightly below fixed summary (`.9869`), and repeated adaptive-minus-fixed is `-.0004` with interval `[-.0309,+.0334]`. |

## How to read these results

The supporting and boundary rows should not be read as extra benchmark cores.
They answer two review questions that are adjacent to, but smaller than, the
main contribution:

1. Does the 2Wiki structured-evidence result appear outside one dataset family?
   HotpotQA and MuSiQue say yes under the same target-blind evidence-purchase
   readout.
2. Does Protocol B report failures or cheap-view dominance rather than forcing
   every dataset into a positive story? DocRED, StrategyQA, and FEVER say yes:
   when a fixed cheap/co-mention/title view dominates, the artifact records that
   boundary instead of promoting it to the paper.

The paper therefore uses dense semantic access and 2Wiki as the validated core,
mentions HotpotQA/MuSiQue only as supporting generalization checks, and leaves
DocRED/StrategyQA/FEVER in the artifact as scoped boundary records.

Small-row smoke caveat: `reports/2wiki_hyperlink_corpus_stress_50.md` is kept
only as a quick handle check and is unstable/negative. The paper-facing
hyperlink readout uses the 2k/5k/10k reports, with 10k as the main robustness
row.
