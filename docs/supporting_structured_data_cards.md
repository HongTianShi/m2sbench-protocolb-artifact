# Supporting Structured-Family Data Cards

These cards cover structured-evidence checks that support or bound the 2Wiki
validated core. They are not promoted to separate official leaderboard cores.

## HotpotQA Distractor

- Status: supporting generalization check.
- Source: HuggingFace-datasets `load_from_disk` export of HotpotQA distractor
  contexts prepared outside the release package.
- Rebuild handle: set `M2SBENCH_HOTPOT_DIR` to the local `load_from_disk`
  export directory and run `scripts/run_hotpot_structured_evidence_audit.py`.
- Upstream split used in frozen reports: `train`.
- Internal route split: repeated train/dev/test routing splits recorded in the
  report; this is not a separate official hidden benchmark split.
- Method-visible sketch: question, title/paragraph sketches, and declared
  evidence-menu metadata.
- Evaluator-held fields: supporting fact titles/sentence ids, answer, full
  evidence scores, oracle route.
- Controls: shuffled supporting titles, random paragraphs/sentences, and
  same-count random evidence where available.

## MuSiQue

- Status: supporting generalization check.
- Source: HuggingFace-datasets `load_from_disk` export of MuSiQue-style
  decomposition/paragraph records prepared outside the release package.
- Rebuild handle: set `M2SBENCH_MUSIQUE_DIR` to the local `load_from_disk`
  export directory and run `scripts/run_musique_structured_evidence_audit.py`.
- Upstream split used in frozen reports: `train`.
- Internal route split: repeated train/dev/test routing splits recorded in the
  report.
- Method-visible sketch: question, decomposition-question/title sketch, coarse
  paragraph statistics.
- Evaluator-held fields: supporting paragraph labels, answer, decomposition
  answers, full evidence scores, oracle route.
- Controls: shuffled paragraph mapping, same-count random paragraphs, and
  decomposition-shuffled variants.

## DocRED

- Status: boundary/candidate check.
- Source: public DocRED-style relation/evidence records used by existing
  artifact scripts.
- Method-visible sketch: entity-pair/co-mention summary and declared evidence
  menu.
- Evaluator-held fields: relation label, gold evidence sentence ids, full/CE
  scores, oracle route.
- Readout: fixed co-mention evidence is already strong, so legal routers do not
  improve over the best fixed view. This is retained as a boundary condition.

## StrategyQA

- Status: exploratory reasoning-frontier check.
- Source: HuggingFace-datasets `load_from_disk` export of StrategyQA
  question/decomposition/fact records prepared outside the release package.
- Rebuild handle: set `M2SBENCH_STRATEGYQA_DIR` to the local `load_from_disk`
  export directory and run `scripts/run_strategyqa_reasoning_evidence_audit.py`.
- Method-visible sketch: question, term/description/decomposition summary.
- Evaluator-held fields: supporting facts, answer, oracle route.
- Readout: shuffled/random facts collapse, but the cheap decomposition sketch is
  already strong. This is retained as a scoped boundary record.

## FEVER

- Status: stopped-after-smoke boundary check.
- Source: official FEVER train JSONL.
- Rebuild handle: set `M2SBENCH_FEVER_TRAIN_JSONL` to the official train JSONL
  file and run `scripts/run_fever_claim_evidence_audit.py`.
- Method-visible sketch: claim text and cheap title/entity overlap statistics.
- Evaluator-held fields: evidence sentence handles, label, answer-equivalent
  verification target, oracle route.
- Readout: claim/title shortcuts dominate in the smoke audit; adaptive routing
  does not improve over fixed summary, so this is not promoted.
