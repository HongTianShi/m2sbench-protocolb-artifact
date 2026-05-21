# Legal Router Feature Tiers

This note summarizes which features are available to reference routers before a
view is bought. The routers in the paper and reports are reference solvers, not
proposed methods. Their role is to show whether method-visible fields contain
usable but incomplete information about view value.

| Slice | Tier | Method-visible examples | Forbidden before purchase |
| --- | --- | --- | --- |
| Dense semantic access | B0 score/stat sketch | Query/cell ids, declared view and cost menus, summary score statistics, list-size and coarse score-shape features. | Qrels, relevance labels, full-vector scores, CE scores, oracle view, per-view utility. |
| Dense semantic access | B1 agreement/profile sketch | Released cheap-view agreement, margin/entropy/slope features, declared profile fields. | Any unpaid full/CE score or hidden relevance field. |
| 2Wiki structured evidence | B0 title/entity sketch | Question text, provided context titles, coarse title/entity overlap and released list statistics. | Supporting facts, evidence triples, answer, support-title labels, unpaid CE scores. |
| 2Wiki structured evidence | B1 graph/path/context sketch | Released degree/path summaries, title/entity agreement, shallow graph/path metadata, context-sketch statistics declared in the manifest. | Gold support facts/triples/answers, full context text unless bought, hidden oracle view. |
| 2Wiki hyperlink stress | B1 hyperlink sketch | Query-local title overlap, graph agreement, path-sketch signal, question type, declared local-pool menu. | Gold support titles/facts, answer, hidden evaluator scores, full local pool unless bought. |
| HotpotQA / MuSiQue supporting checks | B0/B1 structured sketches | Question, title/decomposition sketches, released paragraph/sentence/path metadata used by the report. | Support labels, answers, decomposition answers, unpaid full/CE scores. |
| DocRED / StrategyQA / FEVER boundary checks | B0/B1 exploratory sketches | Entity/co-mention handles, decomposition sketches, claim/title handles, or fact metadata declared by each report. | Gold evidence ids, labels/answers, unpaid evidence scores, hidden support facts. |

Feature tiers are intentionally cheap and auditable. If a future participant
adds a new feature family, it must be declared in the view menu or manifest,
assigned to a visibility tier, and checked against evaluator-held fields before
it can be used in a leaderboard route.
