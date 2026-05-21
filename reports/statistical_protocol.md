# Statistical Protocol Card

This card records how intervals and repeated-split summaries in the frozen
reports should be read during static artifact review.

For static inspection, `reports/repeated_split_audit_index.md` collects the
available repeated-split summaries and points to companion CSVs, including the
per-repeat rows shipped for the standard-qrel dense 60/20/20 audit.

| Report family | Unit | Repeats/resampling | Interval meaning |
| --- | --- | --- | --- |
| Standard-qrel dense access | Query within dataset, paired by route policy | Raw stratified 60/20/20 train/dev/test splits over five public-qrel datasets; bootstrap intervals for paired utility deltas in headline rows | 95% interval for adaptive-minus-reference utility under the declared cost profile. |
| Dense learner-family sweeps | Repeated stratified split summary | 20 repeated stratified 60/20/20 splits; seed recorded as `31`; all rows use the same B1-legal compact feature budget | Interval over repeated split deltas. The sweep is artifact-only learner sensitivity, not a proposed router. |
| 2Wiki structured / hyperlink stress | Query row, paired by fixed/adaptive policy | Repeated held-out splits reported in the frozen audit files; query-local controls are paired to the same question rows | 95% interval for adaptive-minus-best-fixed utility. |
| HotpotQA / MuSiQue supporting checks | Query row, paired by fixed/adaptive policy | Internal train/dev/test route split within the upstream train split | 95% interval for adaptive-minus-best-fixed utility; supporting generalization only. |
| Systems/cost profiles | View/profile row | No inferential CI; profile tables are deterministic rescoring/profiling records | Cost-profile sensitivity, not a statistical significance claim. |

Reports that do not ship row-level repeats are marked as aggregate frozen
reports. They remain useful for checking paper numbers and claim boundaries,
but they should not be read as private-test raw data dumps.
