# Data, Model, And Report Use Notes

The code in this anonymous artifact is released under the top-level MIT
license. Third-party datasets, models, and licensed services keep their own
upstream terms.

## Public Datasets And Models

| Resource | Artifact role | Upstream terms |
| --- | --- | --- |
| BEIR/standard qrel datasets used by the dense semantic audits | Public qrel validity anchors and dense access stress rows | Follow the dataset-specific BEIR/upstream licenses and citation requirements. |
| `voidful/2wikimultihopqa` | 2Wiki structured evidence acquisition and support-title/fact evaluation | Follow the HuggingFace dataset card and upstream 2WikiMultiHopQA terms. |
| 2Wiki `para_with_hyperlink.jsonl` paragraph corpus | 7GB hyperlink-corpus stress audit | Public upstream corpus terms apply; the clean release keeps compact frozen reports, not the full corpus. |
| HotpotQA distractor split local HuggingFace export | Supporting structured-evidence generalization check over hidden supporting facts and paragraph/sentence evidence | Follow the HotpotQA dataset license, citation requirements, and upstream terms. |
| `bdsaglam/musique` local HuggingFace export | Supporting MuSiQue structured-family replication over decomposition questions and paragraph evidence | Follow the HuggingFace dataset card and upstream MuSiQue terms. |
| `tasksource/strategy-qa` local HuggingFace export | Exploratory reasoning-frontier check over question decompositions and supporting facts | Follow the HuggingFace dataset card and upstream StrategyQA terms. |
| FEVER official JSONL local export | Stopped-after-smoke claim-evidence handle check over official evidence title/sentence-id handles | Follow the FEVER dataset license, citation requirements, and upstream terms. |
| Sentence-transformers / BGE / cross-encoder checkpoints named in reports | Dense encoders and paid CE reranker views | Follow each model card/license and provider terms. |
| DocRED, ANTIQUE, FiQA, SciFact, NFCorpus, ArguAna, Douban/JD/TapTap-derived stress records where applicable | Semantic, scale, or stress audits | Follow the respective upstream licenses and data-use restrictions. |

## Licensed Or Optional Services

The Bloomberg adapter is optional and is excluded from the public validation
claim. It contains scripts and request schemas only; returned vendor exports,
row-level licensed data, credentials, and API tokens are not included.

## Generated Reports

Markdown/JSON reports in `reports/` and compact CSV files in `summaries/`
are review artifacts derived from the public or optional sources above. Large
local caches, raw downloaded corpora, local profiler logs, and bytecode caches
are excluded from clean release archives.
