# 2Wiki Structured Evidence Data Card

This card documents the public data assumptions for the 2Wiki structured
evidence and 7GB hyperlink-corpus stress reports. The review artifact ships
compact frozen reports and public-dev fixtures; it does not require reviewers
to rerun the 7GB corpus pass.

## Upstream Sources

| Component | Source | Used fields | Role |
| --- | --- | --- | --- |
| 2WikiMultiHopQA structured QA rows | HuggingFace dataset `voidful/2wikimultihopqa` | `_id`, `type`, `question`, `context`, `supporting_facts`, `evidences`, `answer` | Builds the provided-context structured evidence slice and evaluator-held support-title/fact/triple labels. |
| 2Wiki hyperlink paragraph corpus | Public `2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl` file | paragraph title, text, outgoing hyperlinks | Builds query-local one-hop/two-hop paid evidence views for the 7GB stress audit. |

The paper scores evidence acquisition, not generated answers: answers,
supporting facts, evidence triples, and unpaid CE scores are evaluator-held.

## Frozen Hyperlink Corpus Snapshot

The 10k 7GB stress row was built from the public hyperlink paragraph corpus
snapshot stored locally as
`<local-data-root>/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl`.
For the frozen audit run, the local file check was:

- SHA256:
  `A9A8EC7D92010B3282E606F25DC56EA121926EAEBD2296FD18FC52721A81B349`
- File size: `7023046781` bytes.
- Line count: `5989738` JSONL rows.
- Systems-profile scan: three cold streaming passes, `17969214` total lines
  scanned and `21036225042` total bytes streamed, as recorded in
  `reports/2wiki_hyperlink_systems_profile_10000.md`.

The clean artifact ships frozen aggregate reports and compact public-dev
fixtures, not the 7GB raw corpus. Reviewers can verify that the paper-facing
10k row, systems profile, and lambda sensitivity point to this same snapshot.

## Expected Local Layout For Full Rebuilds

The full hyperlink stress script accepts an explicit local data root. A typical
layout is:

```text
<local-data-root>/
  2WikiMultiHopQA/
    para_with_hyperlink/
      para_with_hyperlink.jsonl
```

For HuggingFace datasets, set `HF_HOME`/`HF_DATASETS_CACHE` outside the artifact
checkout if you rerun the structured slice. The frozen report records use
`<local-data-root>` placeholders rather than personal paths.

## Frozen Report Policy

The canonical paper-facing hyperlink result is:

- `reports/2wiki_hyperlink_corpus_stress_10000.md`
- `reports/2wiki_hyperlink_corpus_stress_10000.json`
- `reports/2wiki_hyperlink_systems_profile_10000.md`
- `reports/2wiki_hyperlink_systems_profile_10000.json`
- `reports/2wiki_lambda_sensitivity.md`
- `reports/2wiki_lambda_sensitivity.json`

The 5k and 2k reports are consistency checks. Large intermediate title maps,
paragraph extracts, cache arrays, and local logs are intentionally excluded from
clean release archives; they can be regenerated from the public sources with
`scripts/run_2wiki_hyperlink_corpus_stress.py` if needed.

`reports/2wiki_lambda_sensitivity.*` is a lightweight aggregate rescore from
the frozen reports, not a rebuild: it recomputes `raw - lambda * mean_cost` for
fixed and adaptive policy rows over a small lambda grid to check whether the
2Wiki gains are tied to one default cost weight.

## Controls And Feature Tiers

| Name | Definition |
| --- | --- |
| `title_only` / `summary` | Uses only the released question and title/entity sketch. |
| `real_1hop` | Buys one-hop hyperlink expansion from provided-context titles. |
| `shuffled_1hop` | Replaces one-hop links with shuffled links as an anti-token control. |
| `degree_random_1hop` | Samples degree-matched random one-hop titles to test whether degree/coverage alone explains gains. |
| `real_2hop` | Buys capped two-hop paths from the hyperlink corpus. |
| `random_2hop` | Replaces two-hop paths with random paths under the same cap. |
| `B0_title` | Legal learner features from title/entity sketch only. |
| `B1_entity` | Adds released entity-overlap features. |
| `B1_graph` | Adds released graph-degree/one-hop coverage sketches. |
| `B1_path` | Adds released two-hop path sketches. |
| `B1_context` | Adds released context-length/agreement sketches. |

These are generic manifest fields, not private 2Wiki-specific feature
engineering requirements: a participant can use them as released, ignore them,
or declare another low-cost summary through the new-view onboarding checklist.
Evaluator-held support titles, supporting facts, evidence triples, answers, and
view scores remain hidden until scoring.

The small `public_dev/` fixtures use deterministic synthetic score sketches to
exercise the submission/scoring loop without shipping the full corpus. The
frozen paper reports above are the source of the real 2Wiki and hyperlink
readouts; public-dev sketch values should be treated as route-schema fixtures,
not as a claim about feature engineering requirements.
For reviewers who want to see the raw-label-to-view-score idea without loading
2Wiki, `public_dev/evaluator_only/2wiki_synthetic_raw_label_toy.json` gives one
synthetic, non-training example with support facts, evidence triples, an answer,
and illustrative view scores.
