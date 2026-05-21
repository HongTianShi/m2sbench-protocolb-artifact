# Official Benchmark Contract Card

This card is the reviewer-facing map for the paper-facing Protocol B release.
It separates validated slices from stress, diagnostic, and portability records.
The release is an evidence-purchase benchmark, not a retrieval-SOTA claim, an
end-to-end RAG benchmark, or a universal deployment-cost benchmark.

In plain terms, a submitted row buys evidence before scoring. The benchmark
does not score generated answers, free-form rationales, or agent trajectories
unless a future slice declares them as structured route fields.

## Official and Supporting Slices

| Slice | Status | Method-visible state | Declared paid views | Evaluator-held fields | Canonical report |
|---|---|---|---|---|---|
| Dense semantic access | Validated public-dev core | Query/cell ids, summary and score statistics, declared view/cost menu | Binary, PQ, int8, HNSW16/64, full dense, CE rerank | Qrels, relevance labels, unpaid full/CE scores, oracle view, view utilities | `reports/protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.md`; `reports/joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md` |
| Compression/CE purchase | Dense-core audit | Same dense state plus declared first-stage pool metadata | Binary, PQ, int8, full dense, CE rerank | Qrels, unpaid dense/CE scores, oracle view, view utilities | `reports/compression_evidence_ladder_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.md`; `reports/ce_purchase.md` |
| 2Wiki structured evidence | Validated structured-evidence core | Question, title/entity sketch, released graph/path/context sketches | One-hop evidence, two-hop paths, full context, CE rerank | Supporting facts, support titles, evidence triples, answers, unpaid CE scores, oracle view | `reports/2wiki_structured_evidence_2000.md` |
| 2Wiki 7GB hyperlink stress | Structured-core stress audit | Question, provided-context title sketch, hyperlink degree/path sketches, local-pool sketch | Provided context, 1-hop hyperlink expansion, 2-hop expansion, full local pool | Supporting facts, support titles, evidence triples, answers, oracle view | `reports/2wiki_hyperlink_corpus_stress_10000.md` |
| IVF-PQ 159k scale stress | Scale stress only | Coarse text-vector cell summaries | PQ/full rerank variants | Label-derived evaluator bits and oracle view | `reports/large_ivf_pq_rerank_adapter_audit.md` |
| Controlled matched-summary lab | Mechanism diagnostic only | Matched compact summary cells | Diagnostic reconstruction/access actions | Hidden target structures and private seeds | `summaries/static_overfit_audit/README.md`; `reports/proxy_diagnostic_routing_audit.md` |
| Other adapters | Portability only | Slice-specific released summaries | Visual/audio/recommendation/licensed or extra graph views | Slice-specific evaluator fields | Listed in `ARTIFACT_INVENTORY.md` |

## Route Row Semantics

- Primary leaderboard action: `route`.
- `ranked_views` is diagnostic; if `route` is absent, the first valid ranked
  view is treated as the primary route.
- `view_set` and `abstain` are diagnostic unless a future slice declares a
  non-oracle set rule or abstention rule. Current official menus reject them
  for leaderboard scoring.
- Submitted rows must match the declared `cost_menu` and released manifest.
- When a manifest row contains `declared_views`, the validator and public-dev
  scorer restrict that row to the intersection of the global view menu and the
  row-level declared view list.
- Unknown top-level keys fail unless `--allow-extra-keys` is explicitly used;
  evaluator-held keys fail recursively inside `diagnostics`.

## Public-Dev Submission/Evaluation Loop

The paper-facing official/core slices ship with lightweight public-dev packs
under `public_dev/`. Each pack contains a method-visible manifest and an
evaluator-held reference file isolated under `public_dev/evaluator_only/`. A
new method may emit any valid route JSONL over the released manifest;
`scripts/score_protocol_b_public_dev.py` validates the route file against the
declared view/cost menus and writes per-row raw quality, charged cost, utility,
regret, oracle headroom, and a leaderboard row. The same contract is used by
private-test scoring, except that the analogous reference file is withheld and
only aggregate leaderboard rows are returned.

Example:

```bash
python scripts/score_protocol_b_public_dev.py \
  submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl \
  --manifest public_dev/2wiki_hyperlink.manifest.jsonl \
  --reference public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl \
  --view-menu menus/2wiki_hyperlink.view_menu.json \
  --cost-menu menus/2wiki_hyperlink.cost_menu.json
```

The `2wiki_hyperlink` public-dev pack is a structured-stress fixture rather
than a separate official leaderboard core; its paper-facing claim is the frozen
10k report listed below.

## Official Metrics

- Primary: cost-adjusted regret under the declared profile.
- Secondary: raw quality, charged cost, buy rate, oracle headroom.
- Diagnostics: latency, QPS, bytes/materialization profile, leakage status,
  repeated-split intervals, and stress controls.

## Cost Profile Provenance

- `C_op` uses coarse, versioned access prices assigned by the
  view-dependency DAG before scoring. The constants are a leaderboard
  convention, not retrieval-outcome tuning and not a regression fitted to one
  local machine.
- `C_mem` is a conservative secondary rescoring profile from per-query bytes
  touched plus resident index/model footprint; it is not a production
  amortization claim.
- `C_lat` is a secondary rescoring profile from p95 latency under the fixed
  profiler configuration, where HNSW runs on CPU and dense/CE views run on GPU.
- Secondary dense profiles are accepted by
  `menus/dense_semantic.view_menu.json` through `accepted_cost_menus`; the
  active scorer field is always `declared_cost`.
- CE has two accounting contexts: incremental CE purchase cost `.45` after a
  first-stage pool exists, and cumulative ladder cost `1.03` when CE is treated
  as the richest text-reranking view.
- Profile disagreement is expected; Protocol B stores raw quality, declared
  cost, bytes, latency, and throughput separately so future releases can rescore
  under different profiles.

## Canonical 2Wiki Status

The paper foregrounds the 10k-row 7GB hyperlink-corpus stress report:
`reports/2wiki_hyperlink_corpus_stress_10000.md`.
The 5k and 2k reports are retained as consistency checks, not as the main
paper number. The provided-context 2Wiki structured slice remains canonical at
2,000 rows: `reports/2wiki_structured_evidence_2000.md`.

HotpotQA and MuSiQue are supporting structured-family checks, not additional
official cores. DocRED, StrategyQA, and FEVER are boundary checks. The compact
map for these rows is `reports/structured_evidence_check_map.md`.

For legal B0/B1 router features and forbidden fields, see
`docs/legal_router_feature_tiers.md`. For profile-relative cost accounting,
including `C_op`, `C_mem`, `C_lat`, lambda, and incremental-vs-cumulative CE
costs, see `docs/cost_profile_changelog.md`.

## Private-Test Governance

A hosted leaderboard can reuse the same `schema/` and `menus/` files while
withholding private qrels, support facts, answers, source labels, paid scores,
or refreshed seeds. Policy for one leaderboard version:

- Submission cap: versioned by release; repeated probing beyond the cap is invalid.
- Feedback: aggregate leaderboard rows only; no per-row private labels, CE
  scores, support facts, or oracle views.
- Solver version: code/config/checkpoint plus submitted JSONL; any change starts
  a new solver version.
- Refresh: maintainers may refresh qrels, support facts, seeds, candidate pools,
  or corpora while preserving route schema, view-menu ids, cost-menu ids, and
  scorer ids.
- External tools: allowed only over method-visible fields and purchased views;
  memorized public qrels/support facts or undeclared caches are violations.

## Files To Inspect

- `manifests/official_slices.json`
- `schema/route_schema.json`
- `schema/view_menu_schema.json`
- `schema/cost_menu_schema.json`
- `menus/*.view_menu.json`
- `menus/*.cost_menu.json`
- `public_dev/README.md`
- `scripts/score_protocol_b_public_dev.py`
- `docs/submission_contract.md`
- `docs/new_view_onboarding.md`
- `docs/provenance_and_genai.md`
- `reports/PAPER_EVIDENCE_INDEX.md`
- `reports/structured_evidence_check_map.md`
- `docs/legal_router_feature_tiers.md`
- `docs/cost_profile_changelog.md`
