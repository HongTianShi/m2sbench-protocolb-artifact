# Public-Dev Feature Dictionary

This dictionary lists the concrete `visible_features` keys used in the
paper-facing public-dev manifests. These features are lightweight, released
summary/context sketches for reference routers. They are not proposed model
features, and they do not contain qrels, answers, support facts, CE scores,
full-view scores, oracle utilities, or hidden targets.

Private-test rows may refresh ids, corpora, seeds, and feature values, but the
same visibility rule applies: a solver may use only fields released in the
manifest and the declared menu before buying a view.

The public-dev values are deterministic scorer-fixture sketches designed to
exercise the Protocol B route contract in a small repository checkout. They
should not be read as the only acceptable feature engineering recipe. The
frozen paper reports document the real dense and 2Wiki feature tiers; future
participants may use the released fields, ignore them, or declare new low-cost
summaries through `docs/new_view_onboarding.md`.

## Shared Row Fields

| Field | Meaning | Legal use |
| --- | --- | --- |
| `query_id` | Opaque row id. | Join submitted route to manifest row. |
| `cell_id` | Opaque evidence cell id. | Join submitted route to manifest row. |
| `slice_id` | Slice name such as `dense_semantic_access`. | Filter/report by slice. |
| `split` | Public-dev split marker. | Audit only. |
| `tier` | Visibility tier such as `B1_score`, `B1_path`, or `B1_hyperlink`. | Select compatible solver policy. |
| `cost_menu` | Declared cost-profile id. | Must match submitted route row. |
| `declared_views` | Legal candidate views for the row. | Route selection menu. |
| `visible_features` | Released cheap summary/context features. | Legal B0/B1 router input. |

## Dense Semantic Access

Manifest: `public_dev/dense_semantic.manifest.jsonl`

| Feature | Tier | Description | Hidden? |
| --- | --- | --- | --- |
| `candidate_count` | B0/B1 score sketch | Number of candidate items in the cheap first-stage pool. | No |
| `cheap_margin` | B1 score sketch | Released confidence/margin statistic from cheap visible scores. | No |
| `cheap_entropy` | B1 score sketch | Released uncertainty statistic from cheap visible scores. | No |
| `agreement_sketch` | B1 agreement sketch | Released coarse agreement proxy among cheap declared views. | No |

Forbidden before purchase: qrels, raw document text not declared visible, full
vector scores, unpaid CE scores, oracle utilities, and any source labels used
only by the evaluator.

## 2Wiki Structured Evidence

Manifest: `public_dev/2wiki_structured.manifest.jsonl`

| Feature | Tier | Description | Hidden? |
| --- | --- | --- | --- |
| `question_type` | B0/B1 metadata | Released coarse type label from the public row metadata. | No |
| `title_overlap` | B0/B1 title sketch | Cheap title/entity overlap statistic. | No |
| `graph_agreement` | B1 graph sketch | Released one-hop neighborhood/agreement proxy. | No |
| `path_sketch_signal` | B1 path sketch | Released coarse two-hop/path evidence proxy. | No |

Forbidden before purchase: gold supporting facts, evidence triples, answer,
unpaid context/CE scores, oracle best view, and evaluator-held support labels.

## 2Wiki Hyperlink Stress

Manifest: `public_dev/2wiki_hyperlink.manifest.jsonl`

| Feature | Tier | Description | Hidden? |
| --- | --- | --- | --- |
| `question_type` | B0/B1 metadata | Released coarse type label from the public row metadata. | No |
| `title_overlap` | B0/B1 title sketch | Cheap title/entity overlap statistic. | No |
| `graph_agreement` | B1 hyperlink sketch | Released real-vs-control hyperlink agreement proxy. | No |
| `path_sketch_signal` | B1 hyperlink/path sketch | Released coarse two-hop hyperlink path proxy. | No |

Forbidden before purchase: gold support facts, evidence triples, answer,
unpaid hyperlink-expanded paragraph contents, CE/full scores, oracle views, and
evaluator-held controls.

## Public-Dev References

Files under `public_dev/evaluator_only/*.reference.jsonl` are evaluator-held
public-dev references for the local scorer. They are provided so reviewers can
exercise scoring without a private server, but are physically separated from
the root `public_dev/*.manifest.jsonl` solver inputs. Solvers should not train
on or import these files; private-test rows withhold the analogous references.
