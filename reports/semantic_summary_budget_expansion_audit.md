# Semantic Summary-Budget Expansion and Access Optimizer Audit

- Test queries: 1,149
- Views: summary, pq, full

## Semantic summary-budget expansion

| budget | utility | cost | regret | non-full oracle share | best-view AUC | gap closed | choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed_summary | 0.1389 | 0.0000 | 0.2920 | nan | -- | 0.000 | summary 1.00 |
| base_centroid_score_sketch | 0.3553 | 0.4658 | 0.0756 | 0.684 | 0.563 | 0.741 | summary 0.01, pq 0.29, full 0.70 |
| plus_query_or_lexical_sketch | 0.3551 | 0.4615 | 0.0758 | 0.684 | 0.560 | 0.740 | summary 0.01, pq 0.30, full 0.69 |
| plus_summary_pq_score_sketch | 0.3563 | 0.4297 | 0.0746 | 0.684 | 0.544 | 0.745 | summary 0.03, pq 0.35, full 0.62 |
| plus_pq_agreement_sketch | 0.3588 | 0.4202 | 0.0721 | 0.684 | 0.560 | 0.753 | summary 0.04, pq 0.36, full 0.60 |
| plus_all_declared_low_cost_context | 0.3578 | 0.4287 | 0.0731 | 0.684 | 0.569 | 0.750 | summary 0.02, pq 0.37, full 0.61 |

## Cost-based access optimizer baselines

| optimizer | utility | raw quality | cost | profiled us | regret | gap closed | choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed_summary | 0.1389 | 0.1389 | 0.0000 | 0.50 | 0.2920 | 0.000 | summary 1.00 |
| fixed_pq | 0.3240 | 0.3400 | 0.2000 | 24.30 | 0.1069 | 0.634 | pq 1.00 |
| fixed_full | 0.3517 | 0.3981 | 0.5800 | 4.10 | 0.0792 | 0.729 | full 1.00 |
| two_threshold_cascade | 0.3494 | 0.3914 | 0.5259 | 5.12 | 0.0815 | 0.721 | summary 0.05, pq 0.06, full 0.89 |
| cost_based_learned_optimizer | 0.3589 | 0.3927 | 0.4222 | 11.17 | 0.0720 | 0.753 | summary 0.04, pq 0.36, full 0.60 |
| oracle_optimizer | 0.4309 | 0.4512 | 0.2544 | 10.11 | 0.0000 | 1.000 | summary 0.33, pq 0.36, full 0.32 |

## Set-valued access recommendation

| split | policy | oracle-view coverage@2 | avg set size | regret@2 | utility@2 | top-1 utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| semi_real | cheapest-first | 0.556 | 2.0 | 0.4274 | 0.2765 | 0.2222 |
| semi_real | richest-first | 0.444 | 2.0 | 0.2952 | 0.4087 | 0.1203 |
| semi_real | uncertainty-first | 0.667 | 2.0 | 0.3228 | 0.3812 | 0.2222 |
| semi_real | learned view ranker | 0.500 | 2.0 | 0.3340 | 0.3700 | 0.1805 |
| synthetic_heldout | cheapest-first | 0.896 | 2.0 | 0.1004 | 0.8377 | 0.5729 |
| synthetic_heldout | richest-first | 0.104 | 2.0 | 0.1307 | 0.8074 | 0.6515 |
| synthetic_heldout | uncertainty-first | 0.646 | 2.0 | 0.0572 | 0.8809 | 0.5720 |
| synthetic_heldout | learned view ranker | 0.417 | 2.0 | 0.0438 | 0.8943 | 0.7073 |

## Graph/KG adapter rows

| adapter | source | cells | task | summary BAcc | best view | best BAcc |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Citation KG topology | Cora/CiteSeer | 1810 | triangle | 0.797 | degree_view | 1.000 |
| Citation KG topology | Cora/CiteSeer | 1810 | bridge | 0.753 | degree_view | 1.000 |
| DocRED relation retrieval | DocRED | 3027 | multi-sentence evidence route | 0.527 | entity_type_view | 0.934 |
| Recommendation preference | Goodreads 10k | 9861 | audience/genre route | 0.123 | tag_sketch_view | 0.692 |

## Sanity controls

| control | expected | observed |
| --- | --- | --- |
| shuffled utilities in router training | route signal should lose most access-optimizer value | utility 0.3449, regret 0.0860, cost 0.4772 |
| hidden-field validator | target/canonical/reference parameters are rejected | covered by tests/test_research_protocol.py::test_blind_method_rejects_forbidden_parameter_name |
| submission-kit validator | malformed or evaluator-only fields fail before scoring | scripts/check_submission_kit.py runs validate_submission.py before evaluate_submission.py |
| cost perturbation | qualitative frontier should not vanish under moderate cost noise | existing cost audit: synthetic best-view flips on 0.2% of cells; IVF-PQ non-full best .764 [.759,.769] |
