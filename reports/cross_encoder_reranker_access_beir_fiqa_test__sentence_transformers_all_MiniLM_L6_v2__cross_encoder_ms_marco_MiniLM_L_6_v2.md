# Cross-Encoder Reranker Access Audit: beir/fiqa/test

- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Documents: 57,638
- Queries: 648
- Candidate pool: top-50; evaluation: NDCG@10

| view | NDCG@10 | cost | utility@10 | mean us/q |
| --- | ---: | ---: | ---: | ---: |
| bi-encoder | 0.369 | 0.000 | 0.369 | 21.7 |
| cross-encoder | 0.387 | 0.450 | 0.351 | 44066.9 |
| cheap-only CE adapter | 0.389 | 0.196 | 0.374 |  |
| oracle route |  |  | 0.439 |  |

- Adapter feature budget: cheap bi-encoder score statistics only
- Raw reranker gain: 0.0183
- Cost-adjusted reranker win share: 0.287
- Break-even lambda: 0.041
- Boundary CSV: `cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.csv`
- Boundary figure: `cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.png`
