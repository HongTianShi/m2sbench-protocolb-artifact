# Cross-Encoder Reranker Access Audit: beir/scifact/test

- Bi-encoder: `BAAI/bge-small-en-v1.5`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Documents: 5,183
- Queries: 300
- Candidate pool: top-50; evaluation: NDCG@10

| view | NDCG@10 | cost | utility@10 | mean us/q |
| --- | ---: | ---: | ---: | ---: |
| bi-encoder | 0.720 | 0.000 | 0.720 | 17.3 |
| cross-encoder | 0.700 | 0.450 | 0.664 | 85451.1 |
| selective router | 0.730 | 0.023 | 0.728 |  |
| oracle route |  |  | 0.774 |  |

- Raw reranker gain: -0.0199
- Cost-adjusted reranker win share: 0.167
- Break-even lambda: -0.044
