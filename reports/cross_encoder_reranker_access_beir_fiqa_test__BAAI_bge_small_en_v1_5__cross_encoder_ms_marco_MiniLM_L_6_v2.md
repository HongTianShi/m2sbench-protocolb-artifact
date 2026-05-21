# Cross-Encoder Reranker Access Audit: beir/fiqa/test

- Bi-encoder: `BAAI/bge-small-en-v1.5`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Documents: 57,638
- Queries: 648
- Candidate pool: top-50; evaluation: NDCG@10

| view | NDCG@10 | cost | utility@10 | mean us/q |
| --- | ---: | ---: | ---: | ---: |
| bi-encoder | 0.385 | 0.000 | 0.385 | 21.3 |
| cross-encoder | 0.384 | 0.450 | 0.348 | 46613.5 |
| selective router | 0.377 | 0.113 | 0.368 |  |
| oracle route |  |  | 0.439 |  |

- Raw reranker gain: -0.0011
- Cost-adjusted reranker win share: 0.247
- Break-even lambda: -0.003
