# GPU FAISS DocRED Evidence Access Audit

Dense exact-qrel evidence retrieval audit using sentence-transformer embeddings and FAISS.

- Corpus sentences: 532,313
- Queries: 30,000
- Extra train-distant distractors requested: 500,000
- Qrels: DocRED evidence sentence annotations
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS GPUs visible: 1
- IVF/PQ: nlist=2048, m=24, nbits=8, nprobe=32

| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.008 | 0.011 | 0.019 | 0.031 | 0.00 | 0.011 | 2.91 |
| pq | 0.485 | 0.514 | 0.644 | 0.740 | 0.20 | 0.498 | 17.94 |
| full | 0.757 | 0.786 | 0.933 | 0.967 | 0.58 | 0.740 | 44.31 |

## Access diagnostics

- Non-full cost-adjusted best share: 0.375
- PQ dominates full before cost share: 0.362
- Best-view shares: summary 0.034, PQ 0.341, full 0.625
- Break-even lambda*: summary->PQ 2.516, PQ->full 0.715

## Held-out adaptive routers

| router | test queries | NDCG@10 | cost | utility@10 | regret@10 | choice shares |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| summary-visible proxy router (B0) | 9,900 | 0.789 | 0.580 | 0.742 | 0.031 | summary 0.00, pq 0.00, full 1.00 |
| PQ-aware proxy router (B1) | 9,900 | 0.780 | 0.554 | 0.735 | 0.033 | pq 0.07, full 0.93 |
