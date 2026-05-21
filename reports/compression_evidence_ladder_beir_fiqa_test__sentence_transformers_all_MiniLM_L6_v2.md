# Compression Evidence Ladder: beir/fiqa/test

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Documents: 57,638
- Queries loaded: 648
- Reported queries: 260 (CE-labeled boundary split)
- FAISS GPUs visible: 1
- Menu oracle utility: 0.448; best fixed: int8 dense 0.337; oracle gap: 0.111

| view | bytes touched | latency us/q | raw NDCG@10 | utility | lambda* from prev | oracle-best share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | cell stats | 47.8 | 0.019 | 0.019 |  | 0.262 |
| binary sign | 48 B | 581.8 | 0.287 | 0.281 | 3.355 | 0.192 |
| IVF-PQ | 24 B code | 46.1 | 0.255 | 0.239 | -0.268 | 0.088 |
| trunc-96 fp32 | 384 B | 3.6 | 0.281 | 0.261 | 0.434 | 0.108 |
| int8 dense | 384 B | 8.4 | 0.361 | 0.337 | 1.991 | 0.119 |
| trunc-192 fp32 | 768 B | 5.0 | 0.334 | 0.302 | -0.273 | 0.065 |
| full dense | 1536 B | 7.5 | 0.364 | 0.317 | 0.166 | 0.015 |
| cross-encoder rerank | top-50 text | 44066.9 | 0.376 | 0.294 | 0.028 | 0.150 |

Binary/int8 rows are post-training compressed views searched through a portable dequantized FAISS path; they are evidence-menu baselines, not new encoder training.
