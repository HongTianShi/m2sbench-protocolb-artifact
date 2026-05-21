#!/usr/bin/env python
"""Same-candidate nested evidence audit for standard IR collections.

The public route scorecard compares view-specific retrieval operators. This
audit removes the largest confound by fixing a single candidate pool for each
query, then reranking that same pool with increasingly rich evidence:

summary: candidate's assigned coarse centroid
pq:      product-quantized reconstruction of the candidate vector
full:    original dense candidate vector

The result is a nested-evidence stress test: candidate generation is held fixed
and only the evidence used for scoring changes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    bootstrap_share,
    dcg,
    encode_or_load,
    eval_run,
    load_collection,
    per_query_ndcg,
    safe_id,
)


def train_centroids(doc_emb: np.ndarray, nlist: int):
    kmeans = faiss.Kmeans(
        doc_emb.shape[1],
        nlist,
        niter=25,
        verbose=False,
        spherical=True,
        gpu=faiss.get_num_gpus() > 0,
    )
    kmeans.train(doc_emb)
    centroids = kmeans.centroids.astype("float32")
    assign_index = faiss.IndexFlatIP(doc_emb.shape[1])
    assign_index.add(centroids)
    _, assign = assign_index.search(doc_emb, 1)
    return centroids, assign[:, 0].astype("int64")


def train_pq_reconstruction(doc_emb: np.ndarray, m: int, nbits: int, seed: int):
    pq = faiss.ProductQuantizer(doc_emb.shape[1], m, nbits)
    rng = np.random.default_rng(seed)
    train_n = min(len(doc_emb), max(8192, 512 * m))
    train = doc_emb[rng.choice(len(doc_emb), train_n, replace=False)]
    pq.train(train)
    codes = pq.compute_codes(doc_emb)
    recon = pq.decode(codes).astype("float32")
    norms = np.linalg.norm(recon, axis=1, keepdims=True)
    recon = recon / np.maximum(norms, 1e-8)
    return recon


def flat_candidates(doc_emb: np.ndarray, query_emb: np.ndarray, candidate_k: int):
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    t0 = time.perf_counter()
    scores, ids = index.search(query_emb, candidate_k)
    return scores, ids.astype("int64"), time.perf_counter() - t0


def rerank_same_candidates(query_emb: np.ndarray, candidates: np.ndarray, view_emb: np.ndarray, top_k: int):
    out = np.full((len(query_emb), top_k), -1, dtype="int64")
    scores = np.zeros((len(query_emb), min(top_k, candidates.shape[1])), dtype="float32")
    t0 = time.perf_counter()
    for qi, cand in enumerate(candidates):
        cand = cand[cand >= 0]
        if len(cand) == 0:
            continue
        sc = view_emb[cand] @ query_emb[qi]
        order = np.argsort(-sc)[:top_k]
        picked = cand[order]
        out[qi, : len(picked)] = picked
        scores[qi, : len(picked)] = sc[order[: scores.shape[1]]]
    return scores, out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--candidate-k", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=512)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    nlist = min(args.nlist, max(8, len(docs) // 20))

    _, candidates, t_candidates = flat_candidates(doc_emb, query_emb, args.candidate_k)
    centroids, assign = train_centroids(doc_emb, nlist)
    summary_emb = centroids[assign].astype("float32")
    pq_emb = train_pq_reconstruction(doc_emb, args.m, args.nbits, args.seed)
    full_emb = doc_emb.astype("float32")

    costs = {"summary": 0.0, "pq": 0.20, "full": 0.58}
    view_embs = {"summary": summary_emb, "pq": pq_emb, "full": full_emb}
    runs = {}
    per = {}
    util_per = {}
    ids_by = {}
    for view, emb in view_embs.items():
        scores, ids, elapsed = rerank_same_candidates(query_emb, candidates, emb, args.k)
        metrics = eval_run(ids, rels)
        metrics["cost"] = costs[view]
        metrics["utility@10"] = metrics["ndcg@10"] - args.lambda_cost * costs[view]
        metrics["mean_us_per_query"] = elapsed / len(queries) * 1e6
        runs[view] = metrics
        ids_by[view] = ids
        per[view] = per_query_ndcg(ids, rels, 10)
        util_per[view] = per[view] - args.lambda_cost * costs[view]

    stacked = np.stack([util_per["summary"], util_per["pq"], util_per["full"]], axis=1)
    best_idx = np.argmax(stacked, axis=1)
    names = np.array(["summary", "pq", "full"])
    best = names[best_idx]
    diag = {
        "candidate_pool": "full_flat_topk_fixed_for_all_views",
        "candidate_k": int(args.candidate_k),
        "non_full_best_share": float(np.mean(best != "full")),
        "summary_best_share": float(np.mean(best == "summary")),
        "pq_best_share": float(np.mean(best == "pq")),
        "full_best_share": float(np.mean(best == "full")),
        "pq_beats_full_precost_share": float(np.mean(per["pq"] >= per["full"])),
        "summary_beats_full_precost_share": float(np.mean(per["summary"] >= per["full"])),
        "lambda_star_summary_to_pq": float((runs["pq"]["ndcg@10"] - runs["summary"]["ndcg@10"]) / 0.20),
        "lambda_star_pq_to_full": float((runs["full"]["ndcg@10"] - runs["pq"]["ndcg@10"]) / 0.38),
        "bootstrap_ci": {
            "non_full_best_share": bootstrap_share(best != "full"),
            "pq_beats_full_precost_share": bootstrap_share(per["pq"] >= per["full"]),
        },
    }

    out = {
        "task": "nested_ir_refinement_audit",
        "dataset": args.dataset,
        "model": args.model,
        "faiss_gpus": int(faiss.get_num_gpus()),
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "nlist": int(nlist),
        "pq_m": args.m,
        "pq_nbits": args.nbits,
        "lambda": args.lambda_cost,
        "candidate_generation": {
            "view": "full_flat",
            "candidate_k": int(args.candidate_k),
            "mean_us_per_query": t_candidates / len(queries) * 1e6,
        },
        "views": runs,
        "diagnostics": diag,
    }
    stem = safe_id(args.dataset)
    json_path = REPORT_DIR / f"nested_ir_refinement_{stem}.json"
    md_path = REPORT_DIR / f"nested_ir_refinement_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# Same-Candidate Nested IR Refinement Audit: {args.dataset}",
        "",
        f"- Documents: {len(docs):,}",
        f"- Queries with qrels: {len(queries):,}",
        f"- Positive/graded qrel pairs retained: {sum(len(r) for r in rels):,}",
        f"- Encoder: `{args.model}`",
        f"- Candidate pool: full flat top-{args.candidate_k}, shared by all views",
        f"- Nested evidence: centroid assignment -> PQ reconstruction -> full vector",
        f"- FAISS GPUs visible: {faiss.get_num_gpus()}",
        "",
        "| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | rerank us/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in ["summary", "pq", "full"]:
        m = runs[v]
        lines.append(
            f"| {v} | {m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} | {m['recall@10']:.3f} | "
            f"{m['hit@10']:.3f} | {m['cost']:.2f} | {m['utility@10']:.3f} | {m['mean_us_per_query']:.2f} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        f"- Non-full cost-adjusted best share: {diag['non_full_best_share']:.3f}",
        f"- PQ beats full before cost share: {diag['pq_beats_full_precost_share']:.3f}",
        f"- Best-view shares: summary {diag['summary_best_share']:.3f}, PQ {diag['pq_best_share']:.3f}, full {diag['full_best_share']:.3f}",
        f"- Break-even lambda*: summary->PQ {diag['lambda_star_summary_to_pq']:.3f}, PQ->full {diag['lambda_star_pq_to_full']:.3f}",
        f"- Bootstrap 95% CI, non-full best share: [{diag['bootstrap_ci']['non_full_best_share']['lo']:.3f}, {diag['bootstrap_ci']['non_full_best_share']['hi']:.3f}]",
        f"- Bootstrap 95% CI, PQ beats full before cost: [{diag['bootstrap_ci']['pq_beats_full_precost_share']['lo']:.3f}, {diag['bootstrap_ci']['pq_beats_full_precost_share']['hi']:.3f}]",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
