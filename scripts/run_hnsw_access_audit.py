#!/usr/bin/env python
"""HNSW access-depth audit for standard IR collections.

This script instantiates a second ANN systems view for M2S-Bench. It keeps the
same query-document qrels as the standard IR audit, but varies HNSW efSearch as
the purchasable evidence-depth knob. The audit records raw NDCG, operation-unit
cost, utility, and per-query latency summaries.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    bootstrap_share,
    encode_or_load,
    eval_run,
    heldout_fixed_views,
    load_collection,
    per_query_ndcg,
    query_features,
    safe_id,
    score_features,
    tree_router,
)


def build_hnsw(doc_emb: np.ndarray, hnsw_m: int, ef_construction: int):
    index = faiss.IndexHNSWFlat(doc_emb.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.add(doc_emb)
    return index


def timed_search(index, query_emb: np.ndarray, k: int, ef_search: int):
    index.hnsw.efSearch = int(ef_search)
    ids = np.empty((len(query_emb), k), dtype="int64")
    scores = np.empty((len(query_emb), k), dtype="float32")
    per_us = np.empty(len(query_emb), dtype="float64")
    for i, q in enumerate(query_emb):
        t0 = time.perf_counter()
        s, r = index.search(q.reshape(1, -1), k)
        per_us[i] = (time.perf_counter() - t0) * 1e6
        scores[i] = s[0]
        ids[i] = r[0]
    return scores, ids, per_us


def latency_summary(per_us: np.ndarray):
    return {
        "mean_us_per_query": float(np.mean(per_us)),
        "p50_us_per_query": float(np.quantile(per_us, 0.50)),
        "p95_us_per_query": float(np.quantile(per_us, 0.95)),
    }


def timed_flat_cpu(doc_emb: np.ndarray, query_emb: np.ndarray, k: int):
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    ids = np.empty((len(query_emb), k), dtype="int64")
    scores = np.empty((len(query_emb), k), dtype="float32")
    per_us = np.empty(len(query_emb), dtype="float64")
    for i, q in enumerate(query_emb):
        t0 = time.perf_counter()
        s, r = index.search(q.reshape(1, -1), k)
        per_us[i] = (time.perf_counter() - t0) * 1e6
        scores[i] = s[0]
        ids[i] = r[0]
    return scores, ids, per_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--ef-search", default="16,64,128")
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    ef_levels = [int(x) for x in args.ef_search.split(",") if x.strip()]
    index = build_hnsw(doc_emb, args.hnsw_m, args.ef_construction)

    views = {}
    per = {}
    util_per = {}
    costs = {}
    scores_by = {}
    ids_by = {}
    max_ef = max(ef_levels)
    for ef in ef_levels:
        name = f"hnsw{ef}"
        # Monotone operation-unit profile: shallow graph walk costs less than
        # full-vector exhaustive reranking but more than a centroid summary.
        cost = 0.10 + 0.30 * (ef / max_ef)
        scores, ids, per_us = timed_search(index, query_emb, args.k, ef)
        metrics = eval_run(ids, rels)
        metrics["cost"] = cost
        metrics["utility@10"] = metrics["ndcg@10"] - args.lambda_cost * cost
        metrics.update(latency_summary(per_us))
        views[name] = metrics
        costs[name] = cost
        per[name] = per_query_ndcg(ids, rels, 10)
        util_per[name] = per[name] - args.lambda_cost * cost
        scores_by[name] = scores
        ids_by[name] = ids

    scores_f, ids_f, full_us = timed_flat_cpu(doc_emb, query_emb, args.k)
    costs["full"] = 0.58
    metrics = eval_run(ids_f, rels)
    metrics["cost"] = 0.58
    metrics["utility@10"] = metrics["ndcg@10"] - args.lambda_cost * 0.58
    metrics.update(latency_summary(full_us))
    views["full"] = metrics
    per["full"] = per_query_ndcg(ids_f, rels, 10)
    util_per["full"] = per["full"] - args.lambda_cost * 0.58

    names = list(views)
    stacked = np.stack([util_per[v] for v in names], axis=1)
    best_idx = np.argmax(stacked, axis=1)
    best_names = np.array(names)[best_idx]

    feat = [query_features(query_emb)]
    for ef in ef_levels[: min(2, len(ef_levels))]:
        feat.append(score_features(scores_by[f"hnsw{ef}"]))
    router = tree_router(np.column_stack(feat), util_per, per, costs, args.lambda_cost)

    shallow = f"hnsw{ef_levels[0]}"
    deepest = f"hnsw{ef_levels[-1]}"
    out = {
        "task": "hnsw_access_audit",
        "dataset": args.dataset,
        "model": args.model,
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "hnsw_m": args.hnsw_m,
        "ef_construction": args.ef_construction,
        "ef_search": ef_levels,
        "lambda": args.lambda_cost,
        "views": views,
        "best_view_shares": {v: float(np.mean(best_names == v)) for v in names},
        "non_full_best_share": float(np.mean(best_names != "full")),
        "shallow_dominates_full_precost_share": float(np.mean(per[shallow] >= per["full"])),
        "deep_dominates_full_precost_share": float(np.mean(per[deepest] >= per["full"])),
        "bootstrap_ci": {
            "non_full_best_share": bootstrap_share(best_names != "full"),
            "shallow_dominates_full_precost_share": bootstrap_share(per[shallow] >= per["full"]),
        },
        "router": router,
    }

    stem = safe_id(args.dataset)
    json_path = REPORT_DIR / f"hnsw_access_{stem}.json"
    md_path = REPORT_DIR / f"hnsw_access_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# HNSW Access Audit: {args.dataset}",
        "",
        f"- Documents: {len(docs):,}",
        f"- Queries with qrels: {len(queries):,}",
        f"- HNSW: M={args.hnsw_m}, efConstruction={args.ef_construction}, efSearch={ef_levels}",
        "",
        "| view | NDCG@10 | cost | utility@10 | mean us/q | p50 us/q | p95 us/q |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in names:
        m = views[v]
        lines.append(
            f"| {v} | {m['ndcg@10']:.3f} | {m['cost']:.3f} | {m['utility@10']:.3f} | "
            f"{m['mean_us_per_query']:.2f} | {m['p50_us_per_query']:.2f} | {m['p95_us_per_query']:.2f} |"
        )
    lines += ["", "## Best-view shares", ""]
    for v, share in out["best_view_shares"].items():
        lines.append(f"- {v}: {share:.3f}")
    lines += [
        "",
        f"- Non-full cost-adjusted best share: {out['non_full_best_share']:.3f}",
        f"- Shallow-HNSW dominates full before cost: {out['shallow_dominates_full_precost_share']:.3f}",
        f"- Bootstrap 95% CI, non-full best share: "
        f"[{out['bootstrap_ci']['non_full_best_share']['lo']:.3f}, {out['bootstrap_ci']['non_full_best_share']['hi']:.3f}]",
    ]
    if router:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in router["choices"].items())
        lines += [
            "",
            "## Held-out random-forest HNSW-depth router",
            "",
            f"- Utility@10: {router['utility@10']:.3f}",
            f"- NDCG@10: {router['ndcg@10']:.3f}",
            f"- Cost: {router['cost']:.3f}",
            f"- Regret: {router['regret']:.3f}",
            f"- Choices: {ch}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
