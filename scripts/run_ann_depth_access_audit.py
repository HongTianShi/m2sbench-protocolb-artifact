#!/usr/bin/env python
"""ANN-native access-depth audit for standard IR collections.

This audit keeps the same query-document qrels as run_standard_ir_access_audit.py
but varies the IVF-PQ probe depth. It shows how Protocol B can encode a
classical ANN systems knob: buy a shallow compressed search, a deeper compressed
search, or full dense reranking.
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
    encode_or_load,
    eval_run,
    gpu_res,
    load_collection,
    per_query_ndcg,
    safe_id,
    score_features,
    search_flat,
    tree_router,
)


def build_ivfpq(doc_emb, nlist, m, nbits):
    index = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(doc_emb.shape[1]),
        doc_emb.shape[1],
        nlist,
        m,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    rng = np.random.default_rng(17)
    train_n = min(len(doc_emb), max(4096, nlist * 80))
    train = doc_emb[rng.choice(len(doc_emb), train_n, replace=False)]
    index.train(train)
    index.add(doc_emb)
    res = gpu_res()
    if res is not None:
        try:
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    return index


def search_levels(index, query_emb, k, nprobes):
    out = {}
    for nprobe in nprobes:
        index.nprobe = int(nprobe)
        t0 = time.perf_counter()
        scores, ids = index.search(query_emb, k)
        out[int(nprobe)] = (scores, ids, time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=512)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--nprobes", default="4,8,16,32")
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    nlist = min(args.nlist, max(8, len(docs) // 20))
    nprobes = [int(x) for x in args.nprobes.split(",") if x.strip()]
    index = build_ivfpq(doc_emb, nlist, args.m, args.nbits)
    level_runs = search_levels(index, query_emb, args.k, nprobes)
    scores_f, ids_f, t_f = search_flat(doc_emb, query_emb, args.k)

    views = {}
    per = {}
    util_per = {}
    costs = {}
    for nprobe, (scores, ids, elapsed) in level_runs.items():
        name = f"pq{nprobe}"
        # Monotone operation-unit proxy: fixed compressed-code setup plus a
        # probe-depth term, normalized so pq32 matches the main-table cost .20.
        cost = 0.05 + 0.15 * (nprobe / max(nprobes))
        costs[name] = cost
        m = eval_run(ids, rels)
        m["cost"] = cost
        m["utility@10"] = m["ndcg@10"] - args.lambda_cost * cost
        m["mean_us_per_query"] = elapsed / len(queries) * 1e6
        views[name] = m
        per[name] = per_query_ndcg(ids, rels, 10)
        util_per[name] = per[name] - args.lambda_cost * cost
    costs["full"] = 0.58
    m = eval_run(ids_f, rels)
    m["cost"] = 0.58
    m["utility@10"] = m["ndcg@10"] - args.lambda_cost * 0.58
    m["mean_us_per_query"] = t_f / len(queries) * 1e6
    views["full"] = m
    per["full"] = per_query_ndcg(ids_f, rels, 10)
    util_per["full"] = per["full"] - args.lambda_cost * 0.58

    names = list(views)
    stacked = np.stack([util_per[v] for v in names], axis=1)
    best_idx = np.argmax(stacked, axis=1)
    best_names = np.array(names)[best_idx]

    feat = []
    for nprobe in nprobes[: min(2, len(nprobes))]:
        feat.append(score_features(level_runs[nprobe][0]))
    feat = np.column_stack(feat)
    router = tree_router(feat, util_per, per, costs, args.lambda_cost)

    out = {
        "task": "ann_depth_access_audit",
        "dataset": args.dataset,
        "model": args.model,
        "faiss_gpus": int(faiss.get_num_gpus()),
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "nlist": int(nlist),
        "pq_m": args.m,
        "pq_nbits": args.nbits,
        "nprobes": nprobes,
        "lambda": args.lambda_cost,
        "views": views,
        "best_view_shares": {v: float(np.mean(best_names == v)) for v in names},
        "non_full_best_share": float(np.mean(best_names != "full")),
        "router": router,
    }
    stem = safe_id(args.dataset)
    json_path = REPORT_DIR / f"ann_depth_access_{stem}.json"
    md_path = REPORT_DIR / f"ann_depth_access_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        f"# ANN Depth Access Audit: {args.dataset}",
        "",
        f"- Documents: {len(docs):,}",
        f"- Queries with qrels: {len(queries):,}",
        f"- IVF/PQ: nlist={nlist}, m={args.m}, nbits={args.nbits}, nprobes={nprobes}",
        "",
        "| view | NDCG@10 | cost | utility@10 | us/query |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for v in names:
        m = views[v]
        lines.append(f"| {v} | {m['ndcg@10']:.3f} | {m['cost']:.3f} | {m['utility@10']:.3f} | {m['mean_us_per_query']:.2f} |")
    lines += ["", "## Best-view shares", ""]
    for v, share in out["best_view_shares"].items():
        lines.append(f"- {v}: {share:.3f}")
    if router:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in router["choices"].items())
        lines += [
            "",
            "## Held-out random-forest depth router",
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
