#!/usr/bin/env python
"""Cascade-style HNSW depth routing audit.

This is a stronger systems baseline than a query-only router.  It first runs a
cheap HNSW search, inspects only the returned scores/ids, and then decides
whether to keep that shallow result or pay for a deeper HNSW/full view.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

import faiss
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    eval_run,
    load_collection,
    per_query_ndcg,
    query_features,
    safe_id,
    score_features,
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


def overlap_features(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    feats = []
    for a, b in zip(ids_a, ids_b):
        sa = set(int(x) for x in a if int(x) >= 0)
        sb = set(int(x) for x in b if int(x) >= 0)
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        feats.append(
            [
                float(inter) / max(1, len(sa)),
                float(inter) / union,
                float(a[0] == b[0]),
                float(np.mean(a == b)),
            ]
        )
    return np.asarray(feats, dtype="float32")


def summarize_view(ids, rels, cost, lam, per_us):
    m = eval_run(ids, rels)
    m["cost"] = float(cost)
    m["utility@10"] = float(m["ndcg@10"] - lam * cost)
    m["mean_us_per_query"] = float(np.mean(per_us))
    m["p95_us_per_query"] = float(np.quantile(per_us, 0.95))
    return m


def route_eval(name, choices, names, per, util_per, costs, lam, test_idx, oracle):
    chosen_u = np.array([util_per[c][i] for c, i in zip(choices, test_idx)], dtype="float32")
    chosen_nd = np.array([per[c][i] for c, i in zip(choices, test_idx)], dtype="float32")
    chosen_cost = np.array([costs[c] for c in choices], dtype="float32")
    out = {
        "name": name,
        "test_queries": int(len(test_idx)),
        "utility@10": float(chosen_u.mean()),
        "ndcg@10": float(chosen_nd.mean()),
        "cost": float(chosen_cost.mean()),
        "regret": float((oracle - chosen_u).mean()),
        "choices": {n: float(np.mean(np.asarray(choices) == n)) for n in names},
    }
    return out


def train_reg_router(model, X, util_per, per, costs, lam, train_idx, test_idx, names):
    preds = []
    for n in names:
        m = model
        # sklearn estimators are cloned by parameter reconstruction for the few
        # model types used here.
        if isinstance(model, RandomForestRegressor):
            m = RandomForestRegressor(**model.get_params())
        elif isinstance(model, ExtraTreesRegressor):
            m = ExtraTreesRegressor(**model.get_params())
        elif isinstance(model, HistGradientBoostingRegressor):
            m = HistGradientBoostingRegressor(**model.get_params())
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X[train_idx], util_per[n][train_idx])
        preds.append(m.predict(X[test_idx]))
    pred = np.stack(preds, axis=1)
    choices = np.asarray(names)[np.argmax(pred, axis=1)]
    oracle = np.max(np.stack([util_per[n][test_idx] for n in names], axis=1), axis=1)
    return route_eval(type(model).__name__, choices, names, per, util_per, costs, lam, test_idx, oracle)


def model_factory(name: str, seed: int):
    if name == "rf":
        return RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.05, random_state=seed)
    return Ridge(alpha=1.0)


def threshold_gate(X_gate, util_per, per, costs, lam, train_idx, test_idx, names):
    # Choose shallow if a visible agreement score exceeds a trained threshold;
    # otherwise choose the best deeper fixed view on the training split.
    shallow, deep = names[0], names[1]
    best = None
    for col, label in [(0, "overlap"), (1, "jaccard"), (2, "top1_same"), (3, "same_rank")]:
        vals = X_gate[train_idx, col]
        grid = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 19)))
        for thr in grid:
            train_choices = np.where(X_gate[train_idx, col] >= thr, shallow, deep)
            train_u = np.array([util_per[c][i] for c, i in zip(train_choices, train_idx)], dtype="float32").mean()
            if best is None or train_u > best[0]:
                best = (train_u, col, float(thr), label)
    _, col, thr, label = best
    choices = np.where(X_gate[test_idx, col] >= thr, shallow, deep)
    oracle = np.max(np.stack([util_per[n][test_idx] for n in names], axis=1), axis=1)
    out = route_eval(f"agreement_gate_{label}", choices, names, per, util_per, costs, lam, test_idx, oracle)
    out["rule"] = f"{label}>={thr:.3f}: {shallow}, else {deep}"
    return out


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
    ap.add_argument("--split-reps", type=int, default=20)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    ef_levels = [int(x) for x in args.ef_search.split(",") if x.strip()]
    index = build_hnsw(doc_emb, args.hnsw_m, args.ef_construction)

    scores_by, ids_by, per_us_by = {}, {}, {}
    for ef in ef_levels:
        scores, ids, per_us = timed_search(index, query_emb, args.k, ef)
        name = f"hnsw{ef}"
        scores_by[name], ids_by[name], per_us_by[name] = scores, ids, per_us
    scores_f, ids_f, per_us_f = timed_flat_cpu(doc_emb, query_emb, args.k)
    scores_by["full"], ids_by["full"], per_us_by["full"] = scores_f, ids_f, per_us_f

    max_ef = max(ef_levels)
    costs = {f"hnsw{ef}": 0.10 + 0.30 * (ef / max_ef) for ef in ef_levels}
    costs["full"] = 0.58
    names = [f"hnsw{ef}" for ef in ef_levels] + ["full"]

    views, per, util_per = {}, {}, {}
    for n in names:
        views[n] = summarize_view(ids_by[n], rels, costs[n], args.lambda_cost, per_us_by[n])
        per[n] = per_query_ndcg(ids_by[n], rels, 10)
        util_per[n] = per[n] - args.lambda_cost * costs[n]

    feat_parts = [query_features(query_emb)]
    for n in names[:-1]:
        feat_parts.append(score_features(scores_by[n]))
    for a, b in zip(names[:-2], names[1:-1]):
        feat_parts.append(overlap_features(ids_by[a], ids_by[b]))
    X = np.column_stack(feat_parts).astype("float32")
    gate = overlap_features(ids_by[names[0]], ids_by[names[1]])

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(queries))
    rng.shuffle(idx)
    cut = int(round(len(idx) * 0.60))
    train_idx, test_idx = idx[:cut], idx[cut:]

    oracle = np.max(np.stack([util_per[n][test_idx] for n in names], axis=1), axis=1)
    routes = []
    for n in names:
        choices = np.asarray([n] * len(test_idx))
        routes.append(route_eval(f"fixed_{n}", choices, names, per, util_per, costs, args.lambda_cost, test_idx, oracle))
    routes.append(threshold_gate(gate, util_per, per, costs, args.lambda_cost, train_idx, test_idx, names[:2]))
    for key in ["rf", "et", "hgb", "ridge"]:
        routes.append(train_reg_router(model_factory(key, args.seed), X, util_per, per, costs, args.lambda_cost, train_idx, test_idx, names))

    stacked = np.stack([util_per[n] for n in names], axis=1)
    best_names = np.asarray(names)[np.argmax(stacked, axis=1)]
    best_fixed = max((r for r in routes if r["name"].startswith("fixed_")), key=lambda r: r["utility@10"])
    for r in routes:
        diff = np.array([util_per[np.asarray(names)[np.argmax(stacked[i])]][i] for i in test_idx], dtype="float32")
        r["oracle_gap"] = float(diff.mean() - r["utility@10"])
        r["diff_vs_best_fixed"] = float(r["utility@10"] - best_fixed["utility@10"])

    split_rows = []
    for rep in range(args.split_reps):
        rrng = np.random.default_rng(args.seed + 1000 + rep)
        ridx = np.arange(len(queries))
        rrng.shuffle(ridx)
        rcut = int(round(len(ridx) * 0.60))
        tr, te = ridx[:rcut], ridx[rcut:]
        oracle_rep = np.max(np.stack([util_per[n][te] for n in names], axis=1), axis=1)
        fixed_routes = []
        for n in names:
            fixed_routes.append(route_eval(f"fixed_{n}", np.asarray([n] * len(te)), names, per, util_per, costs, args.lambda_cost, te, oracle_rep))
        best_fixed_rep = max(fixed_routes, key=lambda r: r["utility@10"])
        rep_routes = fixed_routes + [
            threshold_gate(gate, util_per, per, costs, args.lambda_cost, tr, te, names[:2]),
            train_reg_router(model_factory("et", args.seed + rep), X, util_per, per, costs, args.lambda_cost, tr, te, names),
            train_reg_router(model_factory("ridge", args.seed + rep), X, util_per, per, costs, args.lambda_cost, tr, te, names),
        ]
        best_adaptive = max((r for r in rep_routes if not r["name"].startswith("fixed_")), key=lambda r: r["utility@10"])
        split_rows.append({
            "rep": rep,
            "test_queries": int(len(te)),
            "best_fixed": best_fixed_rep["name"],
            "best_fixed_utility": best_fixed_rep["utility@10"],
            "best_adaptive": best_adaptive["name"],
            "best_adaptive_utility": best_adaptive["utility@10"],
            "adaptive_minus_best_fixed": best_adaptive["utility@10"] - best_fixed_rep["utility@10"],
        })

    if split_rows:
        diffs = np.array([r["adaptive_minus_best_fixed"] for r in split_rows], dtype="float32")
        split_summary = {
            "reps": int(len(split_rows)),
            "mean_adaptive_minus_best_fixed": float(diffs.mean()),
            "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)),
            "positive_share": float(np.mean(diffs > 0)),
        }
    else:
        split_summary = {}

    out = {
        "task": "hnsw_cascade_router_audit",
        "dataset": args.dataset,
        "model": args.model,
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "ef_search": ef_levels,
        "lambda": args.lambda_cost,
        "views": views,
        "best_view_shares": {n: float(np.mean(best_names == n)) for n in names},
        "test_queries": int(len(test_idx)),
        "routes": routes,
        "split_replicates": split_rows,
        "split_summary": split_summary,
    }

    stem = safe_id(args.dataset)
    json_path = REPORT_DIR / f"hnsw_cascade_router_{stem}.json"
    md_path = REPORT_DIR / f"hnsw_cascade_router_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        f"# HNSW Cascade Router Audit: {args.dataset}",
        "",
        f"- Documents: {len(docs):,}",
        f"- Queries with qrels: {len(queries):,}",
        f"- efSearch menu: {ef_levels} plus full",
        "",
        "| route | utility@10 | NDCG@10 | cost | regret | diff vs best fixed | choices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in sorted(routes, key=lambda x: x["utility@10"], reverse=True):
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items() if v)
        rule = f"; {r['rule']}" if "rule" in r else ""
        lines.append(
            f"| {r['name']} | {r['utility@10']:.3f} | {r['ndcg@10']:.3f} | {r['cost']:.3f} | "
            f"{r['regret']:.3f} | {r['diff_vs_best_fixed']:+.3f} | {ch}{rule} |"
        )
    if split_summary:
        lines += [
            "",
            "## Split stability",
            "",
            f"- Repeated 60/40 splits: {split_summary['reps']}",
            f"- Best adaptive minus best fixed utility: {split_summary['mean_adaptive_minus_best_fixed']:+.4f} "
            f"[{split_summary['lo']:+.4f}, {split_summary['hi']:+.4f}]",
            f"- Positive split share: {split_summary['positive_share']:.3f}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:5000])


if __name__ == "__main__":
    main()
