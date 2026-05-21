#!/usr/bin/env python
"""Standard IR collection access audit with sentence embeddings and FAISS.

This script instantiates the summary/PQ/full evidence-access menu on an
ir_datasets collection with ordinary query-document qrels. It is intended as a
compact IR-facing audit: a centroid-only summary view, an IVF-PQ compressed
view, and a full dense-vector view are compared under NDCG@10 and
cost-adjusted utility.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import faiss
import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "outputs" / "standard_ir_access_cache"


def text_of_doc(doc) -> str:
    parts = []
    for field in ["title", "text", "body", "abstract", "contents", "content", "description", "keywords"]:
        if hasattr(doc, field):
            val = getattr(doc, field)
            if val:
                parts.append(str(val))
    if not parts:
        parts.append(str(doc))
    return " ".join(parts)


def text_of_query(query) -> str:
    for field in ["text", "title", "query", "description"]:
        if hasattr(query, field):
            val = getattr(query, field)
            if val:
                return str(val)
    return str(query)


def safe_id(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def load_collection(dataset_id: str, max_docs: int, max_queries: int, seed: int):
    ds = ir_datasets.load(dataset_id)
    docs = list(ds.docs_iter())
    queries = list(ds.queries_iter())
    qrels = list(ds.qrels_iter())
    qrels_by_qid = defaultdict(dict)
    for qr in qrels:
        rel = float(getattr(qr, "relevance", 1))
        if rel > 0:
            qrels_by_qid[str(qr.query_id)][str(qr.doc_id)] = rel

    query_by_id = {str(q.query_id): q for q in queries}
    judged_qids = [qid for qid in query_by_id if qid in qrels_by_qid]
    rng = np.random.default_rng(seed)
    rng.shuffle(judged_qids)
    if max_queries and len(judged_qids) > max_queries:
        judged_qids = judged_qids[:max_queries]

    needed_doc_ids = {did for qid in judged_qids for did in qrels_by_qid[qid].keys()}
    docs_by_id = {str(d.doc_id): d for d in docs}
    doc_ids = list(docs_by_id.keys())
    if max_docs and len(doc_ids) > max_docs:
        nonrel = [did for did in doc_ids if did not in needed_doc_ids]
        rng.shuffle(nonrel)
        keep = list(needed_doc_ids) + nonrel[: max(0, max_docs - len(needed_doc_ids))]
        keep = [did for did in keep if did in docs_by_id]
        docs_by_id = {did: docs_by_id[did] for did in keep}

    kept_doc_ids = list(docs_by_id.keys())
    doc_index = {did: i for i, did in enumerate(kept_doc_ids)}
    q_texts, qids, rels = [], [], []
    for qid in judged_qids:
        filtered = {doc_index[did]: rel for did, rel in qrels_by_qid[qid].items() if did in doc_index}
        if not filtered:
            continue
        qids.append(qid)
        q_texts.append(text_of_query(query_by_id[qid]))
        rels.append(filtered)

    doc_texts = [text_of_doc(docs_by_id[did]) for did in kept_doc_ids]
    return doc_texts, kept_doc_ids, q_texts, qids, rels


def encode_or_load(dataset_id: str, model_name: str, docs: list[str], queries: list[str], batch_size: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = safe_id(dataset_id + "__" + model_name)
    d_path = CACHE_DIR / f"{safe}_docs_{len(docs)}.npy"
    q_path = CACHE_DIR / f"{safe}_queries_{len(queries)}.npy"
    model = SentenceTransformer(model_name, device="cuda")
    if d_path.exists():
        doc_emb = np.load(d_path)
    else:
        doc_emb = model.encode(docs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype("float32")
        np.save(d_path, doc_emb)
    if q_path.exists():
        query_emb = np.load(q_path)
    else:
        query_emb = model.encode(queries, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype("float32")
        np.save(q_path, query_emb)
    return doc_emb, query_emb


def gpu_res():
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        return faiss.StandardGpuResources()
    return None


def search_flat(doc_emb, query_emb, k):
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    res = gpu_res()
    if res is not None:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    t0 = time.perf_counter()
    scores, ids = index.search(query_emb, k)
    return scores, ids, time.perf_counter() - t0


def search_ivfpq(doc_emb, query_emb, k, nlist, m, nbits, nprobe):
    index = faiss.IndexIVFPQ(faiss.IndexFlatIP(doc_emb.shape[1]), doc_emb.shape[1], nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(17)
    train_n = min(len(doc_emb), max(4096, nlist * 80))
    train = doc_emb[rng.choice(len(doc_emb), train_n, replace=False)]
    index.train(train)
    index.add(doc_emb)
    index.nprobe = nprobe
    res = gpu_res()
    if res is not None:
        try:
            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.nprobe = nprobe
        except Exception:
            pass
    t0 = time.perf_counter()
    scores, ids = index.search(query_emb, k)
    return scores, ids, time.perf_counter() - t0


def search_centroid(doc_emb, query_emb, k, nlist):
    kmeans = faiss.Kmeans(doc_emb.shape[1], nlist, niter=25, verbose=False, spherical=True, gpu=faiss.get_num_gpus() > 0)
    kmeans.train(doc_emb)
    centroids = kmeans.centroids.astype("float32")
    assign_index = faiss.IndexFlatIP(doc_emb.shape[1])
    assign_index.add(centroids)
    _, assign = assign_index.search(doc_emb, 1)
    sims = (doc_emb * centroids[assign[:, 0]]).sum(axis=1)
    reps = {}
    for cid in range(nlist):
        members = np.where(assign[:, 0] == cid)[0]
        if len(members):
            reps[cid] = members[np.argsort(-sims[members])][: max(k, 8)].tolist()
    cindex = faiss.IndexFlatIP(doc_emb.shape[1])
    cindex.add(centroids)
    res = gpu_res()
    if res is not None:
        cindex = faiss.index_cpu_to_gpu(res, 0, cindex)
    t0 = time.perf_counter()
    cscores, cids = cindex.search(query_emb, min(nlist, max(32, k * 8)))
    elapsed = time.perf_counter() - t0
    out = np.full((len(query_emb), k), -1, dtype="int64")
    for qi, row in enumerate(cids):
        picked = []
        for cid in row:
            for did in reps.get(int(cid), []):
                if did not in picked:
                    picked.append(did)
                if len(picked) == k:
                    break
            if len(picked) == k:
                break
        out[qi, : len(picked)] = picked
    return cscores, out, elapsed


def dcg(vals):
    return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(vals))


def per_query_ndcg(ids, rels, k):
    vals = []
    for row, rel in zip(ids[:, :k], rels):
        gains = [rel.get(int(x), 0.0) for x in row if int(x) >= 0]
        gains += [0.0] * (k - len(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        ideal += [0.0] * (k - len(ideal))
        vals.append(dcg(gains) / (dcg(ideal) or 1.0))
    return np.array(vals, dtype="float32")


def eval_run(ids, rels):
    out = {}
    for k in [5, 10]:
        nd = per_query_ndcg(ids, rels, k)
        hit = []
        rec = []
        for row, rel in zip(ids[:, :k], rels):
            rel_ids = set(rel)
            got = [int(x) for x in row if int(x) in rel_ids]
            hit.append(1.0 if got else 0.0)
            rec.append(len(set(got)) / max(1, len(rel_ids)))
        out[f"ndcg@{k}"] = float(nd.mean())
        out[f"hit@{k}"] = float(np.mean(hit))
        out[f"recall@{k}"] = float(np.mean(rec))
    return out


def score_features(scores):
    if scores is None:
        return None
    s = np.asarray(scores[:, : min(10, scores.shape[1])], dtype="float32")
    if s.shape[1] == 1:
        margin = np.zeros(len(s), dtype="float32")
    else:
        margin = s[:, 0] - s[:, 1]
    centered = s - s.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    prob = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    entropy = -(prob * np.log(np.maximum(prob, 1e-8))).sum(axis=1)
    return np.column_stack([
        s[:, 0],
        margin,
        s.mean(axis=1),
        s.std(axis=1),
        s[:, 0] - s[:, -1],
        entropy,
    ]).astype("float32")


def ridge_router(features, util_per, per, ids_by, costs, lambda_cost, train_frac=0.6, alpha=1e-2):
    n = len(features)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    train = np.arange(cut)
    test = np.arange(cut, n)
    views = list(util_per.keys())
    x = np.column_stack([np.ones(n, dtype="float32"), features.astype("float32")])
    y = np.column_stack([util_per[v] for v in views]).astype("float32")
    xtx = x[train].T @ x[train]
    reg = alpha * np.eye(xtx.shape[0], dtype="float32")
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + reg, x[train].T @ y[train])
    pred = x[test] @ w
    choice_idx = np.argmax(pred, axis=1)
    choices = np.array(views)[choice_idx]
    chosen_ndcg = np.array([per[v][test[i]] for i, v in enumerate(choices)], dtype="float32")
    chosen_cost = np.array([costs[v] for v in choices], dtype="float32")
    chosen_util = chosen_ndcg - lambda_cost * chosen_cost
    oracle = np.max(y[test], axis=1)
    return {
        "test_queries": int(len(test)),
        "ndcg@10": float(chosen_ndcg.mean()),
        "cost": float(chosen_cost.mean()),
        "utility@10": float(chosen_util.mean()),
        "regret": float((oracle - chosen_util).mean()),
        "choices": {v: float(np.mean(choices == v)) for v in views},
    }


def tree_router(features, util_per, per, costs, lambda_cost, train_frac=0.6, seed=31):
    """Random-forest QPP router trained only on development qrels.

    The features are qrel-free at test time: first-stage score summaries and,
    depending on tier, optional compressed-view score summaries. This is a
    stronger baseline than ridge routing while preserving the Protocol B split
    between method-visible signals and evaluator-only qrels.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except Exception:
        return None

    n = len(features)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    train = np.arange(cut)
    test = np.arange(cut, n)
    views = list(util_per.keys())
    x = features.astype("float32")
    y = np.column_stack([util_per[v] for v in views]).astype("float32")
    base = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base)
    model.fit(x[train], y[train])
    pred = model.predict(x[test])
    choice_idx = np.argmax(pred, axis=1)
    choices = np.array(views)[choice_idx]
    chosen_ndcg = np.array([per[v][test[i]] for i, v in enumerate(choices)], dtype="float32")
    chosen_cost = np.array([costs[v] for v in choices], dtype="float32")
    chosen_util = chosen_ndcg - lambda_cost * chosen_cost
    oracle = np.max(y[test], axis=1)
    return {
        "test_queries": int(len(test)),
        "ndcg@10": float(chosen_ndcg.mean()),
        "cost": float(chosen_cost.mean()),
        "utility@10": float(chosen_util.mean()),
        "regret": float((oracle - chosen_util).mean()),
        "choices": {v: float(np.mean(choices == v)) for v in views},
        "model": "random_forest_qpp",
    }


def threshold_cascade_router(summary_features, pq_features, util_per, per, costs, lambda_cost, train_frac=0.6):
    """Grid-search a tiny selective-reranking cascade on train queries.

    The policy is intentionally simple and qrels-free at test time: stay with
    the summary if one summary score feature crosses a threshold; otherwise use
    PQ if one PQ score feature crosses a threshold; otherwise use full vectors.
    Thresholds are selected on train qrels and evaluated on held-out queries.
    """
    n = len(summary_features)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    train = np.arange(cut)
    test = np.arange(cut, n)
    names = ["top", "margin", "mean", "std", "range", "entropy"]
    best = None
    quant_s = np.linspace(0.1, 0.9, 17)
    quant_p = np.linspace(0.1, 0.9, 17)
    for sj, sname in enumerate(names):
        svals = np.unique(np.quantile(summary_features[train, sj], quant_s))
        for pj, pname in enumerate(names):
            pvals = np.unique(np.quantile(pq_features[train, pj], quant_p))
            for sthr in svals:
                for sop in ("ge", "le"):
                    smask = summary_features[:, sj] >= sthr if sop == "ge" else summary_features[:, sj] <= sthr
                    for pthr in pvals:
                        for pop in ("ge", "le"):
                            pmask = pq_features[:, pj] >= pthr if pop == "ge" else pq_features[:, pj] <= pthr
                            choices = np.full(n, "full", dtype=object)
                            choices[pmask] = "pq"
                            choices[smask] = "summary"
                            chosen_util = np.array([util_per[v][i] for i, v in enumerate(choices)], dtype="float32")
                            train_util = float(chosen_util[train].mean())
                            if best is None or train_util > best["train_utility@10"]:
                                chosen_ndcg = np.array([per[v][i] for i, v in enumerate(choices)], dtype="float32")
                                chosen_cost = np.array([costs[v] for v in choices], dtype="float32")
                                oracle = np.max(np.column_stack([util_per[v] for v in ["summary", "pq", "full"]])[test], axis=1)
                                best = {
                                    "train_utility@10": train_util,
                                    "test_queries": int(len(test)),
                                    "ndcg@10": float(chosen_ndcg[test].mean()),
                                    "cost": float(chosen_cost[test].mean()),
                                    "utility@10": float(chosen_util[test].mean()),
                                    "regret": float((oracle - chosen_util[test]).mean()),
                                    "choices": {v: float(np.mean(choices[test] == v)) for v in ["summary", "pq", "full"]},
                                    "rule": f"summary if {sname} {sop} {sthr:.4g}; else PQ if {pname} {pop} {pthr:.4g}; else full",
                                }
    return best


def query_features(query_emb, k=16):
    q = np.asarray(query_emb, dtype="float32")
    head = q[:, : min(k, q.shape[1])]
    stats = np.column_stack([
        q.mean(axis=1),
        q.std(axis=1),
        q.min(axis=1),
        q.max(axis=1),
        np.mean(q > 0, axis=1),
    ]).astype("float32")
    return np.column_stack([head, stats]).astype("float32")


def heldout_fixed_views(util_per, per, costs, lambda_cost, train_frac=0.6):
    n = len(util_per["summary"])
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    test = np.arange(cut, n)
    y = np.column_stack([util_per[v] for v in ["summary", "pq", "full"]])
    oracle = np.max(y[test], axis=1)
    out = {}
    for v in ["summary", "pq", "full"]:
        util = util_per[v][test]
        out[v] = {
            "test_queries": int(len(test)),
            "ndcg@10": float(per[v][test].mean()),
            "cost": float(costs[v]),
            "utility@10": float(util.mean()),
            "regret": float((oracle - util).mean()),
        }
    return out


def bootstrap_share(values, seed=23, reps=1000):
    arr = np.asarray(values, dtype="float32")
    if len(arr) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype="float32")
    n = len(arr)
    for i in range(reps):
        means[i] = arr[rng.integers(0, n, n)].mean()
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


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
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report-suffix", default="")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    nlist = min(args.nlist, max(8, len(docs) // 20))
    scores_s, ids_s, t_s = search_centroid(doc_emb, query_emb, args.k, nlist)
    scores_pq, ids_pq, t_pq = search_ivfpq(doc_emb, query_emb, args.k, nlist, args.m, args.nbits, args.nprobe)
    scores_f, ids_f, t_f = search_flat(doc_emb, query_emb, args.k)

    costs = {"summary": 0.0, "pq": 0.20, "full": 0.58}
    ids_by = {"summary": ids_s, "pq": ids_pq, "full": ids_f}
    times = {"summary": t_s, "pq": t_pq, "full": t_f}
    runs = {}
    per = {}
    util_per = {}
    for view, ids in ids_by.items():
        metrics = eval_run(ids, rels)
        metrics["cost"] = costs[view]
        metrics["utility@10"] = metrics["ndcg@10"] - args.lambda_cost * costs[view]
        metrics["mean_us_per_query"] = times[view] / len(queries) * 1e6
        runs[view] = metrics
        per[view] = per_query_ndcg(ids, rels, 10)
        util_per[view] = per[view] - args.lambda_cost * costs[view]

    feat_summary = score_features(scores_s)
    feat_pq = score_features(scores_pq)
    feat_query = query_features(query_emb)
    routers = {}
    if feat_summary is not None:
        routers["summary_qpp_b0"] = ridge_router(feat_summary, util_per, per, ids_by, costs, args.lambda_cost)
        routers["query_summary_rf_b0"] = tree_router(np.column_stack([feat_query, feat_summary]), util_per, per, costs, args.lambda_cost)
    if feat_summary is not None and feat_pq is not None:
        routers["summary_pq_qpp_b1"] = ridge_router(np.column_stack([feat_summary, feat_pq]), util_per, per, ids_by, costs, args.lambda_cost)
        routers["query_summary_pq_rf_b1"] = tree_router(np.column_stack([feat_query, feat_summary, feat_pq]), util_per, per, costs, args.lambda_cost)
        routers["selective_rerank_cascade_b1"] = threshold_cascade_router(feat_summary, feat_pq, util_per, per, costs, args.lambda_cost)
    routers = {k: v for k, v in routers.items() if v is not None}

    stacked = np.stack([util_per["summary"], util_per["pq"], util_per["full"]], axis=1)
    best = np.argmax(stacked, axis=1)
    names = np.array(["summary", "pq", "full"])
    best_names = names[best]
    diag = {
        "non_full_best_share": float(np.mean(best_names != "full")),
        "pq_dominates_full_precost_share": float(np.mean(per["pq"] >= per["full"])),
        "summary_best_share": float(np.mean(best_names == "summary")),
        "pq_best_share": float(np.mean(best_names == "pq")),
        "full_best_share": float(np.mean(best_names == "full")),
        "lambda_star_summary_to_pq": float((runs["pq"]["ndcg@10"] - runs["summary"]["ndcg@10"]) / 0.20),
        "lambda_star_pq_to_full": float((runs["full"]["ndcg@10"] - runs["pq"]["ndcg@10"]) / 0.38),
    }
    diag["bootstrap_ci"] = {
        "non_full_best_share": bootstrap_share(best_names != "full"),
        "pq_dominates_full_precost_share": bootstrap_share(per["pq"] >= per["full"]),
    }
    heldout = heldout_fixed_views(util_per, per, costs, args.lambda_cost)
    out = {
        "task": "standard_ir_access_audit",
        "dataset": args.dataset,
        "model": args.model,
        "faiss_gpus": int(faiss.get_num_gpus()),
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "nlist": int(nlist),
        "pq_m": args.m,
        "pq_nbits": args.nbits,
        "nprobe": args.nprobe,
        "lambda": args.lambda_cost,
        "views": runs,
        "diagnostics": diag,
        "heldout_fixed_views": heldout,
        "routers": routers,
    }
    stem = safe_id(args.dataset)
    if args.report_suffix:
        stem = f"{stem}_{safe_id(args.report_suffix)}"
    json_path = REPORT_DIR / f"standard_ir_access_{stem}.json"
    md_path = REPORT_DIR / f"standard_ir_access_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        f"# Standard IR Access Audit: {args.dataset}",
        "",
        f"- Documents: {len(docs):,}",
        f"- Queries with qrels: {len(queries):,}",
        f"- Positive/graded qrel pairs retained: {sum(len(r) for r in rels):,}",
        f"- Encoder: `{args.model}`",
        f"- FAISS GPUs visible: {faiss.get_num_gpus()}",
        f"- IVF/PQ: nlist={nlist}, m={args.m}, nbits={args.nbits}, nprobe={args.nprobe}",
        "",
        "| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in ["summary", "pq", "full"]:
        m = runs[v]
        lines.append(f"| {v} | {m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} | {m['recall@10']:.3f} | {m['hit@10']:.3f} | {m['cost']:.2f} | {m['utility@10']:.3f} | {m['mean_us_per_query']:.2f} |")
    lines += [
        "",
        "## Access diagnostics",
        "",
        f"- Non-full cost-adjusted best share: {diag['non_full_best_share']:.3f}",
        f"- PQ dominates full before cost share: {diag['pq_dominates_full_precost_share']:.3f}",
        f"- Best-view shares: summary {diag['summary_best_share']:.3f}, PQ {diag['pq_best_share']:.3f}, full {diag['full_best_share']:.3f}",
        f"- Break-even lambda*: summary->PQ {diag['lambda_star_summary_to_pq']:.3f}, PQ->full {diag['lambda_star_pq_to_full']:.3f}",
        f"- Bootstrap 95% CI, non-full best share: [{diag['bootstrap_ci']['non_full_best_share']['lo']:.3f}, {diag['bootstrap_ci']['non_full_best_share']['hi']:.3f}]",
        f"- Bootstrap 95% CI, PQ dominates full before cost: [{diag['bootstrap_ci']['pq_dominates_full_precost_share']['lo']:.3f}, {diag['bootstrap_ci']['pq_dominates_full_precost_share']['hi']:.3f}]",
        "",
        "## Held-out fixed-view baselines",
        "",
        "| view | test queries | NDCG@10 | cost | utility@10 | regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in ["summary", "pq", "full"]:
        r = heldout[v]
        lines.append(f"| {v} | {r['test_queries']} | {r['ndcg@10']:.3f} | {r['cost']:.3f} | {r['utility@10']:.3f} | {r['regret']:.3f} |")
    if routers:
        lines += [
            "",
            "## Held-out QPP/access routers",
            "",
            "| router | test queries | NDCG@10 | cost | utility@10 | regret | choices |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, r in routers.items():
            ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items())
            rule = f"; {r['rule']}" if "rule" in r else ""
            lines.append(f"| {name} | {r['test_queries']} | {r['ndcg@10']:.3f} | {r['cost']:.3f} | {r['utility@10']:.3f} | {r['regret']:.3f} | {ch}{rule} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
