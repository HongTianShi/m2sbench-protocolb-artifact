#!/usr/bin/env python
"""GPU FAISS dense-evidence access audit on DocRED exact qrels.

This script instantiates the Protocol B access menu on a real dense retrieval
task:

* corpus items are DocRED sentences;
* queries are (document title, head entity, tail entity, relation) tuples;
* qrels are DocRED evidence sentence annotations;
* summary access searches IVF centroids only;
* PQ access searches a compressed IVF-PQ index;
* full access searches full sentence embeddings.

The audit is intentionally small enough for a laptop GPU but uses the same
retrieval objects that IR reviewers expect: embeddings, FAISS indexes, exact
qrels, NDCG/recall, and latency/cost records.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCRED = Path(os.environ.get("M2SBENCH_DOCRED_DIR", ROOT / "external_data" / "DocRED"))
REPORT_DIR = ROOT / "reports"
OUT_JSON = REPORT_DIR / "gpu_faiss_docred_evidence_audit.json"
OUT_MD = REPORT_DIR / "gpu_faiss_docred_evidence_audit.md"
CACHE_DIR = ROOT / "outputs" / "gpu_faiss_docred_cache"


def _entity_name(vertex_set: list, idx: int) -> str:
    mentions = vertex_set[idx] if 0 <= idx < len(vertex_set) else []
    if not mentions:
        return f"entity_{idx}"
    return mentions[0].get("name") or mentions[0].get("sent_id", f"entity_{idx}")


def _sentence_text(tokens: Iterable[str]) -> str:
    return " ".join(str(t) for t in tokens)


def load_docred(docred_dir: Path, max_queries: int, seed: int, max_distractor_sentences: int):
    rel_info = {}
    rel_path = docred_dir / "rel_info.json"
    if rel_path.exists():
        rel_info = json.loads(rel_path.read_text(encoding="utf-8"))

    corpus: list[str] = []
    corpus_meta: list[dict] = []
    queries: list[str] = []
    qrels: list[list[int]] = []
    query_meta: list[dict] = []

    for split_name in ["train_annotated.json", "dev.json"]:
        data = json.loads((docred_dir / split_name).read_text(encoding="utf-8"))
        for doc_idx, doc in enumerate(data):
            title = doc.get("title", f"{split_name}:{doc_idx}")
            sent_ids = []
            for sent_idx, sent in enumerate(doc.get("sents", [])):
                sent_ids.append(len(corpus))
                corpus.append(f"{title}. {_sentence_text(sent)}")
                corpus_meta.append({"split": split_name, "title": title, "sent_idx": sent_idx})
            vertex_set = doc.get("vertexSet", [])
            for label_idx, label in enumerate(doc.get("labels", [])):
                evidence = sorted(set(int(i) for i in label.get("evidence", []) if 0 <= int(i) < len(sent_ids)))
                if not evidence:
                    continue
                h = _entity_name(vertex_set, int(label.get("h", -1)))
                t = _entity_name(vertex_set, int(label.get("t", -1)))
                rel_id = label.get("r", "")
                rel_desc = rel_info.get(rel_id, rel_id)
                queries.append(f"{title}. relation: {rel_desc}. head entity: {h}. tail entity: {t}.")
                qrels.append([sent_ids[i] for i in evidence])
                query_meta.append(
                    {
                        "split": split_name,
                        "title": title,
                        "label_idx": label_idx,
                        "relation": rel_id,
                        "n_evidence": len(evidence),
                    }
                )

    if max_distractor_sentences > 0:
        distant_path = docred_dir / "train_distant.json"
        if distant_path.exists():
            distant = json.loads(distant_path.read_text(encoding="utf-8"))
            doc_order = list(range(len(distant)))
            random.Random(seed).shuffle(doc_order)
            added = 0
            seen = {(m["title"], m["sent_idx"]) for m in corpus_meta}
            for di in doc_order:
                doc = distant[di]
                title = doc.get("title", f"train_distant:{di}")
                for sent_idx, sent in enumerate(doc.get("sents", [])):
                    key = (title, sent_idx)
                    if key in seen:
                        continue
                    corpus.append(f"{title}. {_sentence_text(sent)}")
                    corpus_meta.append({"split": "train_distant", "title": title, "sent_idx": sent_idx})
                    added += 1
                    if added >= max_distractor_sentences:
                        break
                if added >= max_distractor_sentences:
                    break

    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    if max_queries and len(order) > max_queries:
        order = order[:max_queries]
    queries = [queries[i] for i in order]
    qrels = [qrels[i] for i in order]
    query_meta = [query_meta[i] for i in order]
    return corpus, corpus_meta, queries, qrels, query_meta


def encode_or_load(model_name: str, corpus: list[str], queries: list[str], batch_size: int, seed: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "__")
    c_path = CACHE_DIR / f"{safe}_corpus_{len(corpus)}.npy"
    q_path = CACHE_DIR / f"{safe}_queries_{len(queries)}_seed{seed}.npy"
    model = SentenceTransformer(model_name, device="cuda")
    if c_path.exists():
        corpus_emb = np.load(c_path)
    else:
        corpus_emb = model.encode(
            corpus,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        np.save(c_path, corpus_emb)
    if q_path.exists():
        query_emb = np.load(q_path)
    else:
        query_emb = model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        np.save(q_path, query_emb)
    return corpus_emb, query_emb


def make_gpu_resources():
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        return faiss.StandardGpuResources()
    return None


def search_flat(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int):
    d = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus_emb)
    res = make_gpu_resources()
    if res is not None:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    t0 = time.perf_counter()
    scores, ids = index.search(query_emb, k)
    elapsed = time.perf_counter() - t0
    return scores, ids, elapsed


def search_ivfpq(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int, nlist: int, m: int, nbits: int, nprobe: int):
    d = corpus_emb.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    train_n = min(len(corpus_emb), max(4096, nlist * 80))
    rng = np.random.default_rng(7)
    train = corpus_emb[rng.choice(len(corpus_emb), size=train_n, replace=False)]
    index.train(train)
    index.add(corpus_emb)
    index.nprobe = nprobe
    res = make_gpu_resources()
    if res is not None:
        try:
            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.nprobe = nprobe
        except Exception:
            pass
    t0 = time.perf_counter()
    scores, ids = index.search(query_emb, k)
    elapsed = time.perf_counter() - t0
    return scores, ids, elapsed


def search_summary(corpus_emb: np.ndarray, query_emb: np.ndarray, k: int, nlist: int):
    d = corpus_emb.shape[1]
    kmeans = faiss.Kmeans(d, nlist, niter=25, verbose=False, gpu=faiss.get_num_gpus() > 0, spherical=True)
    kmeans.train(corpus_emb)
    centroids = kmeans.centroids.astype("float32")
    assign_index = faiss.IndexFlatIP(d)
    assign_index.add(centroids)
    _, assign = assign_index.search(corpus_emb, 1)
    reps: dict[int, list[int]] = defaultdict(list)
    sims = (corpus_emb * centroids[assign[:, 0]]).sum(axis=1)
    cluster_sizes = np.bincount(assign[:, 0], minlength=nlist).astype("float32")
    cluster_resid = np.zeros(nlist, dtype="float32")
    for cid in range(nlist):
        members = np.where(assign[:, 0] == cid)[0]
        if len(members) == 0:
            continue
        cluster_resid[cid] = float(np.mean(1.0 - sims[members]))
        members = members[np.argsort(-sims[members])]
        reps[cid] = members[: max(k, 4)].tolist()
    centroid_index = faiss.IndexFlatIP(d)
    centroid_index.add(centroids)
    res = make_gpu_resources()
    if res is not None:
        centroid_index = faiss.index_cpu_to_gpu(res, 0, centroid_index)
    t0 = time.perf_counter()
    centroid_scores, cids = centroid_index.search(query_emb, min(nlist, max(32, k * 8)))
    elapsed = time.perf_counter() - t0
    top_cid = cids[:, 0].astype("int64")
    top_score = centroid_scores[:, 0]
    second_score = centroid_scores[:, 1] if centroid_scores.shape[1] > 1 else centroid_scores[:, 0]
    features = np.column_stack(
        [
            top_score,
            top_score - second_score,
            np.log1p(cluster_sizes[top_cid]),
            cluster_resid[top_cid],
        ]
    ).astype("float32")
    out = np.full((len(query_emb), k), -1, dtype="int64")
    for qi, row in enumerate(cids):
        picked = []
        for cid in row:
            for doc_id in reps.get(int(cid), []):
                if doc_id not in picked:
                    picked.append(doc_id)
                if len(picked) == k:
                    break
            if len(picked) == k:
                break
        out[qi, : len(picked)] = picked
    return features, out, elapsed


def dcg(binary: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(binary))


def evaluate(ids: np.ndarray, qrels: list[list[int]], k_values=(5, 10)):
    out = {}
    qrel_sets = [set(x) for x in qrels]
    for k in k_values:
        ndcgs = []
        hits = []
        recalls = []
        for row, rels in zip(ids[:, :k], qrel_sets):
            binary = [1 if int(x) in rels else 0 for x in row if int(x) >= 0]
            binary += [0] * (k - len(binary))
            ideal = [1] * min(len(rels), k) + [0] * max(0, k - len(rels))
            idcg = dcg(ideal) or 1.0
            ndcgs.append(dcg(binary) / idcg)
            hits.append(1.0 if any(binary) else 0.0)
            recalls.append(sum(binary) / max(1, len(rels)))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs))
        out[f"hit@{k}"] = float(np.mean(hits))
        out[f"recall@{k}"] = float(np.mean(recalls))
    return out


def per_query_ndcg(ids: np.ndarray, qrels: list[list[int]], k: int):
    vals = []
    qrel_sets = [set(x) for x in qrels]
    for row, rels in zip(ids[:, :k], qrel_sets):
        binary = [1 if int(x) in rels else 0 for x in row if int(x) >= 0]
        binary += [0] * (k - len(binary))
        ideal = [1] * min(len(rels), k) + [0] * max(0, k - len(rels))
        vals.append(dcg(binary) / (dcg(ideal) or 1.0))
    return np.array(vals, dtype="float32")


def score_features(scores: np.ndarray) -> np.ndarray:
    top = scores[:, 0]
    second = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
    mean_top = np.mean(scores[:, : min(5, scores.shape[1])], axis=1)
    return np.column_stack([top, top - second, mean_top]).astype("float32")


def ridge_router_report(
    features: np.ndarray,
    per: dict[str, np.ndarray],
    utility_per: dict[str, np.ndarray],
    costs: dict[str, float],
    views: list[str],
    seed: int,
    name: str,
) -> dict:
    n = len(features)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = int(0.67 * n)
    train = order[:split]
    test = order[split:]

    x_train = features[train].astype("float64")
    x_test = features[test].astype("float64")
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])

    y_train = np.column_stack([utility_per[v][train] for v in views]).astype("float64")
    alpha = 1e-3
    xtx = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    w = np.linalg.solve(xtx, x_train.T @ y_train)
    pred = x_test @ w
    choice = np.argmax(pred, axis=1)
    chosen = np.array(views)[choice]

    actual_utility = np.array([utility_per[v][test[i]] for i, v in enumerate(chosen)], dtype="float64")
    actual_ndcg = np.array([per[v][test[i]] for i, v in enumerate(chosen)], dtype="float64")
    actual_cost = np.array([costs[v] for v in chosen], dtype="float64")
    oracle = np.max(np.column_stack([utility_per[v][test] for v in views]), axis=1)

    report = {
        "name": name,
        "views": views,
        "train_queries": int(len(train)),
        "test_queries": int(len(test)),
        "utility@10": float(np.mean(actual_utility)),
        "ndcg@10": float(np.mean(actual_ndcg)),
        "cost": float(np.mean(actual_cost)),
        "oracle_utility@10": float(np.mean(oracle)),
        "regret@10": float(np.mean(oracle - actual_utility)),
        "choice_share": {v: float(np.mean(chosen == v)) for v in views},
    }
    for v in views:
        report[f"fixed_{v}_utility@10_on_test"] = float(np.mean(utility_per[v][test]))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docred-dir", type=Path, default=DEFAULT_DOCRED)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-queries", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-distractor-sentences", type=int, default=0)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    corpus, corpus_meta, queries, qrels, query_meta = load_docred(
        args.docred_dir,
        args.max_queries,
        args.seed,
        args.max_distractor_sentences,
    )
    corpus_emb, query_emb = encode_or_load(args.model, corpus, queries, args.batch_size, args.seed)

    runs = {}
    feat_s, ids_s, t_s = search_summary(corpus_emb, query_emb, args.k, args.nlist)
    scores_pq, ids_pq, t_pq = search_ivfpq(corpus_emb, query_emb, args.k, args.nlist, args.m, args.nbits, args.nprobe)
    scores_full, ids_full, t_full = search_flat(corpus_emb, query_emb, args.k)

    costs = {"summary": 0.0, "pq": 0.20, "full": 0.58}
    ids_by_view = {"summary": ids_s, "pq": ids_pq, "full": ids_full}
    elapsed = {"summary": t_s, "pq": t_pq, "full": t_full}
    for view, ids in ids_by_view.items():
        metrics = evaluate(ids, qrels, k_values=(5, 10))
        metrics["cost"] = costs[view]
        metrics["utility@10"] = metrics["ndcg@10"] - args.lambda_cost * costs[view]
        metrics["mean_us_per_query"] = elapsed[view] / len(queries) * 1e6
        runs[view] = metrics

    per = {view: per_query_ndcg(ids, qrels, 10) for view, ids in ids_by_view.items()}
    utility_per = {view: vals - args.lambda_cost * costs[view] for view, vals in per.items()}
    best = np.argmax(np.stack([utility_per["summary"], utility_per["pq"], utility_per["full"]], axis=1), axis=1)
    best_names = np.array(["summary", "pq", "full"])[best]
    diag = {
        "non_full_best_share": float(np.mean(best_names != "full")),
        "pq_dominates_full_precost_share": float(np.mean(per["pq"] >= per["full"])),
        "summary_best_share": float(np.mean(best_names == "summary")),
        "pq_best_share": float(np.mean(best_names == "pq")),
        "full_best_share": float(np.mean(best_names == "full")),
        "lambda_star_summary_to_pq": float((runs["pq"]["ndcg@10"] - runs["summary"]["ndcg@10"]) / (costs["pq"] - costs["summary"])),
        "lambda_star_pq_to_full": float((runs["full"]["ndcg@10"] - runs["pq"]["ndcg@10"]) / (costs["full"] - costs["pq"])),
    }
    feat_b0 = feat_s
    feat_b1 = np.column_stack([feat_s, score_features(scores_pq)]).astype("float32")
    routers = {
        "summary_visible_proxy_b0": ridge_router_report(
            feat_b0,
            per,
            utility_per,
            costs,
            ["summary", "pq", "full"],
            args.seed,
            "summary-visible proxy router (B0)",
        ),
        "pq_aware_proxy_b1": ridge_router_report(
            feat_b1,
            per,
            utility_per,
            costs,
            ["pq", "full"],
            args.seed + 1,
            "PQ-aware proxy router (B1)",
        ),
    }

    payload = {
        "task": "gpu_faiss_docred_evidence_access",
        "model": args.model,
        "faiss_gpus": int(faiss.get_num_gpus()),
        "corpus_sentences": len(corpus),
        "queries": len(queries),
        "distractor_sentences_requested": args.max_distractor_sentences,
        "qrel_type": "DocRED exact evidence sentence annotations",
        "nlist": args.nlist,
        "pq_m": args.m,
        "pq_nbits": args.nbits,
        "nprobe": args.nprobe,
        "lambda": args.lambda_cost,
        "views": runs,
        "diagnostics": diag,
        "routers": routers,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# GPU FAISS DocRED Evidence Access Audit",
        "",
        "Dense exact-qrel evidence retrieval audit using sentence-transformer embeddings and FAISS.",
        "",
        f"- Corpus sentences: {len(corpus):,}",
        f"- Queries: {len(queries):,}",
        f"- Extra train-distant distractors requested: {args.max_distractor_sentences:,}",
        f"- Qrels: DocRED evidence sentence annotations",
        f"- Encoder: `{args.model}`",
        f"- FAISS GPUs visible: {faiss.get_num_gpus()}",
        f"- IVF/PQ: nlist={args.nlist}, m={args.m}, nbits={args.nbits}, nprobe={args.nprobe}",
        "",
        "| view | NDCG@5 | NDCG@10 | Recall@10 | Hit@10 | cost | utility@10 | us/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in ["summary", "pq", "full"]:
        m = runs[view]
        lines.append(
            f"| {view} | {m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} | {m['recall@10']:.3f} | "
            f"{m['hit@10']:.3f} | {m['cost']:.2f} | {m['utility@10']:.3f} | {m['mean_us_per_query']:.2f} |"
        )
    lines += [
        "",
        "## Access diagnostics",
        "",
        f"- Non-full cost-adjusted best share: {diag['non_full_best_share']:.3f}",
        f"- PQ dominates full before cost share: {diag['pq_dominates_full_precost_share']:.3f}",
        f"- Best-view shares: summary {diag['summary_best_share']:.3f}, PQ {diag['pq_best_share']:.3f}, full {diag['full_best_share']:.3f}",
        f"- Break-even lambda*: summary->PQ {diag['lambda_star_summary_to_pq']:.3f}, PQ->full {diag['lambda_star_pq_to_full']:.3f}",
        "",
        "## Held-out adaptive routers",
        "",
        "| router | test queries | NDCG@10 | cost | utility@10 | regret@10 | choice shares |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ["summary_visible_proxy_b0", "pq_aware_proxy_b1"]:
        r = routers[key]
        shares = ", ".join(f"{v} {s:.2f}" for v, s in r["choice_share"].items())
        lines.append(
            f"| {r['name']} | {r['test_queries']:,} | {r['ndcg@10']:.3f} | {r['cost']:.3f} | "
            f"{r['utility@10']:.3f} | {r['regret@10']:.3f} | {shares} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)
    print(json.dumps(payload, indent=2)[:4000])


if __name__ == "__main__":
    main()
