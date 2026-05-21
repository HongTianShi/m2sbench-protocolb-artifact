#!/usr/bin/env python
"""Compression-aware evidence ladder audit for Protocol B.

The standard semantic audit compares summary, IVF-PQ, and full dense-vector
views. This script expands that menu into a QAMA-style compression ladder:
binary sign codes, IVF-PQ codes, truncated dense vectors, int8 vectors, full
dense vectors, and an optional cross-encoder view from the saved CE boundary
trace. The goal is not to propose a new compressed encoder; it is to make the
benchmark contract ask which compression level is worth buying.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    eval_run,
    load_collection,
    encode_or_load,
    per_query_ndcg,
    safe_id,
    search_centroid,
    search_flat,
    search_ivfpq,
)


def _renorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def _truncated(x: np.ndarray, dim: int) -> np.ndarray:
    return _renorm(x[:, : min(dim, x.shape[1])])


def _int8_view(x: np.ndarray) -> np.ndarray:
    # Symmetric post-training quantization for an access-view audit. We
    # dequantize before FAISS search so the measured implementation remains
    # portable across CPU/GPU FAISS builds.
    q = np.clip(np.rint(x * 127.0), -127, 127).astype("int8")
    return _renorm(q.astype("float32") / 127.0)


def _binary_sign_view(x: np.ndarray) -> np.ndarray:
    signs = np.where(x >= 0, 1.0, -1.0).astype("float32")
    return signs / math.sqrt(signs.shape[1])


def _load_ce_boundary(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["query_index"])] = float(row["cross_ndcg"])
    return out


def _load_ce_latency(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["views"]["cross_encoder"]["mean_us_per_query"])


def _summarize_view(
    name: str,
    per: np.ndarray,
    idx: np.ndarray,
    cost: float,
    mean_us: float,
    bytes_touched: str,
    lambda_cost: float,
) -> dict[str, float | str]:
    raw = per[idx]
    return {
        "view": name,
        "bytes_touched": bytes_touched,
        "latency_us": float(mean_us),
        "ndcg@10": float(raw.mean()),
        "cost": float(cost),
        "utility@10": float((raw - lambda_cost * cost).mean()),
    }


def main() -> None:
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
    ap.add_argument("--ce-boundary", type=Path, default=REPORT_DIR / "cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.csv")
    ap.add_argument("--ce-report", type=Path, default=REPORT_DIR / "cross_encoder_reranker_access_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.json")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, _, queries, _, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    dim = int(doc_emb.shape[1])
    nlist = min(args.nlist, max(8, len(docs) // 20))

    views: dict[str, dict[str, object]] = {}

    scores, ids, elapsed = search_centroid(doc_emb, query_emb, args.k, nlist)
    views["summary"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.00,
        "bytes": "cell stats",
    }

    t0 = time.perf_counter()
    _, ids, elapsed = search_flat(_binary_sign_view(doc_emb), _binary_sign_view(query_emb), args.k)
    views["binary sign"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed + (time.perf_counter() - t0 - elapsed),
        "cost": 0.08,
        "bytes": f"{dim // 8} B",
    }

    _, ids, elapsed = search_ivfpq(doc_emb, query_emb, args.k, nlist, args.m, args.nbits, args.nprobe)
    views["IVF-PQ"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.20,
        "bytes": f"{args.m} B code",
    }

    _, ids, elapsed = search_flat(_truncated(doc_emb, 96), _truncated(query_emb, 96), args.k)
    views["trunc-96 fp32"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.26,
        "bytes": "384 B",
    }

    _, ids, elapsed = search_flat(_int8_view(doc_emb), _int8_view(query_emb), args.k)
    views["int8 dense"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.30,
        "bytes": f"{dim} B",
    }

    _, ids, elapsed = search_flat(_truncated(doc_emb, 192), _truncated(query_emb, 192), args.k)
    views["trunc-192 fp32"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.40,
        "bytes": "768 B",
    }

    _, ids, elapsed = search_flat(doc_emb, query_emb, args.k)
    views["full dense"] = {
        "per": per_query_ndcg(ids, rels, args.k),
        "time": elapsed,
        "cost": 0.58,
        "bytes": f"{dim * 4} B",
    }

    ce = _load_ce_boundary(args.ce_boundary)
    if ce:
        ce_idx = np.array(sorted(ce), dtype="int64")
        idx = ce_idx[(ce_idx >= 0) & (ce_idx < len(queries))]
        ce_per = np.zeros(len(queries), dtype="float32")
        for qi in idx:
            ce_per[qi] = ce[int(qi)]
        views["cross-encoder rerank"] = {
            "per": ce_per,
            "time": (_load_ce_latency(args.ce_report) or 0.0) / 1e6 * max(1, len(queries)),
            "cost": 1.03,
            "bytes": "top-50 text",
        }
    else:
        idx = np.arange(len(queries), dtype="int64")

    if ce:
        idx = np.array(sorted(ce), dtype="int64")
        idx = idx[(idx >= 0) & (idx < len(queries))]
    else:
        idx = np.arange(len(queries), dtype="int64")

    order = [
        "summary",
        "binary sign",
        "IVF-PQ",
        "trunc-96 fp32",
        "int8 dense",
        "trunc-192 fp32",
        "full dense",
    ]
    if "cross-encoder rerank" in views:
        order.append("cross-encoder rerank")

    rows = []
    for name in order:
        view = views[name]
        rows.append(_summarize_view(
            name,
            view["per"],  # type: ignore[arg-type]
            idx,
            float(view["cost"]),
            float(view["time"]) / len(queries) * 1e6,
            str(view["bytes"]),
            args.lambda_cost,
        ))

    for i, row in enumerate(rows):
        if i == 0:
            row["lambda_star_from_prev"] = ""
            continue
        prev = rows[i - 1]
        dc = float(row["cost"]) - float(prev["cost"])
        if dc <= 0:
            row["lambda_star_from_prev"] = ""
        else:
            row["lambda_star_from_prev"] = float((float(row["ndcg@10"]) - float(prev["ndcg@10"])) / dc)

    util_matrix = np.column_stack([
        np.asarray(views[name]["per"], dtype="float32")[idx] - args.lambda_cost * float(views[name]["cost"])
        for name in order
    ])
    best = np.argmax(util_matrix, axis=1)
    for j, row in enumerate(rows):
        row["oracle_best_share"] = float(np.mean(best == j))
    fixed_utils = np.asarray([float(row["utility@10"]) for row in rows], dtype="float32")
    best_fixed_idx = int(np.argmax(fixed_utils))
    oracle_utility = float(np.mean(np.max(util_matrix, axis=1)))
    best_fixed_utility = float(fixed_utils[best_fixed_idx])
    menu_summary = {
        "menu": "compression ladder",
        "legal_choices": order,
        "best_fixed_view": rows[best_fixed_idx]["view"],
        "best_fixed_utility": best_fixed_utility,
        "oracle_utility": oracle_utility,
        "oracle_gap": float(oracle_utility - best_fixed_utility),
    }

    out = {
        "task": "compression_evidence_ladder_audit",
        "dataset": args.dataset,
        "model": args.model,
        "documents": len(docs),
        "queries": len(queries),
        "reported_queries": int(len(idx)),
        "reported_split": "CE-labeled boundary split" if ce else "all queries",
        "lambda": args.lambda_cost,
        "nlist": int(nlist),
        "pq_m": int(args.m),
        "pq_nbits": int(args.nbits),
        "nprobe": int(args.nprobe),
        "faiss_gpus": int(faiss.get_num_gpus()),
        "menu_summary": menu_summary,
        "rows": rows,
    }

    stem = safe_id(args.dataset + "__" + args.model)
    json_path = REPORT_DIR / f"compression_evidence_ladder_{stem}.json"
    md_path = REPORT_DIR / f"compression_evidence_ladder_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# Compression Evidence Ladder: {args.dataset}",
        "",
        f"- Encoder: `{args.model}`",
        f"- Documents: {len(docs):,}",
        f"- Queries loaded: {len(queries):,}",
        f"- Reported queries: {len(idx):,} ({out['reported_split']})",
        f"- FAISS GPUs visible: {faiss.get_num_gpus()}",
        f"- Menu oracle utility: {oracle_utility:.3f}; best fixed: {rows[best_fixed_idx]['view']} {best_fixed_utility:.3f}; oracle gap: {oracle_utility - best_fixed_utility:.3f}",
        "",
        "| view | bytes touched | latency us/q | raw NDCG@10 | utility | lambda* from prev | oracle-best share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lam = row["lambda_star_from_prev"]
        lam_txt = "" if lam == "" else f"{float(lam):.3f}"
        lines.append(
            f"| {row['view']} | {row['bytes_touched']} | {float(row['latency_us']):.1f} | "
            f"{float(row['ndcg@10']):.3f} | {float(row['utility@10']):.3f} | "
            f"{lam_txt} | {float(row['oracle_best_share']):.3f} |"
        )
    lines += [
        "",
        "Binary/int8 rows are post-training compressed views searched through a portable dequantized FAISS path; they are evidence-menu baselines, not new encoder training.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
