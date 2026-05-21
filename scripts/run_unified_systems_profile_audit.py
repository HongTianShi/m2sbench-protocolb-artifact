#!/usr/bin/env python
"""Unified systems profile and cost-profile derivation for dense evidence views."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

import faiss
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from sentence_transformers import CrossEncoder

from run_compression_evidence_ladder_audit import _binary_sign_view, _int8_view, _truncated
from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    load_collection,
    per_query_ndcg,
    safe_id,
)


VIEW_ACCOUNTING = {
    "summary": {"parents": [], "incremental_cost": 0.0, "cumulative_cost": 0.0, "accounting": "root coarse state"},
    "binary_sign": {"parents": ["summary"], "incremental_cost": 0.08, "cumulative_cost": 0.08, "accounting": "replacement compressed-vector view"},
    "ivf_pq": {"parents": ["summary"], "incremental_cost": 0.20, "cumulative_cost": 0.20, "accounting": "replacement compressed-index view"},
    "int8_dense": {"parents": ["summary"], "incremental_cost": 0.30, "cumulative_cost": 0.30, "accounting": "replacement compressed-dense view"},
    "hnsw16": {"parents": ["summary"], "incremental_cost": 0.1375, "cumulative_cost": 0.1375, "accounting": "replacement shallow-ANN view"},
    "hnsw64": {"parents": ["hnsw16"], "incremental_cost": 0.1125, "cumulative_cost": 0.25, "accounting": "deeper ANN view after shallow graph access"},
    "full_dense": {"parents": ["summary"], "incremental_cost": 0.58, "cumulative_cost": 0.58, "accounting": "replacement full-vector view"},
    "cross_encoder_top50": {"parents": ["full_dense"], "incremental_cost": 0.45, "cumulative_cost": 1.03, "accounting": "incremental CE purchase after first-stage pool; cumulative ladder view when charged end-to-end"},
}


def gpu_name() -> str:
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "none"


def quantiles_us(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype="float64") * 1e6
    return {
        "p50_us": float(np.quantile(arr, 0.50)),
        "p95_us": float(np.quantile(arr, 0.95)),
        "p99_us": float(np.quantile(arr, 0.99)),
        "mean_us": float(arr.mean()),
        "qps_from_p50": float(1e6 / max(1e-9, np.quantile(arr, 0.50))),
    }


def sustained_index_qps(index, queries: np.ndarray, batch_size: int, passes: int = 3) -> dict[str, float]:
    q = np.ascontiguousarray(queries.astype("float32"))
    # Warm one full mini-batch before the measured full-pass loop.
    index.search(q[: min(len(q), batch_size)], 10)
    qps_vals = []
    for _ in range(passes):
        seen = 0
        t0 = time.perf_counter()
        for start in range(0, len(q), batch_size):
            batch = q[start : start + batch_size]
            index.search(batch, 10)
            seen += len(batch)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        qps_vals.append(seen / elapsed)
    return {
        "sustained_qps_median": float(np.median(qps_vals)),
        "sustained_qps_min": float(np.min(qps_vals)),
        "sustained_passes": int(passes),
    }


def attach_accounting(row: dict) -> dict:
    acc = VIEW_ACCOUNTING[row["view"]]
    row["parents"] = acc["parents"]
    row["incremental_cost"] = float(acc["incremental_cost"])
    row["cumulative_cost"] = float(acc["cumulative_cost"])
    row["accounting"] = acc["accounting"]
    row["declared_cost"] = float(acc["cumulative_cost"])
    row["qps_definition"] = "qps_from_p50 is 1e6 / p50_us; sustained_qps_median is measured by full passes over all profiler queries"
    return row


def maybe_gpu(index):
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index), "gpu", "included in FAISS search call"
        except Exception:
            return index, "cpu", "no"
    return index, "cpu", "no"


def time_index(name: str, index, queries: np.ndarray, reps: int, batch_size: int) -> tuple[dict, np.ndarray]:
    q = np.ascontiguousarray(queries.astype("float32"))
    n = len(q)
    # Warmup.
    for _ in range(min(5, n)):
        index.search(q[:1], 10)
    times = []
    last_ids = None
    for i in range(reps):
        start = (i * batch_size) % n
        batch = q[start : min(n, start + batch_size)]
        if len(batch) < batch_size:
            batch = q[:batch_size]
        t0 = time.perf_counter()
        _scores, ids = index.search(batch, 10)
        times.append((time.perf_counter() - t0) / len(batch))
        last_ids = ids
    if last_ids is None:
        raise RuntimeError(f"no timing rows for {name}")
    metrics = quantiles_us(times)
    metrics.update(sustained_index_qps(index, q, batch_size=batch_size))
    return metrics, last_ids


def time_ce(cross_model: str, queries: list[str], docs: list[str], candidate_ids: np.ndarray, reps: int, batch_size: int) -> dict:
    ce = CrossEncoder(cross_model, device="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    n = min(len(queries), reps)
    times = []
    for i in range(n):
        ids = [int(x) for x in candidate_ids[i, :50] if int(x) >= 0]
        pairs = [(queries[i], docs[j]) for j in ids]
        t0 = time.perf_counter()
        ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        times.append(time.perf_counter() - t0)
    metrics = quantiles_us(times)
    # Full-pass CE throughput over the sampled candidate pools. This remains a
    # profiler readout, not a hosted service benchmark.
    qps_vals = []
    for _ in range(2):
        seen = 0
        t0 = time.perf_counter()
        for i in range(n):
            ids = [int(x) for x in candidate_ids[i, :50] if int(x) >= 0]
            pairs = [(queries[i], docs[j]) for j in ids]
            ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            seen += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
        qps_vals.append(seen / elapsed)
    metrics.update({
        "sustained_qps_median": float(np.median(qps_vals)),
        "sustained_qps_min": float(np.min(qps_vals)),
        "sustained_passes": 2,
    })
    return metrics


def spearman(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype="float64")
    bb = np.asarray(b, dtype="float64")
    ra = np.empty_like(aa)
    rb = np.empty_like(bb)
    ra[np.argsort(aa)] = np.arange(len(aa))
    rb[np.argsort(bb)] = np.arange(len(bb))
    denom = np.std(ra) * np.std(rb)
    return 0.0 if denom == 0 else float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / denom)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reps", type=int, default=220)
    ap.add_argument("--ce-reps", type=int, default=64)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--ce-model-bytes",
        type=int,
        default=90_000_000,
        help="Approximate resident CE model footprint for memory/materialization profile.",
    )
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, _doc_ids, queries, _qids, rels = load_collection(args.dataset, 0, 0, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, 256)
    dim = int(doc_emb.shape[1])
    n_docs = len(docs)
    n_queries = len(queries)
    nlist = min(512, max(8, n_docs // 20))

    rows = []

    # Summary centroid search.
    kmeans = faiss.Kmeans(dim, nlist, niter=20, verbose=False, spherical=True, gpu=faiss.get_num_gpus() > 0)
    kmeans.train(doc_emb)
    cindex = faiss.IndexFlatIP(dim)
    cindex.add(kmeans.centroids.astype("float32"))
    cindex, device, h2d = maybe_gpu(cindex)
    metrics, ids = time_index("summary", cindex, query_emb, args.reps, args.batch_size)
    # Representative IDs are not materialized here; quality comes from the released compression audit.
    rows.append({
        "view": "summary",
        "device": device,
        "host_to_device": h2d,
        "batch_size": args.batch_size,
        "bytes_touched": int(nlist * dim * 4),
        "index_size_bytes": int(nlist * dim * 4),
        **metrics,
    })

    def flat_row(view: str, docs_emb: np.ndarray, queries_emb: np.ndarray, declared_cost: float, bytes_per_vec: int):
        index = faiss.IndexFlatIP(docs_emb.shape[1])
        index.add(np.ascontiguousarray(docs_emb.astype("float32")))
        index2, device2, h2d2 = maybe_gpu(index)
        m, ids_local = time_index(view, index2, queries_emb, args.reps, args.batch_size)
        per = per_query_ndcg(ids_local, rels[: len(ids_local)], 10) if len(ids_local) == len(rels) else None
        return {
            "view": view,
            "device": device2,
            "host_to_device": h2d2,
            "batch_size": args.batch_size,
            "bytes_touched": int(n_docs * bytes_per_vec),
            "index_size_bytes": int(n_docs * bytes_per_vec),
            **m,
        }, ids_local

    row, _ = flat_row("int8_dense", _int8_view(doc_emb), _int8_view(query_emb), 0.30, dim)
    rows.append(row)

    row, _ = flat_row("binary_sign", _binary_sign_view(doc_emb), _binary_sign_view(query_emb), 0.08, max(1, dim // 8))
    rows.append(row)

    # IVF-PQ.
    pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, 24, 8, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(17)
    train_n = min(len(doc_emb), max(4096, nlist * 80))
    pq.train(doc_emb[rng.choice(len(doc_emb), train_n, replace=False)])
    pq.add(doc_emb)
    pq.nprobe = 32
    pq2, device, h2d = maybe_gpu(pq)
    try:
        pq2.nprobe = 32
    except Exception:
        pass
    metrics, ids_pq = time_index("ivf_pq", pq2, query_emb, args.reps, args.batch_size)
    rows.append({
        "view": "ivf_pq",
        "device": device,
        "host_to_device": h2d,
        "batch_size": args.batch_size,
        "bytes_touched": int(32 * max(1, n_docs // nlist) * 24),
        "index_size_bytes": int(n_docs * 24 + nlist * dim * 4),
        **metrics,
    })

    # HNSW CPU.
    h = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    h.add(doc_emb)
    for ef, declared_cost in [(16, 0.1375), (64, 0.25)]:
        h.hnsw.efSearch = ef
        metrics, _ids_h = time_index(f"hnsw{ef}", h, query_emb, min(args.reps, 160), args.batch_size)
        rows.append({
            "view": f"hnsw{ef}",
            "device": "cpu",
            "host_to_device": "no",
            "batch_size": args.batch_size,
            "bytes_touched": int(ef * dim * 4 + ef * 32 * 4),
            "index_size_bytes": int(n_docs * dim * 4 + n_docs * 32 * 4),
            **metrics,
        })

    row, ids_full = flat_row("full_dense", doc_emb, query_emb, 0.58, dim * 4)
    rows.append(row)

    # CE timing over full dense candidate top-50.
    full50 = faiss.IndexFlatIP(dim)
    full50.add(doc_emb)
    _scores50, ids50 = full50.search(query_emb[: args.ce_reps], 50)
    ce_metrics = time_ce(args.cross_model, queries[: args.ce_reps], docs, ids50, args.ce_reps, 64)
    rows.append({
        "view": "cross_encoder_top50",
        "device": "gpu" if torch is not None and torch.cuda.is_available() else "cpu",
        "host_to_device": "token tensors created per query",
        "batch_size": 64,
        "bytes_touched": int(args.ce_reps * 50 * 512),  # logical text bytes/token proxy for the sampled profile.
        "index_size_bytes": int(args.ce_model_bytes),
        **ce_metrics,
    })

    rows = [attach_accounting(row) for row in rows]

    # Raw quality anchors from existing reports on the same FiQA/MiniLM evidence menu.
    quality = {}
    comp_path = REPORT_DIR / f"compression_evidence_ladder_{safe_id(args.dataset + '__' + args.model)}.json"
    if comp_path.exists():
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        for r in comp["rows"]:
            key = str(r["view"]).replace(" ", "_").replace("-", "_").lower()
            if key == "ivf_pq":
                quality["ivf_pq"] = float(r["ndcg@10"])
            elif key == "int8_dense":
                quality["int8_dense"] = float(r["ndcg@10"])
            elif key == "binary_sign":
                quality["binary_sign"] = float(r["ndcg@10"])
            elif key == "full_dense":
                quality["full_dense"] = float(r["ndcg@10"])
            elif key == "cross_encoder_rerank":
                quality["cross_encoder_top50"] = float(r["ndcg@10"])
            elif key == "summary":
                quality["summary"] = float(r["ndcg@10"])
    hnsw_path = REPORT_DIR / "hnsw_cascade_router_beir_fiqa_test.json"
    if hnsw_path.exists():
        hnsw_report = json.loads(hnsw_path.read_text(encoding="utf-8"))
        quality["hnsw16"] = float(hnsw_report["views"]["hnsw16"]["ndcg@10"])
        quality["hnsw64"] = float(hnsw_report["views"]["hnsw64"]["ndcg@10"])
    for row in rows:
        row["raw_ndcg"] = float(quality.get(row["view"], 0.0))

    for row in rows:
        row["materialized_bytes"] = int(row["bytes_touched"] + row["index_size_bytes"])
    max_bytes = max(r["materialized_bytes"] for r in rows)
    max_p95 = max(r["p95_us"] for r in rows)
    for row in rows:
        row["C_op"] = float(row["declared_cost"])
        row["C_mem"] = float(row["materialized_bytes"] / max(1, max_bytes))
        row["C_lat"] = float(row["p95_us"] / max(1e-9, max_p95))

    profile_rows = []
    for profile in ["C_op", "C_mem", "C_lat"]:
        utils = [float(r["raw_ndcg"] - args.lambda_cost * r[profile]) for r in rows]
        best = int(np.argmax(utils))
        profile_rows.append({
            "profile": profile,
            "winner": rows[best]["view"],
            "winner_utility": utils[best],
            "utilities": {rows[i]["view"]: utils[i] for i in range(len(rows))},
        })
    rank_corr = {
        "op_vs_mem": spearman([r["C_op"] for r in rows], [r["C_mem"] for r in rows]),
        "op_vs_lat": spearman([r["C_op"] for r in rows], [r["C_lat"] for r in rows]),
        "mem_vs_lat": spearman([r["C_mem"] for r in rows], [r["C_lat"] for r in rows]),
    }

    out = {
        "task": "unified_systems_profile_audit",
        "dataset": args.dataset,
        "model": args.model,
        "cross_model": args.cross_model,
        "hardware": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
            "faiss_gpus": int(faiss.get_num_gpus()),
            "gpu": gpu_name(),
        },
        "workload": {
            "documents": n_docs,
            "queries": n_queries,
            "topk": 10,
            "timing_reps": args.reps,
            "batch_size": args.batch_size,
            "ce_timing_queries": args.ce_reps,
            "warmup_queries": 5,
            "concurrency": 1,
            "qps_definition": "qps_from_p50 = 1e6 / p50_us; sustained_qps_median is measured by repeated full passes over all profiler queries at the reported batch size and concurrency 1",
        },
        "rows": rows,
        "profile_formulas": {
            "materialized_bytes": "bytes_touched + index_size_bytes",
            "C_op": "cumulative_cost from the versioned view-dependency DAG",
            "C_mem": "materialized_bytes / max_v materialized_bytes(v)",
            "C_lat": "p95_us / max_v p95_us(v)",
        },
        "profile_winners": profile_rows,
        "profile_rank_correlations": rank_corr,
    }
    json_path = REPORT_DIR / "unified_systems_profile_audit.json"
    md_path = REPORT_DIR / "unified_systems_profile_audit.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Unified Systems Profile Audit",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Encoder: `{args.model}`",
        f"- GPU: {out['hardware']['gpu']}",
        f"- Documents/queries: {n_docs:,}/{n_queries:,}",
        f"- Timing: batch size {args.batch_size}, concurrency 1, {args.reps} index repetitions, 5 warmup queries.",
        "- QPS* is `1e6 / p50_us`; sustained QPS is measured by repeated full passes over all profiler queries at concurrency 1.",
        "",
        "| view | parents | inc C | cum C | device | bytes touched | index bytes | p50 us | p95 us | p99 us | QPS* | sustained QPS | C_op | C_mem | C_lat | raw NDCG |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        parents = ",".join(r["parents"]) if r["parents"] else "-"
        lines.append(
            f"| {r['view']} | {parents} | {r['incremental_cost']:.3f} | {r['cumulative_cost']:.3f} | {r['device']} | {r['bytes_touched']} | {r['index_size_bytes']} | "
            f"{r['p50_us']:.2f} | {r['p95_us']:.2f} | {r['p99_us']:.2f} | {r['qps_from_p50']:.0f} | "
            f"{r['sustained_qps_median']:.0f} | {r['C_op']:.3f} | {r['C_mem']:.3f} | {r['C_lat']:.3f} | {r['raw_ndcg']:.3f} |"
        )
    lines += [
        "",
        "## Profile formulas",
        "",
        "- `materialized_bytes = bytes_touched + index_size_bytes`.",
        "- `C_op(v) = cumulative_cost(v)` from the versioned dependency DAG.",
        "- `C_mem(v) = materialized_bytes(v) / max_v materialized_bytes(v)`.",
        "- `C_lat(v) = p95_us(v) / max_v p95_us(v)`.",
        "- CE has two accounting contexts: incremental CE purchase cost `.45` after a first-stage pool exists, and cumulative ladder cost `1.03` when charged as a full text-reranking view.",
    ]
    lines += ["", "## Profile winners", "", "| profile | winner | utility |", "| --- | --- | ---: |"]
    for r in profile_rows:
        lines.append(f"| {r['profile']} | {r['winner']} | {r['winner_utility']:.4f} |")
    lines += [
        "",
        f"Rank correlations: op/mem {rank_corr['op_vs_mem']:.3f}, op/lat {rank_corr['op_vs_lat']:.3f}, mem/lat {rank_corr['mem_vs_lat']:.3f}.",
        "The intended readout is profile sensitivity: winners can change when the cost profile changes, so Protocol B stores raw quality, declared cost, latency, and bytes separately.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps({"profile_winners": profile_rows, "rank_correlations": rank_corr}, indent=2))


if __name__ == "__main__":
    main()
