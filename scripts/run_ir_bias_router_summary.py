#!/usr/bin/env python
"""Summarize qrel-sparsity/tie diagnostics and held-out routing baselines.

This script is intentionally small and PDF-facing. It aggregates existing
standard IR access reports, then recomputes the same-candidate nested audit
arrays to explain when PQ is no worse than full before cost. The goal is to
make qrel sparsity, ties, and router baselines explicit rather than hiding
them behind aggregate best-view shares.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_nested_ir_refinement_audit import (
    flat_candidates,
    rerank_same_candidates,
    train_centroids,
    train_pq_reconstruction,
)
from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    load_collection,
    per_query_ndcg,
    safe_id,
)


DATASETS = [
    "beir/fiqa/test",
    "beir/scifact/test",
    "beir/nfcorpus/test",
    "beir/arguana",
    "antique/test",
]


def weighted_aggregate(rows, keys, weight_key="queries"):
    total = sum(float(r[weight_key]) for r in rows)
    out = {"queries": int(total)}
    for key in keys:
        out[key] = float(sum(float(r[key]) * float(r[weight_key]) for r in rows) / total)
    return out


def summarize_routers():
    policies = {}
    per_dataset = []
    for dataset in DATASETS:
        path = REPORT_DIR / f"standard_ir_access_{safe_id(dataset)}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"dataset": dataset}
        fixed = data.get("heldout_fixed_views", {})
        routers = data.get("routers", {})
        for name, rec in {
            "fixed_summary": fixed.get("summary"),
            "fixed_pq": fixed.get("pq"),
            "fixed_full": fixed.get("full"),
            **routers,
        }.items():
            if not rec:
                continue
            n = int(rec["test_queries"])
            policies.setdefault(name, []).append(
                {
                    "queries": n,
                    "utility": float(rec["utility@10"]),
                    "ndcg": float(rec["ndcg@10"]),
                    "cost": float(rec["cost"]),
                    "regret": float(rec["regret"]),
                }
            )
            row[f"{name}_utility"] = float(rec["utility@10"])
        per_dataset.append(row)

    agg = {}
    for name, rows in policies.items():
        agg[name] = weighted_aggregate(rows, ["utility", "ndcg", "cost", "regret"])
    return {"aggregate": agg, "per_dataset": per_dataset}


def nested_bias_for_dataset(dataset):
    report = json.loads((REPORT_DIR / f"nested_ir_refinement_{safe_id(dataset)}.json").read_text(encoding="utf-8"))
    docs, doc_ids, queries, qids, rels = load_collection(dataset, 0, 0, 13)
    doc_emb, query_emb = encode_or_load(dataset, report["model"], docs, queries, 256)
    _, candidates, _ = flat_candidates(doc_emb, query_emb, int(report["candidate_generation"]["candidate_k"]))
    nlist = int(report["nlist"])
    centroids, assign = train_centroids(doc_emb, nlist)
    summary_emb = centroids[assign].astype("float32")
    pq_emb = train_pq_reconstruction(doc_emb, int(report["pq_m"]), int(report["pq_nbits"]), 13)
    scores_s, ids_s, _ = rerank_same_candidates(query_emb, candidates, summary_emb, 10)
    scores_p, ids_p, _ = rerank_same_candidates(query_emb, candidates, pq_emb, 10)
    scores_f, ids_f, _ = rerank_same_candidates(query_emb, candidates, doc_emb.astype("float32"), 10)

    nd_s = per_query_ndcg(ids_s, rels, 10)
    nd_p = per_query_ndcg(ids_p, rels, 10)
    nd_f = per_query_ndcg(ids_f, rels, 10)
    diff = nd_p - nd_f
    rel_counts = np.array([len(r) for r in rels], dtype="float32")
    candidate_hits = []
    for cand, rel in zip(candidates, rels):
        rel_ids = set(rel)
        candidate_hits.append(sum(1 for x in cand if int(x) in rel_ids))
    candidate_hits = np.array(candidate_hits, dtype="float32")
    eps = 1e-8
    return {
        "dataset": dataset,
        "queries": int(len(queries)),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "mean_qrels_per_query": float(rel_counts.mean()),
        "median_qrels_per_query": float(np.median(rel_counts)),
        "candidate_top200_recall": float(np.mean(candidate_hits / np.maximum(rel_counts, 1))),
        "candidate_top200_hit": float(np.mean(candidate_hits > 0)),
        "positive_density_in_top200": float(candidate_hits.sum() / (len(queries) * candidates.shape[1])),
        "pq_ge_full_share": float(np.mean(diff >= -eps)),
        "pq_strict_better_share": float(np.mean(diff > eps)),
        "pq_equal_full_share": float(np.mean(np.abs(diff) <= eps)),
        "pq_strict_worse_share": float(np.mean(diff < -eps)),
        "mean_pq_minus_full_ndcg": float(diff.mean()),
        "summary_equal_full_share": float(np.mean(np.abs(nd_s - nd_f) <= eps)),
    }


def summarize_nested_bias():
    rows = [nested_bias_for_dataset(ds) for ds in DATASETS]
    agg = weighted_aggregate(
        rows,
        [
            "mean_qrels_per_query",
            "candidate_top200_recall",
            "candidate_top200_hit",
            "positive_density_in_top200",
            "pq_ge_full_share",
            "pq_strict_better_share",
            "pq_equal_full_share",
            "pq_strict_worse_share",
            "mean_pq_minus_full_ndcg",
            "summary_equal_full_share",
        ],
    )
    return {"aggregate": agg, "per_dataset": rows}


def markdown_table(rows, columns):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    routers = summarize_routers()
    nested = summarize_nested_bias()
    out = {"router_summary": routers, "nested_bias": nested}
    json_path = REPORT_DIR / "ir_bias_router_summary.json"
    md_path = REPORT_DIR / "ir_bias_router_summary.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    router_rows = []
    for name, rec in routers["aggregate"].items():
        router_rows.append({"policy": name, **rec})
    router_rows = sorted(router_rows, key=lambda r: (-r["utility"], r["cost"]))
    bias_rows = nested["per_dataset"]
    agg = nested["aggregate"]
    lines = [
        "# IR Qrel-Bias and Router Summary",
        "",
        "## Held-out route baselines",
        "",
        markdown_table(router_rows, ["policy", "queries", "utility", "ndcg", "cost", "regret"]),
        "",
        "## Same-candidate nested audit: qrel and tie diagnostics",
        "",
        f"- Weighted candidate-pool recall@200: {agg['candidate_top200_recall']:.3f}; hit@200: {agg['candidate_top200_hit']:.3f}.",
        f"- Positive qrel density inside top-200 candidate pools: {agg['positive_density_in_top200']:.4f}.",
        f"- PQ >= full before cost decomposes into strict PQ wins {agg['pq_strict_better_share']:.3f}, exact NDCG ties {agg['pq_equal_full_share']:.3f}, and strict PQ losses {agg['pq_strict_worse_share']:.3f}.",
        f"- Mean PQ-full NDCG@10 difference is {agg['mean_pq_minus_full_ndcg']:.3f}; full remains best on mean utility in the nested audit.",
        "",
        markdown_table(
            bias_rows,
            [
                "dataset",
                "queries",
                "mean_qrels_per_query",
                "candidate_top200_recall",
                "positive_density_in_top200",
                "pq_strict_better_share",
                "pq_equal_full_share",
                "pq_strict_worse_share",
            ],
        ),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:5000])


if __name__ == "__main__":
    main()
