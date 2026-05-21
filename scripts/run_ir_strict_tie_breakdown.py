#!/usr/bin/env python
"""Strict/tie/loss decomposition for standard-qrel access views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    load_collection,
    per_query_ndcg,
    safe_id,
    search_centroid,
    search_flat,
    search_ivfpq,
)


DATASETS = [
    "beir/fiqa/test",
    "beir/scifact/test",
    "beir/nfcorpus/test",
    "beir/arguana",
    "antique/test",
]


def split_shares(diff, eps=1e-8):
    return {
        "strict_better": float(np.mean(diff > eps)),
        "tie": float(np.mean(np.abs(diff) <= eps)),
        "strict_worse": float(np.mean(diff < -eps)),
    }


def main():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    lambda_cost = 0.08
    costs = {"summary": 0.0, "pq": 0.20, "full": 0.58}
    rows = []
    all_nf_diff = []
    all_pq_diff = []
    for dataset in DATASETS:
        report = json.loads((REPORT_DIR / f"standard_ir_access_{safe_id(dataset)}.json").read_text(encoding="utf-8"))
        docs, doc_ids, queries, qids, rels = load_collection(dataset, 0, 0, 13)
        doc_emb, query_emb = encode_or_load(dataset, model, docs, queries, 256)
        nlist = int(report["nlist"])
        pq_m = int(report["pq_m"])
        pq_nbits = int(report["pq_nbits"])
        nprobe = int(report["nprobe"])
        _, ids_s, _ = search_centroid(doc_emb, query_emb, 10, nlist)
        _, ids_pq, _ = search_ivfpq(doc_emb, query_emb, 10, nlist, pq_m, pq_nbits, nprobe)
        _, ids_f, _ = search_flat(doc_emb, query_emb, 10)

        nd_s = per_query_ndcg(ids_s, rels, 10)
        nd_p = per_query_ndcg(ids_pq, rels, 10)
        nd_f = per_query_ndcg(ids_f, rels, 10)
        u_s = nd_s - lambda_cost * costs["summary"]
        u_p = nd_p - lambda_cost * costs["pq"]
        u_f = nd_f - lambda_cost * costs["full"]
        nf_diff = np.maximum(u_s, u_p) - u_f
        pq_diff = nd_p - nd_f
        all_nf_diff.append(nf_diff)
        all_pq_diff.append(pq_diff)
        rows.append({
            "dataset": dataset,
            "queries": int(len(queries)),
            "qrel_pairs": int(sum(len(r) for r in rels)),
            "non_full_utility_vs_full": split_shares(nf_diff),
            "pq_ndcg_vs_full": split_shares(pq_diff),
        })

    nf_all = np.concatenate(all_nf_diff)
    pq_all = np.concatenate(all_pq_diff)
    aggregate = {
        "queries": int(len(nf_all)),
        "non_full_utility_vs_full": split_shares(nf_all),
        "pq_ndcg_vs_full": split_shares(pq_all),
    }
    out = {"task": "ir_strict_tie_breakdown", "lambda": lambda_cost, "aggregate": aggregate, "per_dataset": rows}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "ir_strict_tie_breakdown.json"
    md_path = REPORT_DIR / "ir_strict_tie_breakdown.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# IR Strict/Tie/Loss Breakdown",
        "",
        "| dataset | queries | NF utility >/=/< full | PQ NDCG >/=/< full |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        nf = row["non_full_utility_vs_full"]
        pq = row["pq_ndcg_vs_full"]
        lines.append(
            f"| {row['dataset']} | {row['queries']} | "
            f"{nf['strict_better']:.3f}/{nf['tie']:.3f}/{nf['strict_worse']:.3f} | "
            f"{pq['strict_better']:.3f}/{pq['tie']:.3f}/{pq['strict_worse']:.3f} |"
        )
    nf = aggregate["non_full_utility_vs_full"]
    pq = aggregate["pq_ndcg_vs_full"]
    lines.append(
        f"| weighted | {aggregate['queries']} | "
        f"{nf['strict_better']:.3f}/{nf['tie']:.3f}/{nf['strict_worse']:.3f} | "
        f"{pq['strict_better']:.3f}/{pq['tie']:.3f}/{pq['strict_worse']:.3f} |"
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
