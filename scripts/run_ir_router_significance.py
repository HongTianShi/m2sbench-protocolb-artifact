#!/usr/bin/env python
"""Paired uncertainty checks for standard-qrel route baselines.

The public IR scorecard reports fixed summary/PQ/full views and simple
target-blind routers. This script recomputes the held-out 60/40 split and
reports paired bootstrap confidence intervals against fixed full. It is meant
to keep small router gains from being overinterpreted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    load_collection,
    per_query_ndcg,
    query_features,
    safe_id,
    score_features,
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


def ridge_choice(features, util_per, train_frac=0.6, alpha=1e-2):
    n = len(features)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    train = np.arange(cut)
    test = np.arange(cut, n)
    views = ["summary", "pq", "full"]
    x = np.column_stack([np.ones(n, dtype="float32"), features.astype("float32")])
    y = np.column_stack([util_per[v] for v in views]).astype("float32")
    xtx = x[train].T @ x[train]
    reg = alpha * np.eye(xtx.shape[0], dtype="float32")
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + reg, x[train].T @ y[train])
    pred = x[test] @ w
    return test, np.array(views)[np.argmax(pred, axis=1)]


def rf_choice(features, util_per, train_frac=0.6, seed=31):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    n = len(features)
    cut = max(1, min(n - 1, int(round(n * train_frac))))
    train = np.arange(cut)
    test = np.arange(cut, n)
    views = ["summary", "pq", "full"]
    y = np.column_stack([util_per[v] for v in views]).astype("float32")
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    )
    model.fit(features.astype("float32")[train], y[train])
    pred = model.predict(features.astype("float32")[test])
    return test, np.array(views)[np.argmax(pred, axis=1)]


def paired_ci(diff, reps=5000, seed=913):
    diff = np.asarray(diff, dtype="float32")
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(reps, dtype="float32")
    for i in range(reps):
        means[i] = diff[rng.integers(0, n, n)].mean()
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def utility_for_choices(per, choices, costs, lambda_cost, test):
    return np.array([per[v][test[i]] - lambda_cost * costs[v] for i, v in enumerate(choices)], dtype="float32")


def main():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    lambda_cost = 0.08
    costs = {"summary": 0.0, "pq": 0.20, "full": 0.58}
    by_policy = {
        "fixed_pq": [],
        "fixed_full": [],
        "summary_qpp_b0": [],
        "query_summary_rf_b0": [],
        "summary_pq_qpp_b1": [],
        "query_summary_pq_rf_b1": [],
    }
    per_dataset = []

    for dataset in DATASETS:
        report = json.loads((REPORT_DIR / f"standard_ir_access_{safe_id(dataset)}.json").read_text(encoding="utf-8"))
        docs, doc_ids, queries, qids, rels = load_collection(dataset, 0, 0, 13)
        doc_emb, query_emb = encode_or_load(dataset, model, docs, queries, 256)
        nlist = int(report["nlist"])
        pq_m = int(report["pq_m"])
        pq_nbits = int(report["pq_nbits"])
        nprobe = int(report["nprobe"])
        scores_s, ids_s, _ = search_centroid(doc_emb, query_emb, 10, nlist)
        scores_pq, ids_pq, _ = search_ivfpq(doc_emb, query_emb, 10, nlist, pq_m, pq_nbits, nprobe)
        scores_f, ids_f, _ = search_flat(doc_emb, query_emb, 10)
        ids_by = {"summary": ids_s, "pq": ids_pq, "full": ids_f}
        per = {v: per_query_ndcg(ids, rels, 10) for v, ids in ids_by.items()}
        util_per = {v: per[v] - lambda_cost * costs[v] for v in per}
        n = len(queries)
        cut = max(1, min(n - 1, int(round(n * 0.6))))
        test = np.arange(cut, n)

        feat_s = score_features(scores_s)
        feat_p = score_features(scores_pq)
        feat_q = query_features(query_emb)

        _, ch_qpp = ridge_choice(feat_s, util_per)
        _, ch_rf = rf_choice(np.column_stack([feat_q, feat_s]), util_per)
        _, ch_qpp_pq = ridge_choice(np.column_stack([feat_s, feat_p]), util_per)
        _, ch_rf_pq = rf_choice(np.column_stack([feat_q, feat_s, feat_p]), util_per)

        rows = {
            "fixed_pq": util_per["pq"][test],
            "fixed_full": util_per["full"][test],
            "summary_qpp_b0": utility_for_choices(per, ch_qpp, costs, lambda_cost, test),
            "query_summary_rf_b0": utility_for_choices(per, ch_rf, costs, lambda_cost, test),
            "summary_pq_qpp_b1": utility_for_choices(per, ch_qpp_pq, costs, lambda_cost, test),
            "query_summary_pq_rf_b1": utility_for_choices(per, ch_rf_pq, costs, lambda_cost, test),
        }
        for name, arr in rows.items():
            by_policy[name].append(arr)
        per_dataset.append({
            "dataset": dataset,
            "test_queries": int(len(test)),
            **{name: float(arr.mean()) for name, arr in rows.items()},
        })

    full = np.concatenate(by_policy["fixed_full"])
    aggregate = {}
    for name, chunks in by_policy.items():
        arr = np.concatenate(chunks)
        rec = {"utility": float(arr.mean()), "queries": int(len(arr))}
        if name != "fixed_full":
            rec["paired_diff_vs_full"] = paired_ci(arr - full)
        aggregate[name] = rec

    out = {"task": "ir_router_significance", "lambda": lambda_cost, "aggregate": aggregate, "per_dataset": per_dataset}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "ir_router_significance.json"
    md_path = REPORT_DIR / "ir_router_significance.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# IR Router Paired Uncertainty",
        "",
        "| policy | queries | utility | paired diff vs fixed full (95% CI) |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, rec in aggregate.items():
        if name == "fixed_full":
            diff = "--"
        else:
            d = rec["paired_diff_vs_full"]
            diff = f"{d['mean']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"
        lines.append(f"| {name} | {rec['queries']} | {rec['utility']:.4f} | {diff} |")
    lines += ["", "## Per-dataset held-out utility", ""]
    lines.append("| dataset | test q | fixed full | B0 QPP | B0 RF | B1 QPP+PQ | B1 RF+PQ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_dataset:
        lines.append(
            f"| {row['dataset']} | {row['test_queries']} | {row['fixed_full']:.3f} | "
            f"{row['summary_qpp_b0']:.3f} | {row['query_summary_rf_b0']:.3f} | "
            f"{row['summary_pq_qpp_b1']:.3f} | {row['query_summary_pq_rf_b1']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
