#!/usr/bin/env python
"""Cross-encoder reranker as an expensive Protocol B evidence view.

The audit fixes a bi-encoder candidate pool for a standard-qrel IR collection,
then treats a cross-encoder reranker over that pool as a costly view.  The
method-visible decision is whether to keep the bi-encoder ranking or pay the
reranker cost before scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Plotting is optional; the CSV remains reproducible.
    plt = None

from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    eval_run,
    load_collection,
    per_query_ndcg,
    safe_id,
    search_flat,
)


def _score_features(scores: np.ndarray) -> np.ndarray:
    s = scores.astype("float32")
    top = s[:, 0]
    second = s[:, 1] if s.shape[1] > 1 else s[:, 0]
    gap = top - second
    mean = s.mean(axis=1)
    std = s.std(axis=1)
    span = s.max(axis=1) - s.min(axis=1)
    shifted = s - s.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(prob * np.log(np.maximum(prob, 1e-12))).sum(axis=1) / math.log(s.shape[1])
    return np.column_stack([top, gap, mean, std, span, entropy]).astype("float32")


def _rerank_candidates(
    cross_model: CrossEncoder,
    query_texts: list[str],
    doc_texts: list[str],
    candidate_ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    n, k = candidate_ids.shape
    pairs: list[tuple[str, str]] = []
    for qi in range(n):
        q = query_texts[qi]
        for did in candidate_ids[qi]:
            pairs.append((q, doc_texts[int(did)] if int(did) >= 0 else ""))

    t0 = time.perf_counter()
    scores = cross_model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    elapsed = time.perf_counter() - t0
    score_matrix = np.asarray(scores, dtype="float32").reshape(n, k)
    order = np.argsort(-score_matrix, axis=1)
    reranked = np.take_along_axis(candidate_ids, order, axis=1)
    rerank_scores = np.take_along_axis(score_matrix, order, axis=1)
    return reranked, rerank_scores, elapsed


def _bootstrap_mean(diff: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(diff)
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        vals.append(float(diff[idx].mean()))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def _router(
    features: np.ndarray,
    bi_ndcg: np.ndarray,
    rerank_ndcg: np.ndarray,
    lambda_cost: float,
    reranker_cost: float,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    idx = np.arange(len(bi_ndcg))
    train, test = train_test_split(idx, test_size=0.4, random_state=seed)
    target = (rerank_ndcg - lambda_cost * reranker_cost) - bi_ndcg
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features[train], target[train])
    pred = model.predict(features[test])
    choose_rerank = pred > 0.0
    util = np.where(choose_rerank, rerank_ndcg[test] - lambda_cost * reranker_cost, bi_ndcg[test])
    oracle = np.maximum(bi_ndcg[test], rerank_ndcg[test] - lambda_cost * reranker_cost)
    trace = {
        "test": test,
        "predicted_net_gain": pred,
        "choose_rerank": choose_rerank,
        "bi_ndcg": bi_ndcg[test],
        "rerank_ndcg": rerank_ndcg[test],
        "raw_gain": rerank_ndcg[test] - bi_ndcg[test],
        "net_gain": target[test],
        "oracle_worth_buy": rerank_ndcg[test] - lambda_cost * reranker_cost > bi_ndcg[test],
        "features": features[test],
    }
    return {
        "name": "cheap_only_ce_adapter",
        "feature_budget": "cheap bi-encoder score statistics only",
        "test_queries": int(len(test)),
        "utility@10": float(util.mean()),
        "ndcg@10": float(np.where(choose_rerank, rerank_ndcg[test], bi_ndcg[test]).mean()),
        "cost": float(choose_rerank.mean() * reranker_cost),
        "regret@10": float((oracle - util).mean()),
        "choice_share": {
            "bi_encoder": float(1.0 - choose_rerank.mean()),
            "cross_encoder": float(choose_rerank.mean()),
        },
    }, trace


def _write_boundary_artifacts(
    stem: str,
    trace: dict[str, np.ndarray],
    lambda_cost: float,
    reranker_cost: float,
) -> tuple[Path, Path]:
    csv_path = REPORT_DIR / f"cross_encoder_access_boundary_{stem}.csv"
    png_path = REPORT_DIR / f"cross_encoder_access_boundary_{stem}.png"
    feature_names = ["top_score", "margin", "mean_score", "std_score", "span", "entropy"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "query_index",
                *feature_names,
                "bi_ndcg",
                "cross_ndcg",
                "raw_gain",
                "net_gain",
                "predicted_net_gain",
                "oracle_worth_buy",
                "adapter_buys_cross_encoder",
            ]
        )
        for row_i, query_index in enumerate(trace["test"]):
            writer.writerow(
                [
                    int(query_index),
                    *[f"{float(v):.8f}" for v in trace["features"][row_i]],
                    f"{float(trace['bi_ndcg'][row_i]):.8f}",
                    f"{float(trace['rerank_ndcg'][row_i]):.8f}",
                    f"{float(trace['raw_gain'][row_i]):.8f}",
                    f"{float(trace['net_gain'][row_i]):.8f}",
                    f"{float(trace['predicted_net_gain'][row_i]):.8f}",
                    int(bool(trace["oracle_worth_buy"][row_i])),
                    int(bool(trace["choose_rerank"][row_i])),
                ]
            )

    margin = trace["features"][:, 1]
    raw_gain = trace["raw_gain"]
    worth = trace["oracle_worth_buy"]
    buys = trace["choose_rerank"]
    threshold = lambda_cost * reranker_cost

    if plt is None:
        return csv_path, Path("")

    plt.figure(figsize=(6.6, 3.55))
    plt.scatter(
        margin[~worth],
        raw_gain[~worth],
        s=26,
        c="#4C78A8",
        alpha=0.70,
        label="CE not cost-beneficial",
        linewidths=0.0,
    )
    plt.scatter(
        margin[worth],
        raw_gain[worth],
        s=30,
        c="#F58518",
        alpha=0.82,
        label="CE cost-beneficial",
        linewidths=0.0,
    )
    plt.scatter(
        margin[buys],
        raw_gain[buys],
        s=52,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.85,
        label="cheap-only adapter buys CE",
    )
    plt.axhline(threshold, color="#333333", linestyle="--", linewidth=1.15, label=r"break-even: $\lambda\Delta C$")
    plt.xlabel("Cheap-view confidence gap")
    plt.ylabel("Raw NDCG gain from buying CE")
    plt.grid(axis="y", alpha=0.24, linewidth=0.7)
    plt.legend(frameon=False, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()
    return csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--bi-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--reranker-cost", type=float, default=0.45)
    ap.add_argument("--embed-batch-size", type=int, default=256)
    ap.add_argument("--rerank-batch-size", type=int, default=64)
    # Keep the default aligned with run_standard_ir_access_audit.py because that
    # script's query-embedding cache is keyed by collection/model/length.
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs, doc_ids, queries, qids, rels = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.bi_model, docs, queries, args.embed_batch_size)
    bi_scores, bi_ids, bi_elapsed = search_flat(doc_emb, query_emb, args.candidate_k)

    cross = CrossEncoder(args.cross_model, device="cuda")
    rerank_ids, rerank_scores, rerank_elapsed = _rerank_candidates(
        cross, queries, docs, bi_ids, args.rerank_batch_size
    )

    bi_metrics = eval_run(bi_ids[:, : args.eval_k], rels)
    rerank_metrics = eval_run(rerank_ids[:, : args.eval_k], rels)
    bi_per = per_query_ndcg(bi_ids[:, : args.eval_k], rels, args.eval_k)
    rerank_per = per_query_ndcg(rerank_ids[:, : args.eval_k], rels, args.eval_k)
    rerank_util_per = rerank_per - args.lambda_cost * args.reranker_cost
    oracle_util_per = np.maximum(bi_per, rerank_util_per)

    # Protocol B routers decide whether to buy the expensive view before seeing it.
    # The reference cheap-only CE adapter therefore uses only cheap bi-encoder score
    # statistics and learns the expensive-view gain from held-in labeled calls.
    features = _score_features(bi_scores).astype("float32")
    router, router_trace = _router(features, bi_per, rerank_per, args.lambda_cost, args.reranker_cost, args.seed)

    fixed_bi_utility = float(bi_per.mean())
    fixed_rerank_utility = float(rerank_util_per.mean())
    gain = rerank_per - bi_per
    out = {
        "task": "cross_encoder_reranker_access_audit",
        "dataset": args.dataset,
        "bi_model": args.bi_model,
        "cross_model": args.cross_model,
        "documents": len(docs),
        "queries": len(queries),
        "qrel_pairs": int(sum(len(r) for r in rels)),
        "candidate_k": args.candidate_k,
        "eval_k": args.eval_k,
        "lambda": args.lambda_cost,
        "reranker_cost": args.reranker_cost,
        "views": {
            "bi_encoder": {
                **bi_metrics,
                "cost": 0.0,
                "utility@10": fixed_bi_utility,
                "mean_us_per_query": float(1e6 * bi_elapsed / max(1, len(queries))),
            },
            "cross_encoder": {
                **rerank_metrics,
                "cost": args.reranker_cost,
                "utility@10": fixed_rerank_utility,
                "mean_us_per_query": float(1e6 * rerank_elapsed / max(1, len(queries))),
            },
            "oracle": {
                "utility@10": float(oracle_util_per.mean()),
                "reranker_share": float((rerank_util_per > bi_per).mean()),
            },
        },
        "diagnostics": {
            "reranker_raw_gain@10": float(gain.mean()),
            "reranker_gain_ci": _bootstrap_mean(gain, draws=400, seed=args.seed + 101),
            "reranker_cost_adjusted_win_share": float((rerank_util_per > bi_per).mean()),
            "reranker_precost_win_share": float((rerank_per > bi_per).mean()),
            "lambda_star_bi_to_reranker": float(gain.mean() / args.reranker_cost) if args.reranker_cost > 0 else None,
        },
        "router": router,
    }

    stem = safe_id(args.dataset + "__" + args.bi_model + "__" + args.cross_model)
    boundary_csv, boundary_png = _write_boundary_artifacts(stem, router_trace, args.lambda_cost, args.reranker_cost)
    out["router"]["boundary_csv"] = str(boundary_csv)
    out["router"]["boundary_figure"] = str(boundary_png)
    json_path = REPORT_DIR / f"cross_encoder_reranker_access_{stem}.json"
    md_path = REPORT_DIR / f"cross_encoder_reranker_access_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# Cross-Encoder Reranker Access Audit: {args.dataset}",
        "",
        f"- Bi-encoder: `{args.bi_model}`",
        f"- Cross-encoder: `{args.cross_model}`",
        f"- Documents: {len(docs):,}",
        f"- Queries: {len(queries):,}",
        f"- Candidate pool: top-{args.candidate_k}; evaluation: NDCG@{args.eval_k}",
        "",
        "| view | NDCG@10 | cost | utility@10 | mean us/q |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| bi-encoder | {bi_metrics['ndcg@10']:.3f} | 0.000 | {fixed_bi_utility:.3f} | {out['views']['bi_encoder']['mean_us_per_query']:.1f} |",
        f"| cross-encoder | {rerank_metrics['ndcg@10']:.3f} | {args.reranker_cost:.3f} | {fixed_rerank_utility:.3f} | {out['views']['cross_encoder']['mean_us_per_query']:.1f} |",
        f"| cheap-only CE adapter | {router['ndcg@10']:.3f} | {router['cost']:.3f} | {router['utility@10']:.3f} |  |",
        f"| oracle route |  |  | {out['views']['oracle']['utility@10']:.3f} |  |",
        "",
        f"- Adapter feature budget: {router['feature_budget']}",
        f"- Raw reranker gain: {out['diagnostics']['reranker_raw_gain@10']:.4f}",
        f"- Cost-adjusted reranker win share: {out['diagnostics']['reranker_cost_adjusted_win_share']:.3f}",
        f"- Break-even lambda: {out['diagnostics']['lambda_star_bi_to_reranker']:.3f}",
        f"- Boundary CSV: `{boundary_csv.name}`",
        f"- Boundary figure: `{boundary_png.name}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
