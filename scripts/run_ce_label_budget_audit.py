#!/usr/bin/env python
"""CE-purchase adapter ablation and expensive-view label budget audit.

The cross-encoder boundary trace already records per-query cheap bi-encoder
statistics and CE gains. This script asks two follow-up questions:

1. Which cheap feature budget is enough for a surrogate-light CE-purchase
   adapter?
2. If CE labels are expensive, which queries should be CE-labeled first to
   train that adapter?

All training features are method-visible cheap signals: bi-encoder score
statistics, optional PQ/full agreement, and optional centroid/cell summaries.
Cross-encoder scores are used only as training labels and evaluator targets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import faiss
import numpy as np

from run_standard_ir_access_audit import (
    REPORT_DIR,
    encode_or_load,
    load_collection,
    query_features,
    safe_id,
    score_features,
    search_centroid,
    search_flat,
    search_ivfpq,
)


CSV_NAME = "cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__cross_encoder_ms_marco_MiniLM_L_6_v2.csv"


def _read_boundary(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        out[key] = np.array([float(r[key]) for r in rows], dtype="float32")
    out["query_index"] = out["query_index"].astype("int64")
    return out


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def _standardize_train(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x[train].mean(axis=0, keepdims=True)
    sd = x[train].std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    return (x - mu) / sd, mu, sd


def _ridge_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, eval_idx: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    xs, _, _ = _standardize_train(x.astype("float32"), train)
    design = np.column_stack([np.ones(len(xs), dtype="float32"), xs])
    xtx = design[train].T @ design[train]
    reg = alpha * np.eye(xtx.shape[0], dtype="float32")
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + reg, design[train].T @ y[train])
    return design[eval_idx] @ w


def _best_threshold(pred_gain: np.ndarray, train_idx: np.ndarray, bi: np.ndarray, ce: np.ndarray, lambda_cost: float, ce_cost: float) -> float:
    vals = pred_gain[train_idx]
    candidates = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 19)))
    candidates = np.concatenate(([float("-inf")], candidates, [float("inf")]))
    best_thr = 0.0
    best_util = float("-inf")
    for thr in candidates:
        choose_ce = vals > thr
        util = np.where(choose_ce, ce[train_idx] - lambda_cost * ce_cost, bi[train_idx])
        mean_util = float(util.mean())
        if mean_util > best_util:
            best_util = mean_util
            best_thr = float(thr)
    return best_thr


def _evaluate(pred_gain: np.ndarray, eval_idx: np.ndarray, bi: np.ndarray, ce: np.ndarray, lambda_cost: float, ce_cost: float, threshold: float = 0.0) -> dict[str, float]:
    choose_ce = pred_gain > threshold
    bi_eval = bi[eval_idx]
    ce_eval = ce[eval_idx]
    util = np.where(choose_ce, ce_eval - lambda_cost * ce_cost, bi_eval)
    raw = np.where(choose_ce, ce_eval, bi_eval)
    oracle = np.maximum(bi_eval, ce_eval - lambda_cost * ce_cost)
    cheap = bi_eval
    denom = float((oracle - cheap).mean())
    gap_closed = 0.0 if denom <= 1e-8 else float((util.mean() - cheap.mean()) / denom)
    return {
        "queries": int(len(eval_idx)),
        "ndcg@10": float(raw.mean()),
        "cost": float(choose_ce.mean() * ce_cost),
        "utility@10": float(util.mean()),
        "regret": float((oracle - util).mean()),
        "ce_buy_rate": float(choose_ce.mean()),
        "oracle_gap_closed": gap_closed,
        "threshold": float(threshold),
    }


def _overlap_features(ids_a: np.ndarray, ids_b: np.ndarray, qidx: np.ndarray, k: int = 10) -> np.ndarray:
    feats = []
    for qi in qidx:
        a = set(int(x) for x in ids_a[int(qi), :k] if int(x) >= 0)
        b = set(int(x) for x in ids_b[int(qi), :k] if int(x) >= 0)
        feats.append(len(a & b) / max(1, len(a | b)))
    return np.array(feats, dtype="float32")[:, None]


def _build_feature_sets(args: argparse.Namespace, boundary: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    qidx = boundary["query_index"].astype("int64")
    score_stats = np.column_stack([
        boundary["top_score"],
        boundary["mean_score"],
        boundary["std_score"],
        boundary["span"],
    ]).astype("float32")
    margin_entropy = np.column_stack([
        score_stats,
        boundary["margin"],
        boundary["entropy"],
    ]).astype("float32")

    docs, _, queries, _, _ = load_collection(args.dataset, args.max_docs, args.max_queries, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    nlist = min(args.nlist, max(8, len(docs) // 20))
    scores_summary, ids_summary, _ = search_centroid(doc_emb, query_emb, args.candidate_k, nlist)
    scores_pq, ids_pq, _ = search_ivfpq(doc_emb, query_emb, args.candidate_k, nlist, args.m, args.nbits, args.nprobe)
    scores_full, ids_full, _ = search_flat(doc_emb, query_emb, args.candidate_k)

    pq_score = score_features(scores_pq)[qidx]
    summary_score = score_features(scores_summary)[qidx]
    overlap_pq = _overlap_features(ids_full, ids_pq, qidx)
    overlap_summary = _overlap_features(ids_full, ids_summary, qidx)
    agreement = np.column_stack([margin_entropy, overlap_pq, overlap_summary, pq_score[:, :3]]).astype("float32")

    qfeat = query_features(query_emb)[qidx]
    cell_features = np.column_stack([agreement, summary_score, qfeat[:, :8]]).astype("float32")
    return {
        "score stats only": score_stats,
        "+ margin/entropy": margin_entropy,
        "+ cheap agreement": agreement,
        "+ cell/context proxies": cell_features,
    }


def _fixed_policy_rows(bi: np.ndarray, ce: np.ndarray, lambda_cost: float, ce_cost: float) -> list[dict[str, float | str]]:
    oracle = np.maximum(bi, ce - lambda_cost * ce_cost)
    cheap = {
        "policy": "fixed cheap",
        "queries": int(len(bi)),
        "ndcg@10": float(bi.mean()),
        "cost": 0.0,
        "utility@10": float(bi.mean()),
        "regret": float((oracle - bi).mean()),
        "ce_buy_rate": 0.0,
        "oracle_gap_closed": 0.0,
    }
    ce_util = ce - lambda_cost * ce_cost
    fixed_ce = {
        "policy": "fixed CE",
        "queries": int(len(bi)),
        "ndcg@10": float(ce.mean()),
        "cost": float(ce_cost),
        "utility@10": float(ce_util.mean()),
        "regret": float((oracle - ce_util).mean()),
        "ce_buy_rate": 1.0,
        "oracle_gap_closed": float((ce_util.mean() - bi.mean()) / max(float((oracle - bi).mean()), 1e-8)),
    }
    return [cheap, fixed_ce]


def _feature_ablation(features: dict[str, np.ndarray], bi: np.ndarray, ce: np.ndarray, lambda_cost: float, ce_cost: float) -> list[dict[str, float | str]]:
    n = len(bi)
    rng = np.random.default_rng(37)
    order = rng.permutation(n)
    cut = max(1, min(n - 1, int(round(n * 0.6))))
    train = order[:cut]
    test = order[cut:]
    target = (ce - lambda_cost * ce_cost) - bi
    rows = []
    oracle = np.maximum(bi[test], ce[test] - lambda_cost * ce_cost)
    cheap_gap = float((oracle - bi[test]).mean())
    for name, x in features.items():
        all_idx = np.arange(n)
        pred_all = _ridge_predict(x, target, train, all_idx)
        threshold = _best_threshold(pred_all, train, bi, ce, lambda_cost, ce_cost)
        row = _evaluate(pred_all[test], test, bi, ce, lambda_cost, ce_cost, threshold)
        row["feature_budget"] = name
        row["ce_labels"] = float(len(train) / n)
        row["policy"] = "surrogate-light CE adapter"
        # Re-normalize against the shared held-out split so rows are comparable.
        row["oracle_gap_closed"] = 0.0 if cheap_gap <= 1e-8 else float((row["utility@10"] - bi[test].mean()) / cheap_gap)
        rows.append(row)
    return rows


def _label_budget_rows(features: np.ndarray, boundary: dict[str, np.ndarray], lambda_cost: float, ce_cost: float) -> list[dict[str, float | str]]:
    bi = boundary["bi_ndcg"]
    ce = boundary["cross_ndcg"]
    target = (ce - lambda_cost * ce_cost) - bi
    n = len(bi)
    all_idx = np.arange(n)
    rng_split = np.random.default_rng(41)
    split = rng_split.permutation(n)
    pool = split[: int(round(n * 0.6))]
    test = np.sort(split[int(round(n * 0.6)):])
    margin = boundary["margin"]
    entropy = boundary["entropy"]
    span = boundary["span"]
    # High-value labels are cheap-ambiguous: low margin, high entropy, and wide
    # score spread. This uses only cheap statistics, then spends CE labels.
    high_value_score = -_z(margin) + _z(entropy) + 0.5 * _z(span)
    rows: list[dict[str, float | str]] = []
    for budget in [0.05, 0.10, 0.20]:
        train_n = max(4, min(len(pool), int(round(n * budget))))
        random_metrics = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            train = rng.choice(pool, train_n, replace=False)
            pred_all = _ridge_predict(features, target, train, all_idx)
            threshold = _best_threshold(pred_all, train, bi, ce, lambda_cost, ce_cost)
            random_metrics.append(_evaluate(pred_all[test], test, bi, ce, lambda_cost, ce_cost, threshold))
        avg = {k: float(np.mean([m[k] for m in random_metrics])) for k in random_metrics[0] if k != "queries"}
        rows.append({
            "selector": "random CE labels",
            "ce_label_share": budget,
            "queries": int(round(np.mean([m["queries"] for m in random_metrics]))),
            **avg,
        })

        pool_order = pool[np.argsort(-high_value_score[pool])]
        train = pool_order[:train_n]
        pred_all = _ridge_predict(features, target, train, all_idx)
        threshold = _best_threshold(pred_all, train, bi, ce, lambda_cost, ce_cost)
        row = _evaluate(pred_all[test], test, bi, ce, lambda_cost, ce_cost, threshold)
        row["selector"] = "high-value cheap-ambiguous labels"
        row["ce_label_share"] = budget
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-csv", type=Path, default=REPORT_DIR / CSV_NAME)
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--nlist", type=int, default=512)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--ce-cost", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = _read_boundary(args.boundary_csv)
    bi = boundary["bi_ndcg"]
    ce = boundary["cross_ndcg"]
    features = _build_feature_sets(args, boundary)
    ablation = _feature_ablation(features, bi, ce, args.lambda_cost, args.ce_cost)
    label_budget = _label_budget_rows(features["+ margin/entropy"], boundary, args.lambda_cost, args.ce_cost)
    fixed = _fixed_policy_rows(bi, ce, args.lambda_cost, args.ce_cost)

    out = {
        "task": "ce_label_budget_audit",
        "dataset": args.dataset,
        "model": args.model,
        "boundary_csv": str(args.boundary_csv),
        "queries": int(len(bi)),
        "lambda": args.lambda_cost,
        "ce_cost": args.ce_cost,
        "fixed_policies": fixed,
        "feature_ablation": ablation,
        "label_budget": label_budget,
        "faiss_gpus": int(faiss.get_num_gpus()),
    }
    stem = safe_id(args.dataset + "__" + args.model)
    json_path = REPORT_DIR / f"ce_label_budget_{stem}.json"
    md_path = REPORT_DIR / f"ce_label_budget_{stem}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# CE Label Budget Audit: {args.dataset}",
        "",
        f"- Boundary queries: {len(bi):,}",
        f"- Lambda: {args.lambda_cost:.3f}",
        f"- CE purchase cost: {args.ce_cost:.3f}",
        f"- FAISS GPUs visible: {faiss.get_num_gpus()}",
        "",
        "## Fixed policies on the CE-labeled split",
        "",
        "| policy | raw NDCG | cost | utility | regret | CE buy rate | oracle gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fixed:
        lines.append(f"| {row['policy']} | {row['ndcg@10']:.3f} | {row['cost']:.3f} | {row['utility@10']:.3f} | {row['regret']:.3f} | {row['ce_buy_rate']:.3f} | {row['oracle_gap_closed']:.3f} |")

    lines += [
        "",
        "## Surrogate-light CE adapter feature ablation",
        "",
        "| feature budget | CE-labeled train share | utility | regret | CE buy rate | oracle gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ablation:
        lines.append(f"| {row['feature_budget']} | {row['ce_labels']:.2f} | {row['utility@10']:.3f} | {row['regret']:.3f} | {row['ce_buy_rate']:.3f} | {row['oracle_gap_closed']:.3f} |")

    lines += [
        "",
        "## Expensive-view labeling budget",
        "",
        "| selector | CE label share | utility | regret | CE buy rate | oracle gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in label_budget:
        lines.append(f"| {row['selector']} | {row['ce_label_share']:.2f} | {row['utility@10']:.3f} | {row['regret']:.3f} | {row['ce_buy_rate']:.3f} | {row['oracle_gap_closed']:.3f} |")

    lines += [
        "",
        "High-value labels are chosen by cheap-view ambiguity only: low margin, high entropy, and wide score spread. CE scores never enter method-visible features.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
