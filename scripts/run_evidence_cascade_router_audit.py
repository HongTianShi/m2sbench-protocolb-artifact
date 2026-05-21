#!/usr/bin/env python
"""Evidence-tier cascade routing audit over the public IR scorecard.

This script evaluates a stronger target-blind routing baseline than query-only
QPP.  The router first pays for cheap summary/PQ evidence, inspects only
returned scores and result-set agreement, and then chooses summary, PQ, or full
reranking.  It never uses qrels, labels, full vectors, document text, or hidden
targets at test time.
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    "beir/fiqa/test",
    "beir/scifact/test",
    "beir/nfcorpus/test",
    "beir/arguana",
    "antique/test",
]
VIEWS = np.asarray(["summary", "pq", "full"], dtype=object)
COSTS = {"summary": 0.0, "pq": 0.20, "full": 0.58}


def overlap_features(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    rows = []
    for a, b in zip(ids_a, ids_b):
        aa = [int(x) for x in a if int(x) >= 0]
        bb = [int(x) for x in b if int(x) >= 0]
        sa, sb = set(aa), set(bb)
        inter = sa & sb
        union = sa | sb
        rr = 0.0
        if aa:
            rr = sum(1.0 / (1 + i) for i, x in enumerate(aa) if x in sb) / len(aa)
        rows.append(
            [
                len(inter) / max(1, len(sa)),
                len(inter) / max(1, len(sb)),
                len(inter) / max(1, len(union)),
                float(bool(aa and bb and aa[0] == bb[0])),
                float(np.mean(np.asarray(aa[: len(bb)]) == np.asarray(bb[: len(aa)]))) if aa and bb else 0.0,
                rr,
            ]
        )
    return np.asarray(rows, dtype="float32")


def paired_ci(diff: np.ndarray, reps: int = 2000, seed: int = 17) -> dict:
    diff = np.asarray(diff, dtype="float32")
    rng = np.random.default_rng(seed)
    n = len(diff)
    vals = np.empty(reps, dtype="float32")
    for i in range(reps):
        vals[i] = diff[rng.integers(0, n, n)].mean()
    return {"mean": float(diff.mean()), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975))}


def clone_model(model, seed: int):
    if isinstance(model, ExtraTreesRegressor):
        p = model.get_params()
        p["random_state"] = seed
        return ExtraTreesRegressor(**p)
    if isinstance(model, RandomForestRegressor):
        p = model.get_params()
        p["random_state"] = seed
        return RandomForestRegressor(**p)
    if isinstance(model, HistGradientBoostingRegressor):
        p = model.get_params()
        p["random_state"] = seed
        return HistGradientBoostingRegressor(**p)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def model_factory(name: str, seed: int):
    if name == "et":
        return ExtraTreesRegressor(n_estimators=900, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=350, learning_rate=0.03, l2_regularization=0.05, min_samples_leaf=10, random_state=seed)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def eval_route(name: str, choices: np.ndarray, y: np.ndarray, ndcg: np.ndarray, test_idx: np.ndarray) -> dict:
    choice_idx = np.asarray(choices, dtype=int)
    rows = np.arange(len(test_idx))
    util = y[test_idx][rows, choice_idx]
    raw = ndcg[test_idx][rows, choice_idx]
    cost_arr = np.asarray([COSTS[VIEWS[i]] for i in choice_idx], dtype="float32")
    oracle = y[test_idx].max(axis=1)
    return {
        "name": name,
        "test_queries": int(len(test_idx)),
        "utility": float(util.mean()),
        "ndcg": float(raw.mean()),
        "cost": float(cost_arr.mean()),
        "regret": float((oracle - util).mean()),
        "choices": {str(VIEWS[i]): float(np.mean(choice_idx == i)) for i in range(3)},
        "_utility_vector": util.astype(float).tolist(),
        "_choice_vector": choice_idx.astype(int).tolist(),
    }


def train_multi_reg(name: str, model, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    preds = []
    for j in range(3):
        m = clone_model(model, seed + j)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(x[train_idx], y[train_idx, j])
        preds.append(m.predict(x[test_idx]))
    pred = np.stack(preds, axis=1)
    return eval_route(name, pred.argmax(axis=1), y, ndcg, test_idx)


def threshold_gate(features: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    # Train a readable two-stage rule: use summary on high agreement/confidence,
    # otherwise PQ or full depending on a second visible threshold.
    best = None
    cols = range(features.shape[1])
    qvals = np.linspace(0.05, 0.95, 19)
    for c1 in cols:
        vals1 = np.unique(np.quantile(features[train_idx, c1], qvals))
        for thr1 in vals1:
            for op1 in ("ge", "le"):
                mask1_train = features[train_idx, c1] >= thr1 if op1 == "ge" else features[train_idx, c1] <= thr1
                for c2 in cols:
                    vals2 = np.unique(np.quantile(features[train_idx, c2], qvals))
                    for thr2 in vals2:
                        for op2 in ("ge", "le"):
                            mask2_train = features[train_idx, c2] >= thr2 if op2 == "ge" else features[train_idx, c2] <= thr2
                            choices = np.full(len(train_idx), 2, dtype=int)
                            choices[mask2_train] = 1
                            choices[mask1_train] = 0
                            util = y[train_idx][np.arange(len(train_idx)), choices].mean()
                            if best is None or util > best[0]:
                                best = (float(util), c1, float(thr1), op1, c2, float(thr2), op2)
    _, c1, thr1, op1, c2, thr2, op2 = best
    mask1 = features[test_idx, c1] >= thr1 if op1 == "ge" else features[test_idx, c1] <= thr1
    mask2 = features[test_idx, c2] >= thr2 if op2 == "ge" else features[test_idx, c2] <= thr2
    choices = np.full(len(test_idx), 2, dtype=int)
    choices[mask2] = 1
    choices[mask1] = 0
    out = eval_route("two_threshold_cascade", choices, y, ndcg, test_idx)
    out["rule"] = f"summary if f{c1} {op1} {thr1:.4g}; else pq if f{c2} {op2} {thr2:.4g}; else full"
    return out


def selective_full_regression(name: str, model, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    """Cost-sensitive selective reranking baseline.

    The policy first assumes the compressed/PQ view, then pays for full
    reranking only when a model predicts positive utility gain over PQ.  The
    decision threshold is selected on train utility, not on test qrels.
    """
    m = clone_model(model, seed)
    target = y[:, 2] - y[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(x[train_idx], target[train_idx])
    pred_train = m.predict(x[train_idx])
    pred_test = m.predict(x[test_idx])
    candidates = np.unique(np.quantile(pred_train, np.linspace(0.02, 0.98, 49)))
    candidates = np.r_[candidates.min() - 1e-6, candidates, candidates.max() + 1e-6]
    best = None
    for thr in candidates:
        choices = np.where(pred_train > thr, 2, 1)
        util = y[train_idx][np.arange(len(train_idx)), choices].mean()
        if best is None or util > best[0]:
            best = (float(util), float(thr))
    thr = best[1]
    test_choices = np.where(pred_test > thr, 2, 1)
    out = eval_route(name, test_choices, y, ndcg, test_idx)
    out["rule"] = f"full if predicted U(full)-U(pq) > {thr:.4g}; else pq"
    return out


def selective_full_classifier(name: str, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    label = (y[:, 2] > y[:, 1]).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, l2_regularization=0.05, min_samples_leaf=12, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x[train_idx], label[train_idx])
    prob_train = clf.predict_proba(x[train_idx])[:, 1]
    prob_test = clf.predict_proba(x[test_idx])[:, 1]
    candidates = np.unique(np.quantile(prob_train, np.linspace(0.02, 0.98, 49)))
    candidates = np.r_[0.0, candidates, 1.0]
    best = None
    for thr in candidates:
        choices = np.where(prob_train > thr, 2, 1)
        util = y[train_idx][np.arange(len(train_idx)), choices].mean()
        if best is None or util > best[0]:
            best = (float(util), float(thr))
    thr = best[1]
    test_choices = np.where(prob_test > thr, 2, 1)
    out = eval_route(name, test_choices, y, ndcg, test_idx)
    out["rule"] = f"full if calibrated P(U_full>U_pq) > {thr:.3f}; else pq"
    return out


def load_pooled(lambda_cost: float, model_name: str, batch_size: int, seed: int):
    xs, ys, ndcgs, groups = [], [], [], []
    meta = []
    for di, dataset in enumerate(DATASETS):
        print("load", dataset, flush=True)
        report = json.loads((REPORT_DIR / f"standard_ir_access_{safe_id(dataset)}.json").read_text(encoding="utf-8"))
        docs, _doc_ids, queries, _qids, rels = load_collection(dataset, 0, 0, seed)
        doc_emb, query_emb = encode_or_load(dataset, model_name, docs, queries, batch_size)
        nlist = int(report["nlist"])
        pq_m = int(report["pq_m"])
        pq_nbits = int(report["pq_nbits"])
        nprobe = int(report["nprobe"])

        scores_s, ids_s, _ = search_centroid(doc_emb, query_emb, 10, nlist)
        scores_pq, ids_pq, _ = search_ivfpq(doc_emb, query_emb, 10, nlist, pq_m, pq_nbits, nprobe)
        scores_f, ids_f, _ = search_flat(doc_emb, query_emb, 10)
        per = np.column_stack(
            [
                per_query_ndcg(ids_s, rels, 10),
                per_query_ndcg(ids_pq, rels, 10),
                per_query_ndcg(ids_f, rels, 10),
            ]
        ).astype("float32")
        costs = np.asarray([COSTS[v] for v in VIEWS], dtype="float32")
        util = per - lambda_cost * costs[None, :]

        onehot = np.zeros((len(queries), len(DATASETS)), dtype="float32")
        onehot[:, di] = 1.0
        feat = np.column_stack(
            [
                onehot,
                query_features(query_emb, k=64),
                score_features(scores_s),
                score_features(scores_pq),
                overlap_features(ids_s, ids_pq),
            ]
        ).astype("float32")
        xs.append(feat)
        ys.append(util)
        ndcgs.append(per)
        groups.extend([dataset] * len(queries))
        meta.append({"dataset": dataset, "queries": len(queries), "qrel_pairs": int(sum(len(r) for r in rels))})
    return np.vstack(xs), np.vstack(ys), np.vstack(ndcgs), np.asarray(groups), meta


def feature_slices(n_features: int) -> dict[str, np.ndarray]:
    # Layout: dataset one-hot (5), query embedding summaries (69),
    # summary scores (6), PQ scores (6), summary/PQ returned-ID agreement (6).
    d0, d1 = 0, len(DATASETS)
    q0, q1 = d1, d1 + 69
    s0, s1 = q1, q1 + 6
    p0, p1 = s1, s1 + 6
    a0, a1 = p1, n_features
    dataset = np.arange(d0, d1)
    query = np.arange(q0, q1)
    summary_scores = np.arange(s0, s1)
    pq_scores = np.arange(p0, p1)
    agreement = np.arange(a0, a1)
    return {
        "B0_query_summary": np.r_[dataset, query, summary_scores],
        "B0_no_dataset": np.r_[query, summary_scores],
        "B1_scores_agreement": np.r_[summary_scores, pq_scores, agreement],
        "B1_no_dataset": np.r_[query, summary_scores, pq_scores, agreement],
        "B1_full": np.arange(n_features),
        "agreement_only": agreement,
        "score_only": np.r_[summary_scores, pq_scores],
    }


def fixed_lambda_frontier(ndcg: np.ndarray, lambda_grid: list[float]) -> list[dict]:
    costs = np.asarray([COSTS[v] for v in VIEWS], dtype="float32")
    rows = []
    for lam in lambda_grid:
        util = ndcg - float(lam) * costs[None, :]
        best = util.argmax(axis=1)
        fixed = util.mean(axis=0)
        rows.append(
            {
                "lambda": float(lam),
                "fixed_summary": float(fixed[0]),
                "fixed_pq": float(fixed[1]),
                "fixed_full": float(fixed[2]),
                "best_fixed": str(VIEWS[int(np.argmax(fixed))]),
                "oracle": float(util.max(axis=1).mean()),
                "oracle_choices": {str(VIEWS[i]): float(np.mean(best == i)) for i in range(3)},
            }
        )
    return rows


def run_split(x, y, ndcg, train_idx, test_idx, seed: int):
    routes = [eval_route(f"fixed_{VIEWS[i]}", np.full(len(test_idx), i), y, ndcg, test_idx) for i in range(3)]
    # Tree/linear routers use query, score, and summary-PQ agreement features.
    for key in ("et", "rf", "hgb", "ridge"):
        routes.append(train_multi_reg(f"{key}_evidence_cascade", model_factory(key, seed), x, y, ndcg, train_idx, test_idx, seed))
    routes.append(threshold_gate(x[:, -6:], y, ndcg, train_idx, test_idx))
    best_fixed = max((r for r in routes if r["name"].startswith("fixed_")), key=lambda r: r["utility"])
    for r in routes:
        r["diff_vs_best_fixed"] = float(r["utility"] - best_fixed["utility"])
    return routes, best_fixed


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split-reps", type=int, default=30)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, ndcg, groups, meta = load_pooled(args.lambda_cost, args.model, args.batch_size, args.seed)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    cut = int(round(0.6 * len(idx)))
    train_idx, test_idx = idx[:cut], idx[cut:]
    routes, best_fixed = run_split(x, y, ndcg, train_idx, test_idx, args.seed)
    full_vec = np.asarray(next(r for r in routes if r["name"] == "fixed_full")["_utility_vector"], dtype="float32")
    for r in routes:
        if r["name"] != "fixed_full":
            r["paired_diff_vs_fixed_full"] = paired_ci(np.asarray(r["_utility_vector"], dtype="float32") - full_vec)

    feature_ablation = []
    for label, cols in feature_slices(x.shape[1]).items():
        r = train_multi_reg(f"rf_{label}", model_factory("rf", args.seed + 910), x[:, cols], y, ndcg, train_idx, test_idx, args.seed + 910)
        r["paired_diff_vs_fixed_full"] = paired_ci(np.asarray(r["_utility_vector"], dtype="float32") - full_vec)
        feature_ablation.append({k: v for k, v in r.items() if not k.startswith("_")})

    selective_routes = []
    slices = feature_slices(x.shape[1])
    for label in ("B1_no_dataset", "B1_scores_agreement", "B0_no_dataset", "B1_full"):
        cols = slices[label]
        for key in ("rf", "et", "hgb", "ridge"):
            r = selective_full_regression(
                f"selective_{key}_{label}",
                model_factory(key, args.seed + 1200),
                x[:, cols],
                y,
                ndcg,
                train_idx,
                test_idx,
                args.seed + 1200,
            )
            r["paired_diff_vs_fixed_full"] = paired_ci(np.asarray(r["_utility_vector"], dtype="float32") - full_vec)
            selective_routes.append({k: v for k, v in r.items() if not k.startswith("_")})
        if label != "B1_full":
            r = selective_full_classifier(f"selective_hgb_cls_{label}", x[:, cols], y, ndcg, train_idx, test_idx, args.seed + 1300)
            r["paired_diff_vs_fixed_full"] = paired_ci(np.asarray(r["_utility_vector"], dtype="float32") - full_vec)
            selective_routes.append({k: v for k, v in r.items() if not k.startswith("_")})

    split_rows = []
    per_dataset = []
    for rep in range(args.split_reps):
        rrng = np.random.default_rng(args.seed + 1000 + rep)
        ridx = np.arange(len(x))
        rrng.shuffle(ridx)
        rcut = int(round(0.6 * len(ridx)))
        tr, te = ridx[:rcut], ridx[rcut:]
        rep_routes, rep_fixed = run_split(x, y, ndcg, tr, te, args.seed + rep)
        best_adaptive = max((r for r in rep_routes if not r["name"].startswith("fixed_")), key=lambda r: r["utility"])
        split_rows.append(
            {
                "rep": rep,
                "best_fixed": rep_fixed["name"],
                "best_fixed_utility": rep_fixed["utility"],
                "best_adaptive": best_adaptive["name"],
                "best_adaptive_utility": best_adaptive["utility"],
                "adaptive_minus_best_fixed": best_adaptive["utility"] - rep_fixed["utility"],
            }
        )
    diffs = np.asarray([r["adaptive_minus_best_fixed"] for r in split_rows], dtype="float32")
    split_summary = {
        "reps": int(len(split_rows)),
        "mean_adaptive_minus_best_fixed": float(diffs.mean()),
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "positive_share": float(np.mean(diffs > 0)),
    }

    leave_one_dataset_out = []
    for dataset in DATASETS:
        tr = np.where(groups != dataset)[0]
        te = np.where(groups == dataset)[0]
        lodo_routes, lodo_fixed = run_split(x, y, ndcg, tr, te, args.seed + 77)
        best_adaptive = max((r for r in lodo_routes if not r["name"].startswith("fixed_")), key=lambda r: r["utility"])
        leave_one_dataset_out.append(
            {
                "heldout_dataset": dataset,
                "train_queries": int(len(tr)),
                "test_queries": int(len(te)),
                "best_fixed": lodo_fixed["name"],
                "best_fixed_utility": lodo_fixed["utility"],
                "best_adaptive": best_adaptive["name"],
                "best_adaptive_utility": best_adaptive["utility"],
                "adaptive_minus_best_fixed": best_adaptive["utility"] - lodo_fixed["utility"],
                "best_adaptive_choices": best_adaptive["choices"],
            }
        )

    for dataset in DATASETS:
        mask = groups[test_idx] == dataset
        if not np.any(mask):
            continue
        sub_test = test_idx[mask]
        sub_routes = [eval_route(f"fixed_{VIEWS[i]}", np.full(len(sub_test), i), y, ndcg, sub_test) for i in range(3)]
        best_route = max(routes, key=lambda r: r["utility"])
        sub_choices = np.asarray(best_route["_choice_vector"], dtype=int)[mask]
        sub_adapt = eval_route(best_route["name"], sub_choices, y, ndcg, sub_test)
        sub_full = np.asarray(sub_routes[2]["_utility_vector"], dtype="float32")
        sub_adapt_vec = np.asarray(sub_adapt["_utility_vector"], dtype="float32")
        per_dataset.append(
            {
                "dataset": dataset,
                "test_queries": int(mask.sum()),
                "fixed_summary": sub_routes[0]["utility"],
                "fixed_pq": sub_routes[1]["utility"],
                "fixed_full": sub_routes[2]["utility"],
                "adaptive_policy": best_route["name"],
                "adaptive_utility": sub_adapt["utility"],
                "adaptive_ndcg": sub_adapt["ndcg"],
                "adaptive_cost": sub_adapt["cost"],
                "adaptive_choices": sub_adapt["choices"],
                "paired_diff_vs_fixed_full": paired_ci(sub_adapt_vec - sub_full),
            }
        )

    public_routes = []
    for r in sorted(routes, key=lambda z: z["utility"], reverse=True):
        public_routes.append({k: v for k, v in r.items() if not k.startswith("_")})

    out = {
        "task": "evidence_cascade_router_audit",
        "lambda": args.lambda_cost,
        "features": "dataset_onehot+query_features+summary_scores+pq_scores+summary_pq_result_agreement",
        "datasets": meta,
        "total_queries": int(len(x)),
        "test_queries": int(len(test_idx)),
        "routes": public_routes,
        "feature_ablation": feature_ablation,
        "selective_reranking_baselines": selective_routes,
        "lambda_frontier": fixed_lambda_frontier(ndcg, [0.0, 0.02, 0.05, 0.08, 0.12, 0.20]),
        "split_replicates": split_rows,
        "split_summary": split_summary,
        "leave_one_dataset_out": leave_one_dataset_out,
        "per_dataset_test_slice": per_dataset,
    }
    json_path = REPORT_DIR / "evidence_cascade_router_audit.json"
    md_path = REPORT_DIR / "evidence_cascade_router_audit.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Evidence Cascade Router Audit",
        "",
        f"- Total queries: {len(x):,}",
        f"- Test queries in displayed split: {len(test_idx):,}",
        f"- Features: {out['features']}",
        "",
        "| route | utility | NDCG | cost | regret | diff vs fixed full | choices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in public_routes:
        ci = r.get("paired_diff_vs_fixed_full")
        diff = "--" if ci is None else f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items() if v)
        rule = f"; {r['rule']}" if "rule" in r else ""
        lines.append(f"| {r['name']} | {r['utility']:.4f} | {r['ndcg']:.4f} | {r['cost']:.4f} | {r['regret']:.4f} | {diff} | {ch}{rule} |")
    lines += [
        "",
        "## Cost-sensitive selective reranking baselines",
        "",
        "| policy | utility | NDCG | cost | diff vs fixed full | choices |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in sorted(selective_routes, key=lambda z: z["utility"], reverse=True)[:12]:
        ci = r.get("paired_diff_vs_fixed_full")
        diff = "--" if ci is None else f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items() if v)
        rule = f"; {r['rule']}" if "rule" in r else ""
        lines.append(f"| {r['name']} | {r['utility']:.4f} | {r['ndcg']:.4f} | {r['cost']:.4f} | {diff} | {ch}{rule} |")
    lines += [
        "",
        "## Feature ablation on the displayed split",
        "",
        "| feature set | utility | NDCG | cost | diff vs fixed full | choices |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in sorted(feature_ablation, key=lambda z: z["utility"], reverse=True):
        ci = r.get("paired_diff_vs_fixed_full")
        diff = "--" if ci is None else f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items() if v)
        lines.append(f"| {r['name']} | {r['utility']:.4f} | {r['ndcg']:.4f} | {r['cost']:.4f} | {diff} | {ch} |")
    lines += [
        "",
        "## Lambda frontier for fixed views and oracle routing",
        "",
        "| lambda | fixed S | fixed PQ | fixed F | best fixed | oracle | oracle choices |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for r in out["lambda_frontier"]:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["oracle_choices"].items() if v)
        lines.append(
            f"| {r['lambda']:.2f} | {r['fixed_summary']:.4f} | {r['fixed_pq']:.4f} | {r['fixed_full']:.4f} | "
            f"{r['best_fixed']} | {r['oracle']:.4f} | {ch} |"
        )
    lines += [
        "",
        "## Per-dataset read of the displayed split",
        "",
        "| dataset | test q | fixed S | fixed PQ | fixed F | adaptive | diff vs F | choices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in per_dataset:
        ci = r["paired_diff_vs_fixed_full"]
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["adaptive_choices"].items() if v)
        lines.append(
            f"| {r['dataset']} | {r['test_queries']} | {r['fixed_summary']:.4f} | {r['fixed_pq']:.4f} | "
            f"{r['fixed_full']:.4f} | {r['adaptive_utility']:.4f} | "
            f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}] | {ch} |"
        )
    lines += [
        "",
        "## Split stability",
        "",
        f"- Repeated 60/40 splits: {split_summary['reps']}",
        f"- Best adaptive minus best fixed utility: {split_summary['mean_adaptive_minus_best_fixed']:+.4f} "
        f"[{split_summary['lo']:+.4f}, {split_summary['hi']:+.4f}]",
        f"- Positive split share: {split_summary['positive_share']:.3f}",
        "",
        "## Leave-one-dataset-out stress",
        "",
        "| held-out dataset | best fixed | best adaptive | delta | adaptive choices |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for r in leave_one_dataset_out:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["best_adaptive_choices"].items() if v)
        lines.append(
            f"| {r['heldout_dataset']} | {r['best_fixed']} {r['best_fixed_utility']:.4f} | "
            f"{r['best_adaptive']} {r['best_adaptive_utility']:.4f} | "
            f"{r['adaptive_minus_best_fixed']:+.4f} | {ch} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps(out, indent=2)[:5000])


if __name__ == "__main__":
    main()
