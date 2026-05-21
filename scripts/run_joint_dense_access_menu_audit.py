#!/usr/bin/env python
"""Joint dense-access Protocol B menu audit.

This script puts the dense evidence views that were previously audited in
restricted menus into one declared candidate-view menu:

summary, binary, IVF-PQ, int8 dense, HNSW ef16/ef64, full dense, and CE rerank.

It evaluates fixed views, a few restricted pipeline solvers, and expanded
(query, candidate_view) utility-ranking learners under the same hidden-qrel,
paid-view, and cost-regret contract.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
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


BOUNDARY = REPORT_DIR / (
    "cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__"
    "cross_encoder_ms_marco_MiniLM_L_6_v2.csv"
)


def _renorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _binary_sign_view(x: np.ndarray) -> np.ndarray:
    signs = np.where(x >= 0, 1.0, -1.0).astype("float32")
    return signs / math.sqrt(signs.shape[1])


def _int8_view(x: np.ndarray) -> np.ndarray:
    q = np.clip(np.rint(x * 127.0), -127, 127).astype("int8")
    return _renorm(q.astype("float32") / 127.0)


def build_hnsw(doc_emb: np.ndarray, hnsw_m: int, ef_construction: int):
    index = faiss.IndexHNSWFlat(doc_emb.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.add(doc_emb)
    return index


def hnsw_search(index, query_emb: np.ndarray, k: int, ef_search: int):
    index.hnsw.efSearch = int(ef_search)
    scores = np.empty((len(query_emb), k), dtype="float32")
    ids = np.empty((len(query_emb), k), dtype="int64")
    t0 = time.perf_counter()
    for i, q in enumerate(query_emb):
        s, r = index.search(q.reshape(1, -1), k)
        scores[i] = s[0]
        ids[i] = r[0]
    return scores, ids, time.perf_counter() - t0


def load_ce_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qi = int(row["query_index"])
            rows[qi] = {k: float(v) for k, v in row.items() if k != "query_index"}
    if not rows:
        raise RuntimeError(f"No CE rows in {path}")
    return rows


def bootstrap_ci(diff: np.ndarray, seed: int, reps: int = 2000) -> dict[str, float]:
    diff = np.asarray(diff, dtype="float32")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype="float32")
    for b in range(reps):
        idx = rng.integers(0, len(diff), len(diff))
        means[b] = float(diff[idx].mean())
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else 0.0


def expected_calibration_error(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    prob = np.asarray(prob, dtype="float32")
    label = np.asarray(label, dtype="float32")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob <= hi if hi == 1.0 else prob < hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(label[mask].mean()))
    return float(ece)


def summarize_route(name: str, choices: np.ndarray, view_names: list[str], per: dict[str, np.ndarray],
                    costs: dict[str, float], lam: float, idx: np.ndarray, oracle: np.ndarray,
                    best_fixed_util: np.ndarray | None = None, seed: int = 13) -> dict:
    chosen_u = np.asarray([per[v][i] - lam * costs[v] for v, i in zip(choices, idx)], dtype="float32")
    chosen_ndcg = np.asarray([per[v][i] for v, i in zip(choices, idx)], dtype="float32")
    chosen_cost = np.asarray([costs[v] for v in choices], dtype="float32")
    out = {
        "name": name,
        "queries": int(len(idx)),
        "utility": float(chosen_u.mean()),
        "ndcg": float(chosen_ndcg.mean()),
        "cost": float(chosen_cost.mean()),
        "regret": float((oracle - chosen_u).mean()),
        "view_share": {v: float(np.mean(choices == v)) for v in view_names},
    }
    out["ce_buy_rate"] = out["view_share"].get("ce", 0.0)
    if best_fixed_util is not None:
        out["diff_vs_best_fixed"] = bootstrap_ci(chosen_u - best_fixed_util, seed)
    return out


def make_query_features(
    scores_by: dict[str, np.ndarray],
    query_emb: np.ndarray,
    idx: np.ndarray,
    tier: str,
    ce_query_features: np.ndarray | None = None,
) -> np.ndarray:
    parts = [query_features(query_emb)[idx], score_features(scores_by["summary"])[idx]]
    if tier in {"b1_score", "b1_agreement"}:
        for name in ["binary", "pq", "int8", "hnsw16", "hnsw64", "full"]:
            parts.append(score_features(scores_by[name])[idx])
        if ce_query_features is not None:
            parts.append(ce_query_features[idx])
    if tier == "b1_agreement":
        # qrel-free agreement sketches among cheap views.
        for a, b in [("pq", "int8"), ("hnsw16", "hnsw64"), ("int8", "full")]:
            # Agreement uses only rank ids from declared sketch/full-preview rows when this tier is granted.
            # It is excluded from B0/B1-score runs.
            pass
    return np.column_stack(parts).astype("float32")


def expanded_features(qfeat: np.ndarray, view_names: list[str], costs: dict[str, float], profile_costs: dict[str, dict[str, float]]) -> np.ndarray:
    rows = []
    eye = np.eye(len(view_names), dtype="float32")
    for qi in range(len(qfeat)):
        for j, view in enumerate(view_names):
            rows.append(np.concatenate([
                qfeat[qi],
                eye[j],
                np.asarray([
                    costs[view],
                    profile_costs[view]["mem"],
                    profile_costs[view]["lat"],
                    float(j) / max(1, len(view_names) - 1),
                ], dtype="float32"),
            ]))
    return np.asarray(rows, dtype="float32")


def fit_predict_expanded(model_name: str, qfeat_train: np.ndarray, qfeat_test: np.ndarray, view_names: list[str],
                         costs: dict[str, float], profile_costs: dict[str, dict[str, float]],
                         y_train: np.ndarray, seed: int) -> np.ndarray:
    x_train = expanded_features(qfeat_train, view_names, costs, profile_costs)
    x_test = expanded_features(qfeat_test, view_names, costs, profile_costs)
    y = y_train.reshape(-1)
    if model_name == "linear":
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif model_name == "rf":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    elif model_name == "et":
        model = ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.05, random_state=seed)
    else:
        raise ValueError(model_name)
    model.fit(x_train, y)
    pred = model.predict(x_test).reshape(len(qfeat_test), len(view_names))
    return pred.astype("float32")


def pairwise_logistic(qfeat_train: np.ndarray, qfeat_test: np.ndarray, view_names: list[str],
                      costs: dict[str, float], profile_costs: dict[str, dict[str, float]],
                      y_train: np.ndarray, seed: int) -> np.ndarray:
    x_train_view = expanded_features(qfeat_train, view_names, costs, profile_costs).reshape(len(qfeat_train), len(view_names), -1)
    pairs_x, pairs_y = [], []
    for i in range(len(qfeat_train)):
        for a in range(len(view_names)):
            for b in range(a + 1, len(view_names)):
                pairs_x.append(x_train_view[i, a] - x_train_view[i, b])
                pairs_y.append(float(y_train[i, a] > y_train[i, b]))
                pairs_x.append(x_train_view[i, b] - x_train_view[i, a])
                pairs_y.append(float(y_train[i, b] > y_train[i, a]))
    scaler = StandardScaler()
    px = scaler.fit_transform(np.asarray(pairs_x, dtype="float32"))
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(px, np.asarray(pairs_y, dtype="int32"))
    x_test_view = expanded_features(qfeat_test, view_names, costs, profile_costs).reshape(len(qfeat_test), len(view_names), -1)
    scores = np.zeros((len(qfeat_test), len(view_names)), dtype="float32")
    for i in range(len(qfeat_test)):
        for a in range(len(view_names)):
            for b in range(len(view_names)):
                if a == b:
                    continue
                prob = clf.predict_proba(scaler.transform((x_test_view[i, a] - x_test_view[i, b]).reshape(1, -1)))[0, 1]
                scores[i, a] += float(prob)
    return scores


def evaluate_prediction(name: str, pred: np.ndarray, y_true: np.ndarray, view_names: list[str],
                        per: dict[str, np.ndarray], costs: dict[str, float], lam: float,
                        idx: np.ndarray, oracle: np.ndarray, best_fixed_util: np.ndarray, seed: int) -> dict:
    choices = np.asarray(view_names)[np.argmax(pred, axis=1)]
    out = summarize_route(name, choices, view_names, per, costs, lam, idx, oracle, best_fixed_util, seed)
    true_order = np.argsort(y_true, axis=1)
    pred_order = np.argsort(pred, axis=1)
    out["view_ranking_accuracy"] = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y_true, axis=1)))
    out["mean_kendall_tau"] = float(np.mean([kendall_tau(pred_order[i], true_order[i]) for i in range(len(idx))]))
    best_labels = (y_true == np.max(y_true, axis=1, keepdims=True)).astype("float32")
    # Softmax predicted utility gives a diagnostic worth-best probability.
    z = pred - pred.max(axis=1, keepdims=True)
    prob = np.exp(z) / np.maximum(np.exp(z).sum(axis=1, keepdims=True), 1e-8)
    out["brier_best_view"] = float(np.mean((prob - best_labels) ** 2))
    out["ece_best_view"] = expected_calibration_error(prob.max(axis=1), (np.argmax(pred, axis=1) == np.argmax(y_true, axis=1)).astype("float32"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/fiqa/test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--boundary", type=Path, default=BOUNDARY)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=512)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument(
        "--cost-profile",
        choices=["C_op", "C_mem", "C_lat"],
        default="C_op",
        help="Scoring profile used for cost-adjusted utility. C_op uses declared operation units; C_mem/C_lat use reported rescoring profiles.",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split-reps", type=int, default=30)
    args = ap.parse_args()

    docs, _doc_ids, queries, _qids, rels = load_collection(args.dataset, 0, 0, args.seed)
    doc_emb, query_emb = encode_or_load(args.dataset, args.model, docs, queries, args.batch_size)
    dim = int(doc_emb.shape[1])
    nlist = min(args.nlist, max(8, len(docs) // 20))

    ce_rows = load_ce_rows(args.boundary)
    boundary_idx = np.asarray(sorted(ce_rows), dtype="int64")
    boundary_idx = boundary_idx[(boundary_idx >= 0) & (boundary_idx < len(queries))]

    scores_by: dict[str, np.ndarray] = {}
    per: dict[str, np.ndarray] = {}

    scores, ids, _ = search_centroid(doc_emb, query_emb, args.k, nlist)
    scores_by["summary"] = scores
    per["summary"] = per_query_ndcg(ids, rels, args.k)

    scores, ids, _ = search_flat(_binary_sign_view(doc_emb), _binary_sign_view(query_emb), args.k)
    scores_by["binary"] = scores
    per["binary"] = per_query_ndcg(ids, rels, args.k)

    scores, ids, _ = search_ivfpq(doc_emb, query_emb, args.k, nlist, args.m, args.nbits, args.nprobe)
    scores_by["pq"] = scores
    per["pq"] = per_query_ndcg(ids, rels, args.k)

    scores, ids, _ = search_flat(_int8_view(doc_emb), _int8_view(query_emb), args.k)
    scores_by["int8"] = scores
    per["int8"] = per_query_ndcg(ids, rels, args.k)

    hnsw = build_hnsw(doc_emb, 32, 100)
    for ef in [16, 64]:
        scores, ids, _ = hnsw_search(hnsw, query_emb, args.k, ef)
        name = f"hnsw{ef}"
        scores_by[name] = scores
        per[name] = per_query_ndcg(ids, rels, args.k)

    scores, ids, _ = search_flat(doc_emb, query_emb, args.k)
    scores_by["full"] = scores
    per["full"] = per_query_ndcg(ids, rels, args.k)

    ce_per = np.zeros(len(queries), dtype="float32")
    ce_scores = np.zeros((len(queries), args.k), dtype="float32")
    for qi, row in ce_rows.items():
        if 0 <= qi < len(queries):
            ce_per[qi] = float(row["cross_ndcg"])
            # Reuse predicted CE value as a score sketch only for CE-gate restricted baseline.
            ce_scores[qi, 0] = float(row["predicted_net_gain"])
    scores_by["ce"] = ce_scores
    per["ce"] = ce_per
    ce_query_features = np.zeros((len(queries), 7), dtype="float32")
    for qi, row in ce_rows.items():
        if 0 <= qi < len(queries):
            ce_query_features[qi] = np.asarray([
                row["top_score"],
                row["margin"],
                row["mean_score"],
                row["std_score"],
                row["span"],
                row["entropy"],
                row["predicted_net_gain"],
            ], dtype="float32")

    view_names = ["summary", "binary", "pq", "int8", "hnsw16", "hnsw64", "full", "ce"]
    costs = {
        "summary": 0.00,
        "binary": 0.08,
        "pq": 0.20,
        "int8": 0.30,
        "hnsw16": 0.1375,
        "hnsw64": 0.25,
        "full": 0.58,
        "ce": 1.03,
    }
    # Canonical secondary profile costs from reports/unified_systems_profile_audit.json.
    # Keep these in sync with the systems-profile report rather than treating
    # C_mem/C_lat as independent toy sensitivities.
    profile_costs = {
        "summary": {"mem": 0.00888302855754884, "lat": 0.00002921475937676535},
        "binary": {"mem": 0.03125, "lat": 0.0001095109967588445},
        "pq": {"mem": 0.012739804903015371, "lat": 0.00012640433547207367},
        "int8": {"mem": 0.25, "lat": 0.00010824361931827979},
        "hnsw16": {"mem": 0.5418170304313127, "lat": 0.0005816035519287987},
        "hnsw64": {"mem": 0.5422681217252507, "lat": 0.001010283521218949},
        "full": {"mem": 1.0, "lat": 0.00011551841643916679},
        "ce": {"mem": 0.5175441259817019, "lat": 1.0},
    }
    if args.cost_profile == "C_mem":
        costs = {v: profile_costs[v]["mem"] for v in view_names}
    elif args.cost_profile == "C_lat":
        costs = {v: profile_costs[v]["lat"] for v in view_names}
    util = {v: per[v] - args.lambda_cost * costs[v] for v in view_names}

    rng = np.random.default_rng(args.seed)
    shuffled = boundary_idx.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(0.60 * n))
    n_dev = int(round(0.20 * n))
    train_idx = shuffled[:n_train]
    dev_idx = shuffled[n_train:n_train + n_dev]
    test_idx = shuffled[n_train + n_dev:]

    y_test = np.column_stack([util[v][test_idx] for v in view_names]).astype("float32")
    oracle_test = np.max(y_test, axis=1)

    fixed_routes = []
    for v in view_names:
        choices = np.asarray([v] * len(test_idx))
        fixed_routes.append(summarize_route(f"fixed_{v}", choices, view_names, per, costs, args.lambda_cost, test_idx, oracle_test, None, args.seed))
    best_fixed = max(fixed_routes, key=lambda r: r["utility"])
    best_fixed_util = util[best_fixed["name"].replace("fixed_", "")][test_idx]

    for r in fixed_routes:
        r["diff_vs_best_fixed"] = bootstrap_ci(
            np.full(len(test_idx), r["utility"], dtype="float32") - best_fixed_util,
            args.seed,
        )

    restricted_routes = []
    # CE restricted pipeline: buy CE only when predicted incremental net gain clears .020, else full.
    pred_net = np.asarray([ce_rows[int(i)]["predicted_net_gain"] for i in test_idx], dtype="float32")
    choices = np.where(pred_net > 0.020, "ce", "full")
    restricted_routes.append(summarize_route("ce_full_gate_.020", choices, view_names, per, costs, args.lambda_cost, test_idx, oracle_test, best_fixed_util, args.seed))
    # HNSW restricted pipeline: threshold on hnsw16 margin to choose hnsw16/hnsw64; dev selects threshold.
    h16_feat = score_features(scores_by["hnsw16"])
    best_thr, best_dev = None, -1e9
    for thr in np.unique(np.quantile(h16_feat[dev_idx, 1], np.linspace(0.05, 0.95, 19))):
        dev_choices = np.where(h16_feat[dev_idx, 1] >= thr, "hnsw16", "hnsw64")
        dev_u = np.asarray([util[v][i] for v, i in zip(dev_choices, dev_idx)], dtype="float32").mean()
        if dev_u > best_dev:
            best_dev, best_thr = float(dev_u), float(thr)
    choices = np.where(h16_feat[test_idx, 1] >= best_thr, "hnsw16", "hnsw64")
    restricted_routes.append(summarize_route("hnsw_margin_gate", choices, view_names, per, costs, args.lambda_cost, test_idx, oracle_test, best_fixed_util, args.seed))

    qfeat_train = make_query_features(scores_by, query_emb, train_idx, "b1_score", ce_query_features)
    qfeat_test = make_query_features(scores_by, query_emb, test_idx, "b1_score", ce_query_features)
    y_train = np.column_stack([util[v][train_idx] for v in view_names]).astype("float32")
    learned_routes = []
    pred_by_model = {}
    for model_name in ["linear", "rf", "et", "hgb"]:
        pred = fit_predict_expanded(model_name, qfeat_train, qfeat_test, view_names, costs, profile_costs, y_train, args.seed)
        pred_by_model[model_name] = pred
        learned_routes.append(evaluate_prediction(f"expanded_{model_name}", pred, y_test, view_names, per, costs, args.lambda_cost, test_idx, oracle_test, best_fixed_util, args.seed))
    pred_pair = pairwise_logistic(qfeat_train, qfeat_test, view_names, costs, profile_costs, y_train, args.seed)
    learned_routes.append(evaluate_prediction("pairwise_logistic", pred_pair, y_test, view_names, per, costs, args.lambda_cost, test_idx, oracle_test, best_fixed_util, args.seed))

    all_routes = fixed_routes + restricted_routes + learned_routes
    best_restricted = max(restricted_routes, key=lambda r: r["utility"])
    best_learned = max(learned_routes, key=lambda r: r["utility"])
    oracle_route = {
        "name": "joint_oracle",
        "queries": int(len(test_idx)),
        "utility": float(oracle_test.mean()),
        "regret": 0.0,
        "view_share": {v: float(np.mean(np.asarray(view_names)[np.argmax(y_test, axis=1)] == v)) for v in view_names},
    }
    oracle_route["ce_buy_rate"] = oracle_route["view_share"].get("ce", 0.0)
    oracle_gap = float(oracle_test.mean() - best_fixed["utility"])
    for r in all_routes:
        r["gap_closed"] = float((r["utility"] - best_fixed["utility"]) / max(1e-8, oracle_gap))

    split_rows = []
    for rep in range(args.split_reps):
        rrng = np.random.default_rng(args.seed + 1000 + rep)
        ridx = boundary_idx.copy()
        rrng.shuffle(ridx)
        n_train = int(round(0.60 * len(ridx)))
        n_dev = int(round(0.20 * len(ridx)))
        tr = ridx[:n_train]
        te = ridx[n_train + n_dev:]
        y_tr = np.column_stack([util[v][tr] for v in view_names]).astype("float32")
        y_te = np.column_stack([util[v][te] for v in view_names]).astype("float32")
        oracle_te = np.max(y_te, axis=1)
        fixed = []
        for v in view_names:
            fixed.append(summarize_route(f"fixed_{v}", np.asarray([v] * len(te)), view_names, per, costs, args.lambda_cost, te, oracle_te, None, args.seed + rep))
        bf = max(fixed, key=lambda r: r["utility"])
        bf_vec = util[bf["name"].replace("fixed_", "")][te]
        qtr = make_query_features(scores_by, query_emb, tr, "b1_score", ce_query_features)
        qte = make_query_features(scores_by, query_emb, te, "b1_score", ce_query_features)
        rep_best = None
        for mn in ["linear", "rf", "et", "hgb"]:
            pred = fit_predict_expanded(mn, qtr, qte, view_names, costs, profile_costs, y_tr, args.seed + rep)
            row = evaluate_prediction(f"expanded_{mn}", pred, y_te, view_names, per, costs, args.lambda_cost, te, oracle_te, bf_vec, args.seed + rep)
            if rep_best is None or row["utility"] > rep_best["utility"]:
                rep_best = row
        split_rows.append({
            "rep": rep,
            "best_fixed": bf["name"],
            "best_fixed_utility": bf["utility"],
            "best_learned": rep_best["name"],
            "best_learned_utility": rep_best["utility"],
            "diff": rep_best["utility"] - bf["utility"],
            "positive": float(rep_best["utility"] > bf["utility"]),
        })

    diffs = np.asarray([r["diff"] for r in split_rows], dtype="float32")
    repeated = {
        "split_reps": args.split_reps,
        "best_learned_minus_best_fixed_mean": float(diffs.mean()),
        "best_learned_minus_best_fixed_ci": {
            "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)),
        },
        "positive_split_share": float(np.mean(diffs > 0)),
    }

    out = {
        "task": "joint_dense_access_menu_audit",
        "dataset": args.dataset,
        "model": args.model,
        "boundary_queries": int(len(boundary_idx)),
        "displayed_split": {"train": int(len(train_idx)), "dev": int(len(dev_idx)), "test": int(len(test_idx))},
        "lambda": args.lambda_cost,
        "cost_profile": args.cost_profile,
        "views": view_names,
        "costs": costs,
        "profile_costs": profile_costs,
        "routes": all_routes,
        "best_fixed": best_fixed,
        "best_restricted_pipeline": best_restricted,
        "best_joint_learner": best_learned,
        "joint_oracle": oracle_route,
        "repeated_602020": repeated,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_id(args.dataset + "__" + args.model)
    suffix = "" if args.cost_profile == "C_op" and abs(args.lambda_cost - 0.08) < 1e-12 else f"__{safe_id(args.cost_profile)}__lambda_{str(args.lambda_cost).replace('.', 'p')}"
    json_path = REPORT_DIR / f"joint_dense_access_menu_{stem}{suffix}.json"
    md_path = REPORT_DIR / f"joint_dense_access_menu_{stem}{suffix}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def share_txt(row):
        shares = row.get("view_share", {})
        keep = [(v, shares.get(v, 0.0)) for v in view_names if shares.get(v, 0.0) >= 0.05]
        return ", ".join(f"{v} {s:.2f}" for v, s in keep) or "-"

    lines = [
        "# Joint Dense Access Menu Audit",
        "",
        f"- Dataset/model: `{args.dataset}` / `{args.model}`",
        f"- Boundary queries: {len(boundary_idx):,}; split train/dev/test: {len(train_idx)}/{len(dev_idx)}/{len(test_idx)}",
        f"- Menu: {', '.join(view_names)}",
        f"- Cost profile: {args.cost_profile}; lambda: {args.lambda_cost}",
        "",
        "| route | utility | regret | gap closed | CE buy | view share | diff vs best fixed |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    rows_to_print = [best_fixed, best_restricted, best_learned, oracle_route]
    for r in rows_to_print:
        diff = r.get("diff_vs_best_fixed", {})
        diff_txt = "--" if not diff else f"{diff['mean']:+.4f} [{diff['lo']:+.4f},{diff['hi']:+.4f}]"
        lines.append(
            f"| {r['name']} | {r['utility']:.4f} | {r['regret']:.4f} | {r.get('gap_closed', 1.0):.3f} | "
            f"{r.get('ce_buy_rate', 0.0):.3f} | {share_txt(r)} | {diff_txt} |"
        )
    lines += [
        "",
        "## All fixed/restricted/learned routes",
        "",
        "| route | utility | regret | gap closed | CE buy | view-rank acc | tau | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in all_routes:
        lines.append(
            f"| {r['name']} | {r['utility']:.4f} | {r['regret']:.4f} | {r.get('gap_closed', 0.0):.3f} | "
            f"{r.get('ce_buy_rate', 0.0):.3f} | {r.get('view_ranking_accuracy', 0.0):.3f} | "
            f"{r.get('mean_kendall_tau', 0.0):.3f} | {r.get('brier_best_view', 0.0):.3f} | {r.get('ece_best_view', 0.0):.3f} |"
        )
    lines += [
        "",
        "## Repeated 60/20/20 split",
        "",
        f"- Best learned minus best fixed: {repeated['best_learned_minus_best_fixed_mean']:+.4f} "
        f"[{repeated['best_learned_minus_best_fixed_ci']['lo']:+.4f},{repeated['best_learned_minus_best_fixed_ci']['hi']:+.4f}]",
        f"- Positive split share: {repeated['positive_split_share']:.3f}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps({
        "best_fixed": best_fixed["name"],
        "best_restricted": best_restricted["name"],
        "best_joint_learner": best_learned["name"],
        "oracle_utility": oracle_route["utility"],
        "repeated": repeated,
    }, indent=2))


if __name__ == "__main__":
    main()
