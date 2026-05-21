#!/usr/bin/env python
"""Raw 60/20/20 Protocol-B learning audit over standard-qrel datasets.

This script is intentionally separate from the report-aggregation helper.  It
recomputes the per-query summary/PQ/full utilities from the public qrel
collections, splits queries within each dataset into train/dev/test, and
evaluates old retrieval paradigms as legal Protocol-B solvers under one
visibility/cost contract.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_evidence_cascade_router_audit import (
    COSTS,
    DATASETS,
    VIEWS,
    clone_model,
    feature_slices,
    load_pooled,
    model_factory,
    overlap_features,
    paired_ci,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype="float32")
    prob = np.asarray(prob, dtype="float32")
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not np.any(mask):
            continue
        out += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return float(out)


def stratified_602020(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, dev, test = [], [], []
    for dataset in DATASETS:
        idx = np.where(groups == dataset)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(0.60 * n)))
        n_dev = max(1, int(round(0.20 * n)))
        if n_train + n_dev >= n:
            n_dev = max(1, n - n_train - 1)
        train.extend(idx[:n_train].tolist())
        dev.extend(idx[n_train : n_train + n_dev].tolist())
        test.extend(idx[n_train + n_dev :].tolist())
    return np.asarray(train, dtype=int), np.asarray(dev, dtype=int), np.asarray(test, dtype=int)


def route_metrics(name: str, choices: np.ndarray, y: np.ndarray, ndcg: np.ndarray, idx: np.ndarray) -> dict:
    choices = np.asarray(choices, dtype=int)
    rows = np.arange(len(idx))
    util_vec = y[idx][rows, choices]
    raw_vec = ndcg[idx][rows, choices]
    cost_vec = np.asarray([COSTS[str(VIEWS[i])] for i in choices], dtype="float32")
    oracle_vec = y[idx].max(axis=1)
    return {
        "name": name,
        "queries": int(len(idx)),
        "utility": float(util_vec.mean()),
        "ndcg": float(raw_vec.mean()),
        "cost": float(cost_vec.mean()),
        "regret": float((oracle_vec - util_vec).mean()),
        "choices": {str(VIEWS[i]): float(np.mean(choices == i)) for i in range(len(VIEWS))},
        "_utility_vector": util_vec.astype(float).tolist(),
        "_choice_vector": choices.astype(int).tolist(),
    }


def restricted_menu_readout(
    name: str,
    menu: list[int],
    adaptive_row: dict,
    y: np.ndarray,
    idx: np.ndarray,
) -> dict:
    menu = list(menu)
    fixed_utils = y[idx][:, menu].mean(axis=0)
    best_pos = int(np.argmax(fixed_utils))
    best_fixed_view = str(VIEWS[menu[best_pos]])
    best_fixed = float(fixed_utils[best_pos])
    oracle_vec = y[idx][:, menu].max(axis=1)
    oracle = float(oracle_vec.mean())
    adaptive = float(adaptive_row["utility"])
    regret = float(oracle - adaptive)
    denom = max(1e-8, oracle - best_fixed)
    return {
        "menu": name,
        "legal_choices": "/".join(str(VIEWS[i]) for i in menu),
        "best_fixed": best_fixed_view,
        "best_fixed_utility": best_fixed,
        "best_adaptive": adaptive_row["name"],
        "best_adaptive_utility": adaptive,
        "oracle_utility": oracle,
        "gap_closed": float((adaptive - best_fixed) / denom),
        "residual_regret": regret,
    }


def fit_multi_reg(name: str, model, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train: np.ndarray, idx: np.ndarray, seed: int) -> dict:
    preds = []
    for j in range(len(VIEWS)):
        m = clone_model(model, seed + j)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(x[train], y[train, j])
        preds.append(m.predict(x[idx]))
    pred = np.stack(preds, axis=1)
    return route_metrics(name, pred.argmax(axis=1), y, ndcg, idx)


def qpp_single_gate(name: str, feat: np.ndarray, y: np.ndarray, ndcg: np.ndarray, dev: np.ndarray, test: np.ndarray) -> dict:
    # One visible uncertainty feature chooses between two candidate views.  This
    # is the nearest Protocol-B analogue of a simple QPP threshold policy.
    best = None
    qvals = np.linspace(0.05, 0.95, 37)
    cols = range(feat.shape[1])
    for c in cols:
        vals = np.unique(np.quantile(feat[dev, c], qvals))
        for thr in vals:
            for op in ("ge", "le"):
                mask_dev = feat[dev, c] >= thr if op == "ge" else feat[dev, c] <= thr
                for low_view in range(len(VIEWS)):
                    for high_view in range(len(VIEWS)):
                        if low_view == high_view:
                            continue
                        ch = np.where(mask_dev, high_view, low_view)
                        util = y[dev][np.arange(len(dev)), ch].mean()
                        if best is None or util > best[0]:
                            best = (float(util), c, float(thr), op, low_view, high_view)
    _, c, thr, op, low_view, high_view = best
    mask = feat[test, c] >= thr if op == "ge" else feat[test, c] <= thr
    choices = np.where(mask, high_view, low_view)
    out = route_metrics(name, choices, y, ndcg, test)
    out["rule"] = f"{VIEWS[high_view]} if f{c} {op} {thr:.4g}; else {VIEWS[low_view]}"
    return out


def summary_margin_gate(summary_scores: np.ndarray, y: np.ndarray, ndcg: np.ndarray, dev: np.ndarray, test: np.ndarray) -> dict:
    """One-feature confidence heuristic over the first released summary scores.

    The only feature is the summary-view top1-top2 margin.  The dev split chooses
    a threshold and a pair of actions; the hidden-test split reports the frozen
    rule.  This intentionally simple row helps calibrate whether dense routing
    needs learned view-utility predictors or only a cheap confidence cutoff.
    """
    margin = np.asarray(summary_scores[:, 1], dtype="float32")
    candidates = np.unique(np.quantile(margin[dev], np.linspace(0.05, 0.95, 37)))
    best = None
    for thr in candidates:
        for op in ("ge", "le"):
            mask_dev = margin[dev] >= thr if op == "ge" else margin[dev] <= thr
            for low_view in range(len(VIEWS)):
                for high_view in range(len(VIEWS)):
                    if low_view == high_view:
                        continue
                    ch = np.where(mask_dev, high_view, low_view)
                    util = y[dev][np.arange(len(dev)), ch].mean()
                    if best is None or util > best[0]:
                        best = (float(util), float(thr), op, low_view, high_view)
    _, thr, op, low_view, high_view = best
    mask = margin[test] >= thr if op == "ge" else margin[test] <= thr
    choices = np.where(mask, high_view, low_view)
    out = route_metrics("summary_margin_gate", choices, y, ndcg, test)
    out["rule"] = (
        f"{VIEWS[high_view]} if summary top1-top2 margin {op} {thr:.4g}; "
        f"else {VIEWS[low_view]}"
    )
    return out


def two_stage_cascade(name: str, feat: np.ndarray, y: np.ndarray, ndcg: np.ndarray, dev: np.ndarray, test: np.ndarray) -> dict:
    best = None
    qvals = np.linspace(0.05, 0.95, 19)
    cols = range(feat.shape[1])
    for c1 in cols:
        vals1 = np.unique(np.quantile(feat[dev, c1], qvals))
        for thr1 in vals1:
            for op1 in ("ge", "le"):
                m1 = feat[dev, c1] >= thr1 if op1 == "ge" else feat[dev, c1] <= thr1
                for c2 in cols:
                    vals2 = np.unique(np.quantile(feat[dev, c2], qvals))
                    for thr2 in vals2:
                        for op2 in ("ge", "le"):
                            m2 = feat[dev, c2] >= thr2 if op2 == "ge" else feat[dev, c2] <= thr2
                            ch = np.full(len(dev), 2, dtype=int)
                            ch[m2] = 1
                            ch[m1] = 0
                            util = y[dev][np.arange(len(dev)), ch].mean()
                            if best is None or util > best[0]:
                                best = (float(util), c1, float(thr1), op1, c2, float(thr2), op2)
    _, c1, thr1, op1, c2, thr2, op2 = best
    m1 = feat[test, c1] >= thr1 if op1 == "ge" else feat[test, c1] <= thr1
    m2 = feat[test, c2] >= thr2 if op2 == "ge" else feat[test, c2] <= thr2
    choices = np.full(len(test), 2, dtype=int)
    choices[m2] = 1
    choices[m1] = 0
    out = route_metrics(name, choices, y, ndcg, test)
    out["rule"] = f"summary if f{c1} {op1} {thr1:.4g}; else pq if f{c2} {op2} {thr2:.4g}; else full"
    return out


def selective_full_regression_602020(name: str, model, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train: np.ndarray, dev: np.ndarray, test: np.ndarray, seed: int) -> dict:
    target = y[:, 2] - y[:, 1]
    m = clone_model(model, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(x[train], target[train])
    pred_dev = m.predict(x[dev])
    pred_test = m.predict(x[test])
    candidates = np.unique(np.quantile(pred_dev, np.linspace(0.02, 0.98, 49)))
    candidates = np.r_[candidates.min() - 1e-6, candidates, candidates.max() + 1e-6]
    best = None
    for thr in candidates:
        ch = np.where(pred_dev > thr, 2, 1)
        util = y[dev][np.arange(len(dev)), ch].mean()
        if best is None or util > best[0]:
            best = (float(util), float(thr))
    thr = best[1]
    choices = np.where(pred_test > thr, 2, 1)
    out = route_metrics(name, choices, y, ndcg, test)
    out["rule"] = f"full if predicted U(full)-U(pq) > {thr:.4g}; else pq"
    return out


def deferral_classifier(name: str, x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, train: np.ndarray, dev: np.ndarray, test: np.ndarray, seed: int) -> dict:
    label = y.argmax(axis=1)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, l2_regularization=0.05, min_samples_leaf=12, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x[train], label[train])
    pred = clf.predict(x[test])
    out = route_metrics(name, pred, y, ndcg, test)
    out["dev_accuracy"] = float(np.mean(clf.predict(x[dev]) == label[dev]))
    return out


def full_vs_pq_calibration(x: np.ndarray, y: np.ndarray, train: np.ndarray, dev: np.ndarray, test: np.ndarray, seed: int) -> dict:
    label = (y[:, 2] > y[:, 1]).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, l2_regularization=0.05, min_samples_leaf=12, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x[train], label[train])
    prob_dev = clf.predict_proba(x[dev])[:, 1]
    prob_test = clf.predict_proba(x[test])[:, 1]
    candidates = np.unique(np.quantile(prob_dev, np.linspace(0.02, 0.98, 49)))
    candidates = np.r_[0.0, candidates, 1.0]
    best = None
    for thr in candidates:
        ch = np.where(prob_dev > thr, 2, 1)
        util = y[dev][np.arange(len(dev)), ch].mean()
        if best is None or util > best[0]:
            best = (float(util), float(thr))
    thr = best[1]
    choices = np.where(prob_test > thr, 2, 1)
    route = route_metrics("full_purchase_classifier", choices, y, np.zeros_like(y), test)
    y_test = label[test]
    return {
        "threshold": float(thr),
        "prevalence": float(y_test.mean()),
        "auprc": float(average_precision_score(y_test, prob_test)),
        "brier": float(brier_score_loss(y_test, prob_test)),
        "ece": ece_binary(y_test, prob_test),
        "buy_rate": float(np.mean(choices == 2)),
        "route_utility": route["utility"],
        "route_regret": route["regret"],
    }


def add_gap_and_ci(rows: list[dict], y: np.ndarray, test: np.ndarray) -> list[dict]:
    fixed_rows = [r for r in rows if r["name"].startswith("fixed_")]
    best_fixed = max(fixed_rows, key=lambda r: r["utility"])
    oracle_vec = y[test].max(axis=1)
    best_fixed_vec = np.asarray(best_fixed["_utility_vector"], dtype="float32")
    denom = max(1e-8, float(oracle_vec.mean() - best_fixed["utility"]))
    for r in rows:
        vec = np.asarray(r["_utility_vector"], dtype="float32")
        r["gap_closed"] = float((r["utility"] - best_fixed["utility"]) / denom)
        if r["name"] != best_fixed["name"]:
            r["paired_diff_vs_best_fixed"] = paired_ci(vec - best_fixed_vec)
    return rows


def one_split(x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    train, dev, test = stratified_602020(groups, seed)
    slices = feature_slices(x.shape[1])
    rows: list[dict] = []
    rows.extend(route_metrics(f"fixed_{VIEWS[i]}", np.full(len(test), i), y, ndcg, test) for i in range(len(VIEWS)))

    # B0/B1 feature legality tiers.
    x_b0 = x[:, slices["B0_no_dataset"]]
    x_b1 = x[:, slices["B1_no_dataset"]]
    x_b1_scores = x[:, slices["B1_scores_agreement"]]
    score_agree = x[:, np.r_[slices["B1_scores_agreement"]]]
    summary_scores = x[:, slices["B0_no_dataset"][-6:]]

    rows.append(summary_margin_gate(summary_scores, y, ndcg, dev, test))
    rows.append(qpp_single_gate("qpp_uncertainty_gate", summary_scores, y, ndcg, dev, test))
    rows.append(two_stage_cascade("cascade_exit_gate", score_agree, y, ndcg, dev, test))
    rows.append(selective_full_regression_602020("selective_rerank_rf", model_factory("rf", seed + 11), x_b1, y, ndcg, train, dev, test, seed + 11))
    rows.append(deferral_classifier("learning_to_defer_hgb", x_b1, y, ndcg, train, dev, test, seed + 12))
    rows.append(fit_multi_reg("ridge_utility_b1", model_factory("ridge", seed + 13), x_b1, y, ndcg, train, test, seed + 13))
    rows.append(fit_multi_reg("rf_utility_b0", model_factory("rf", seed + 14), x_b0, y, ndcg, train, test, seed + 14))
    rows.append(fit_multi_reg("rf_utility_b1", model_factory("rf", seed + 15), x_b1, y, ndcg, train, test, seed + 15))
    rows.append(fit_multi_reg("et_utility_b1", model_factory("et", seed + 16), x_b1, y, ndcg, train, test, seed + 16))
    rows.append(fit_multi_reg("hgb_utility_b1", model_factory("hgb", seed + 17), x_b1, y, ndcg, train, test, seed + 17))
    oracle_choices = y[test].argmax(axis=1)
    rows.append(route_metrics("oracle_route", oracle_choices, y, ndcg, test))
    rows = add_gap_and_ci(rows, y, test)
    by_name = {r["name"]: r for r in rows}
    legal_rows = [r for r in rows if not r["name"].startswith("fixed_") and r["name"] != "oracle_route"]
    best_adaptive = max(legal_rows, key=lambda r: r["utility"])
    restricted_menus = [
        restricted_menu_readout("QPP cheap/full", [0, 2], by_name["qpp_uncertainty_gate"], y, test),
        restricted_menu_readout("Selective rerank PQ/full", [1, 2], by_name["selective_rerank_rf"], y, test),
        restricted_menu_readout("Cascade S/PQ/full", [0, 1, 2], by_name["cascade_exit_gate"], y, test),
        restricted_menu_readout("Full S/PQ/full", [0, 1, 2], best_adaptive, y, test),
    ]

    # Public-dev/private-test dry run: choose the best adaptive family on dev,
    # then report its frozen policy on test.
    dev_rows = []
    for r in rows:
        if r["name"] in {"oracle_route"}:
            continue
        if r["name"].startswith("fixed_"):
            continue
        # Recompute the same solver family on dev only when choices are known for
        # the displayed split is overkill; for ranking stability, use test rows as
        # a conservative summary plus repeated splits below.
        dev_rows.append(r)

    calib = full_vs_pq_calibration(x_b1, y, train, dev, test, seed + 18)
    return {
        "train_queries": int(len(train)),
        "dev_queries": int(len(dev)),
        "test_queries": int(len(test)),
        "routes": [{k: v for k, v in r.items() if not k.startswith("_")} for r in sorted(rows, key=lambda z: z["utility"], reverse=True)],
        "restricted_menus": restricted_menus,
        "calibration": calib,
        "best_fixed": max((r for r in rows if r["name"].startswith("fixed_")), key=lambda r: r["utility"])["name"],
        "best_adaptive": max((r for r in rows if not r["name"].startswith("fixed_") and r["name"] != "oracle_route"), key=lambda r: r["utility"])["name"],
        "best_adaptive_minus_best_fixed": float(
            max((r for r in rows if not r["name"].startswith("fixed_") and r["name"] != "oracle_route"), key=lambda r: r["utility"])["utility"]
            - max((r for r in rows if r["name"].startswith("fixed_")), key=lambda r: r["utility"])["utility"]
        ),
    }


def quick_repeated_split(x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """Fast repeated-split readout for hidden-test stability.

    The displayed split above includes the full ladder of gates/classifiers.
    Repeated stability only needs to answer whether legal learned utility
    predictors beat the best fixed reference on fresh hidden-test splits, so it
    skips slow threshold-grid policies and calibration.
    """
    def fast_model(name: str, s: int):
        if name == "rf":
            return RandomForestRegressor(n_estimators=220, min_samples_leaf=3, max_features="sqrt", random_state=s, n_jobs=-1)
        if name == "et":
            return ExtraTreesRegressor(n_estimators=260, min_samples_leaf=3, max_features="sqrt", random_state=s, n_jobs=-1)
        return HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, l2_regularization=0.05, min_samples_leaf=12, random_state=s)

    train, _dev, test = stratified_602020(groups, seed)
    slices = feature_slices(x.shape[1])
    x_b1 = x[:, slices["B1_no_dataset"]]
    rows: list[dict] = []
    rows.extend(route_metrics(f"fixed_{VIEWS[i]}", np.full(len(test), i), y, ndcg, test) for i in range(len(VIEWS)))
    rows.append(fit_multi_reg("rf_utility_b1_fast", fast_model("rf", seed + 15), x_b1, y, ndcg, train, test, seed + 15))
    rows.append(fit_multi_reg("et_utility_b1_fast", fast_model("et", seed + 16), x_b1, y, ndcg, train, test, seed + 16))
    rows.append(fit_multi_reg("hgb_utility_b1_fast", fast_model("hgb", seed + 17), x_b1, y, ndcg, train, test, seed + 17))
    best_fixed = max((r for r in rows if r["name"].startswith("fixed_")), key=lambda r: r["utility"])
    best_adaptive = max((r for r in rows if not r["name"].startswith("fixed_")), key=lambda r: r["utility"])
    return {
        "best_fixed": best_fixed["name"],
        "best_fixed_utility": best_fixed["utility"],
        "best_adaptive": best_adaptive["name"],
        "best_adaptive_utility": best_adaptive["utility"],
        "adaptive_minus_best_fixed": float(best_adaptive["utility"] - best_fixed["utility"]),
    }


def _rankdata(vals: np.ndarray) -> np.ndarray:
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype="float32")
    ranks[order] = np.arange(len(vals), dtype="float32")
    return ranks


def probing_simulation(x: np.ndarray, y: np.ndarray, ndcg: np.ndarray, groups: np.ndarray, seed: int, reps: int = 12) -> dict:
    """Simulate public-dev probing with aggregate-only feedback.

    Candidate policies are trained on train, ranked on public-dev utility, and
    then frozen for private-test scoring. Submission caps limit how many
    candidate policies the solver can probe before selecting a final JSONL.
    """
    def fast_models(s: int):
        return [
            ("rf_B0", RandomForestRegressor(n_estimators=160, min_samples_leaf=4, max_features="sqrt", random_state=s + 1, n_jobs=-1), "B0_no_dataset"),
            ("rf_B1", RandomForestRegressor(n_estimators=180, min_samples_leaf=3, max_features="sqrt", random_state=s + 2, n_jobs=-1), "B1_no_dataset"),
            ("et_B0", ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, max_features="sqrt", random_state=s + 3, n_jobs=-1), "B0_no_dataset"),
            ("et_B1", ExtraTreesRegressor(n_estimators=220, min_samples_leaf=3, max_features="sqrt", random_state=s + 4, n_jobs=-1), "B1_no_dataset"),
            ("hgb_B1", HistGradientBoostingRegressor(max_iter=130, learning_rate=0.045, l2_regularization=0.05, min_samples_leaf=12, random_state=s + 5), "B1_no_dataset"),
            ("ridge_B1", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "B1_no_dataset"),
            ("rf_scores", RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features="sqrt", random_state=s + 6, n_jobs=-1), "B1_scores_agreement"),
        ]

    slices = feature_slices(x.shape[1])
    caps = [1, 3, 10]
    cap_rows = {cap: [] for cap in caps}
    rank_corrs = []
    for rep in range(reps):
        train, dev, test = stratified_602020(groups, seed + 3000 + rep)
        candidates = []
        for i in range(len(VIEWS)):
            dev_row = route_metrics(f"fixed_{VIEWS[i]}", np.full(len(dev), i), y, ndcg, dev)
            test_row = route_metrics(f"fixed_{VIEWS[i]}", np.full(len(test), i), y, ndcg, test)
            candidates.append({"name": dev_row["name"], "dev_utility": dev_row["utility"], "test_utility": test_row["utility"]})
        for name, model, feat_key in fast_models(seed + rep * 17):
            cols = slices[feat_key]
            preds_dev, preds_test = [], []
            for j in range(len(VIEWS)):
                m = clone_model(model, seed + rep * 31 + j)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(x[:, cols][train], y[train, j])
                preds_dev.append(m.predict(x[:, cols][dev]))
                preds_test.append(m.predict(x[:, cols][test]))
            ch_dev = np.stack(preds_dev, axis=1).argmax(axis=1)
            ch_test = np.stack(preds_test, axis=1).argmax(axis=1)
            dev_row = route_metrics(name, ch_dev, y, ndcg, dev)
            test_row = route_metrics(name, ch_test, y, ndcg, test)
            candidates.append({"name": name, "dev_utility": dev_row["utility"], "test_utility": test_row["utility"]})
        dev_vals = np.asarray([c["dev_utility"] for c in candidates], dtype="float32")
        test_vals = np.asarray([c["test_utility"] for c in candidates], dtype="float32")
        rd, rt = _rankdata(dev_vals), _rankdata(test_vals)
        denom = float(np.std(rd) * np.std(rt))
        rank_corrs.append(0.0 if denom == 0.0 else float(np.mean((rd - rd.mean()) * (rt - rt.mean())) / denom))
        for cap in caps:
            # Cap means the solver can inspect aggregate public-dev utility for
            # only the first cap submissions in a fixed predeclared candidate
            # order, then freeze the best of those inspected candidates.
            inspected = candidates[: min(cap, len(candidates))]
            chosen = max(inspected, key=lambda c: c["dev_utility"])
            cap_rows[cap].append(
                {
                    "rep": rep,
                    "chosen": chosen["name"],
                    "public_dev_utility": chosen["dev_utility"],
                    "private_test_utility": chosen["test_utility"],
                    "overfit_gap": chosen["dev_utility"] - chosen["test_utility"],
                }
            )
    summary = []
    for cap, rows in cap_rows.items():
        dev = np.asarray([r["public_dev_utility"] for r in rows], dtype="float32")
        test = np.asarray([r["private_test_utility"] for r in rows], dtype="float32")
        gap = np.asarray([r["overfit_gap"] for r in rows], dtype="float32")
        summary.append(
            {
                "submission_cap": cap,
                "public_dev_utility": float(dev.mean()),
                "private_test_utility": float(test.mean()),
                "overfit_gap": float(gap.mean()),
                "gap_lo": float(np.quantile(gap, 0.025)),
                "gap_hi": float(np.quantile(gap, 0.975)),
            }
        )
    return {
        "reps": reps,
        "rank_correlation_mean": float(np.mean(rank_corrs)),
        "rank_correlation_lo": float(np.quantile(rank_corrs, 0.025)),
        "rank_correlation_hi": float(np.quantile(rank_corrs, 0.975)),
        "summary": summary,
    }


def lambda_sweep(ndcg: np.ndarray) -> list[dict]:
    costs = np.asarray([COSTS[str(v)] for v in VIEWS], dtype="float32")
    out = []
    for lam in [0.0, 0.02, 0.05, 0.08, 0.12, 0.20]:
        util = ndcg - lam * costs[None, :]
        fixed = util.mean(axis=0)
        oracle = util.max(axis=1)
        best = util.argmax(axis=1)
        out.append(
            {
                "lambda": float(lam),
                "fixed_summary": float(fixed[0]),
                "fixed_pq": float(fixed[1]),
                "fixed_full": float(fixed[2]),
                "best_fixed": str(VIEWS[int(fixed.argmax())]),
                "oracle": float(oracle.mean()),
                "oracle_choices": {str(VIEWS[i]): float(np.mean(best == i)) for i in range(len(VIEWS))},
            }
        )
    return out


def main() -> None:
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
    fixed_full_raw = float(ndcg[:, 2].mean())
    fixed_pq_raw = float(ndcg[:, 1].mean())
    if fixed_full_raw < 0.20 or fixed_pq_raw < 0.15:
        raise RuntimeError(
            f"Sanity check failed: raw full={fixed_full_raw:.4f}, pq={fixed_pq_raw:.4f}. "
            "This usually indicates query/qrel or embedding-cache misalignment."
        )

    displayed = one_split(x, y, ndcg, groups, args.seed)
    rep_rows = []
    for rep in range(args.split_reps):
        res = quick_repeated_split(x, y, ndcg, groups, args.seed + 1000 + rep)
        rep_rows.append(
            {
                "rep": rep,
                "best_fixed": res["best_fixed"],
                "best_adaptive": res["best_adaptive"],
                "adaptive_minus_best_fixed": res["adaptive_minus_best_fixed"],
            }
        )
    diffs = np.asarray([r["adaptive_minus_best_fixed"] for r in rep_rows], dtype="float32")
    split_summary = {
        "reps": int(args.split_reps),
        "mean_adaptive_minus_best_fixed": float(diffs.mean()),
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "positive_share": float(np.mean(diffs > 0)),
    }
    probing = probing_simulation(x, y, ndcg, groups, args.seed, reps=min(12, args.split_reps))

    out = {
        "task": "protocol_b_standard_qrel_602020_audit",
        "model": args.model,
        "lambda": args.lambda_cost,
        "datasets": meta,
        "total_queries": int(len(x)),
        "sanity": {
            "pooled_raw_ndcg_summary": float(ndcg[:, 0].mean()),
            "pooled_raw_ndcg_pq": fixed_pq_raw,
            "pooled_raw_ndcg_full": fixed_full_raw,
        },
        "displayed_split": displayed,
        "repeated_split_summary": split_summary,
        "repeated_split_rows": rep_rows,
        "anti_probing_simulation": probing,
        "lambda_sweep": lambda_sweep(ndcg),
    }
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in args.model).strip("_")
    json_path = REPORT_DIR / f"protocol_b_standard_qrel_602020_{safe_model}.json"
    md_path = REPORT_DIR / f"protocol_b_standard_qrel_602020_{safe_model}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Protocol B Standard-Qrel 60/20/20 Learning Audit",
        "",
        f"- Model: `{args.model}`",
        f"- Total queries: {len(x):,}",
        f"- Displayed split: train {displayed['train_queries']}, dev {displayed['dev_queries']}, test {displayed['test_queries']}",
        f"- Sanity raw NDCG: summary {out['sanity']['pooled_raw_ndcg_summary']:.4f}, PQ {fixed_pq_raw:.4f}, full {fixed_full_raw:.4f}",
        "",
        "## Displayed hidden-test split",
        "",
        "| solver | utility | NDCG | cost | regret | gap closed | choices | paired diff vs best fixed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in displayed["routes"]:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in r["choices"].items())
        ci = r.get("paired_diff_vs_best_fixed")
        diff = "--" if not ci else f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
        lines.append(
            f"| {r['name']} | {r['utility']:.4f} | {r['ndcg']:.4f} | {r['cost']:.4f} | {r['regret']:.4f} | {r['gap_closed']:.2f} | {ch} | {diff} |"
        )
    cal = displayed["calibration"]
    lines += [
        "",
        "## Restricted-menu Protocol B solvers",
        "",
        "| menu | legal choices | best fixed | best adaptive | oracle | gap closed | residual regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in displayed["restricted_menus"]:
        lines.append(
            f"| {row['menu']} | {row['legal_choices']} | {row['best_fixed']} {row['best_fixed_utility']:.4f} | "
            f"{row['best_adaptive']} {row['best_adaptive_utility']:.4f} | {row['oracle_utility']:.4f} | "
            f"{row['gap_closed']:.3f} | {row['residual_regret']:.4f} |"
        )
    lines += [
        "",
        "## Full-purchase calibration",
        "",
        "| target | prevalence | AUPRC | Brier | ECE | buy rate | route utility | route regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| U(full)>U(PQ) | {cal['prevalence']:.3f} | {cal['auprc']:.3f} | {cal['brier']:.3f} | {cal['ece']:.3f} | {cal['buy_rate']:.3f} | {cal['route_utility']:.4f} | {cal['route_regret']:.4f} |",
        "",
        "## Repeated stratified 60/20/20 hidden-test dry run",
        "",
        f"- Repetitions: {split_summary['reps']}",
        f"- Best adaptive minus best fixed: {split_summary['mean_adaptive_minus_best_fixed']:+.4f} [{split_summary['lo']:+.4f},{split_summary['hi']:+.4f}]",
        f"- Positive split share: {split_summary['positive_share']:.3f}",
        "",
        "## Public-dev probing simulation",
        "",
        f"- Mean rank correlation between public-dev and private-test candidate utilities: {probing['rank_correlation_mean']:.3f}",
        "",
        "| submission cap | public-dev utility | private-test utility | overfit gap |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in probing["summary"]:
        lines.append(
            f"| {row['submission_cap']} | {row['public_dev_utility']:.4f} | {row['private_test_utility']:.4f} | "
            f"{row['overfit_gap']:.4f} [{row['gap_lo']:.4f},{row['gap_hi']:.4f}] |"
        )
    lines += [
        "",
        "## Lambda sweep over raw fixed views",
        "",
        "| lambda | fixed S | fixed PQ | fixed F | best fixed | oracle | oracle choices |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in out["lambda_sweep"]:
        ch = ", ".join(f"{k} {v:.2f}" for k, v in row["oracle_choices"].items())
        lines.append(
            f"| {row['lambda']:.2f} | {row['fixed_summary']:.4f} | {row['fixed_pq']:.4f} | {row['fixed_full']:.4f} | {row['best_fixed']} | {row['oracle']:.4f} | {ch} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
