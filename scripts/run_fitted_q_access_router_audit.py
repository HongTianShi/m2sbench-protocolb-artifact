"""Offline fitted-Q / contextual-bandit access router audit.

The access decision in M2S-Bench is a one-step decision problem: observe
method-visible summary/context features, choose one view, and receive
cost-adjusted utility. A fitted-Q router is therefore the natural lightweight
RL baseline. This script evaluates it without evaluator feedback or hidden
targets at test time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "summaries"
REPORTS = ROOT / "reports"
OUT = SUMMARIES / "fitted_q_access_router"


def _mean(x: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=float)))


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def _markdown_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _case_features(case_ids: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cid in case_ids.astype(str):
        m = re.match(r"budget_(\d+)_([^:]+):([^:]+):(\d+)", cid)
        if m:
            budget_id = int(m.group(1))
            budget_name = m.group(2)
            family = m.group(3)
            seed = int(m.group(4))
        else:
            budget_id, budget_name, family, seed = -1, "unknown", "unknown", -1
        rows.append(
            {
                "budget_id": budget_id,
                "budget_name": budget_name,
                "family_from_id": family,
                "seed_mod": seed,
                "case_len": len(cid),
            }
        )
    return pd.DataFrame(rows)


def _fit_action_models(
    X: pd.DataFrame,
    rewards: pd.DataFrame,
    groups: pd.Series | np.ndarray | None,
    *,
    n_splits: int,
    model_kind: str,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    actions = list(rewards.columns)
    pred = np.zeros((len(X), len(actions)), dtype=float)
    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, groups=groups)

    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    fold_rows: list[dict[str, float | int]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        for j, action in enumerate(actions):
            if model_kind == "rf":
                model = RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=random_state + 31 * fold + j,
                    n_jobs=-1,
                )
            else:
                model = ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_state + 31 * fold + j,
                    n_jobs=-1,
                )
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X.iloc[train_idx], rewards.iloc[train_idx][action])
            pred[test_idx, j] = pipe.predict(X.iloc[test_idx])
        chosen = pred[test_idx].argmax(axis=1)
        actual = rewards.iloc[test_idx].to_numpy()
        chosen_reward = actual[np.arange(len(test_idx)), chosen]
        oracle_reward = actual.max(axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(test_idx)),
                "utility": _mean(chosen_reward),
                "oracle_utility": _mean(oracle_reward),
                "regret": _mean(oracle_reward - chosen_reward),
                "best_view_acc": _mean(chosen == actual.argmax(axis=1)),
            }
        )
    return pred, pd.DataFrame(fold_rows)


def _evaluate_policy(
    name: str,
    rewards: pd.DataFrame,
    actions: list[str],
    chosen_idx: np.ndarray,
    costs: dict[str, float],
    *,
    ndcg: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    actual = rewards.to_numpy()
    chosen_reward = actual[np.arange(len(rewards)), chosen_idx]
    oracle = actual.max(axis=1)
    chosen_actions = np.asarray(actions, dtype=object)[chosen_idx]
    row: dict[str, float | str] = {
        "policy": name,
        "cases": int(len(rewards)),
        "utility": _mean(chosen_reward),
        "oracle_utility": _mean(oracle),
        "regret": _mean(oracle - chosen_reward),
        "best_view_acc": _mean(chosen_idx == actual.argmax(axis=1)),
        "avg_cost": _mean([costs.get(str(a), 0.0) for a in chosen_actions]),
        "non_richest_share": _mean(chosen_actions != actions[-1]),
    }
    if ndcg is not None:
        ndcg_vals = ndcg.to_numpy()[np.arange(len(ndcg)), chosen_idx]
        row["ndcg"] = _mean(ndcg_vals)
    return row


def run_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SUMMARIES / "cikm_batch1_information_access_20260505" / "access_view_ranking_cases_scored.csv"
    df = pd.read_csv(path)
    df = df[df["split"].isin(["synthetic_heldout", "semi_real"])].copy()
    actions = ["moments_only", "occupancy_grid", "raster", "point_cloud"]
    df = df[df["access_level"].isin(actions)]
    rewards = df.pivot_table(index=["split", "case_id"], columns="access_level", values="utility", aggfunc="first")
    rewards = rewards[actions].dropna()
    base = rewards.reset_index()[["split", "case_id"]]
    X = pd.concat([base[["split"]].reset_index(drop=True), _case_features(base["case_id"])], axis=1)
    groups = X["budget_id"].astype(str) + ":" + X["split"].astype(str)
    pred, folds = _fit_action_models(X, rewards.reset_index(drop=True), groups, n_splits=4, model_kind="extra", random_state=17)
    chosen = pred.argmax(axis=1)

    costs = {"moments_only": 0.0, "occupancy_grid": 0.28, "raster": 0.42, "point_cloud": 0.58}
    rows = []
    for policy, action in [
        ("fixed_summary", "moments_only"),
        ("fixed_occupancy", "occupancy_grid"),
        ("fixed_raster", "raster"),
        ("fixed_point", "point_cloud"),
    ]:
        rows.append(_evaluate_policy(policy, rewards, actions, np.full(len(rewards), actions.index(action)), costs))
    rows.append(_evaluate_policy("fitted_q_router", rewards, actions, chosen, costs))
    rows.append(_evaluate_policy("oracle_eval_only", rewards, actions, rewards.to_numpy().argmax(axis=1), costs))
    out = pd.DataFrame(rows)
    out.insert(0, "scope", "synthetic_access")
    folds.insert(0, "scope", "synthetic_access")
    return out, folds


def _cluster_log_bandit(
    df: pd.DataFrame,
    rewards: pd.DataFrame,
    ndcg: pd.DataFrame,
    actions: list[str],
    costs: dict[str, float],
) -> dict[str, float | str]:
    """Query-held-out cluster-log policy.

    This is a B1/context-budget baseline: train folds provide historical
    utility records for the same coarse cells, and test queries are unseen.
    It simulates the logs or caches available in deployed retrieval systems.
    """
    keys = (df["dataset"].astype(str) + ":" + df["cluster_id"].astype(str)).to_numpy()
    chosen = np.zeros(len(df), dtype=int)
    splits = KFold(n_splits=5, shuffle=True, random_state=44).split(df)
    for train_idx, test_idx in splits:
        train = pd.DataFrame(rewards.to_numpy()[train_idx], columns=actions)
        train["key"] = keys[train_idx]
        means = train.groupby("key")[actions].mean()
        global_best = int(rewards.to_numpy()[train_idx].mean(axis=0).argmax())
        for idx in test_idx:
            if keys[idx] in means.index:
                chosen[idx] = int(means.loc[keys[idx]].to_numpy().argmax())
            else:
                chosen[idx] = global_best
    return _evaluate_policy("cluster_log_bandit_b1", rewards, actions, chosen, costs, ndcg=ndcg)


def run_ivf() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SUMMARIES / "ivf_pq_rerank_large_adapter" / "large_ivf_pq_rerank_per_query_sample.csv"
    df = pd.read_csv(path)
    actions = ["summary", "pq", "full"]
    rewards = df[["summary_utility", "pq_utility", "full_utility"]].copy()
    rewards.columns = actions
    ndcg = df[["summary_ndcg", "pq_ndcg", "full_ndcg"]].copy()
    ndcg.columns = actions
    X = df[
        [
            "dataset",
            "candidate_count",
            "cluster_n",
            "cluster_radius",
            "cluster_radius_std",
            "query_centroid_distance",
        ]
    ].copy()
    groups = df["dataset"].astype(str) + ":" + df["cluster_id"].astype(str)
    n_splits = min(5, groups.nunique())
    pred, folds = _fit_action_models(X, rewards, groups, n_splits=n_splits, model_kind="rf", random_state=23)
    chosen = pred.argmax(axis=1)
    costs = {"summary": 0.0, "pq": 0.2, "full": 0.58}

    rows = []
    for policy, action in [("fixed_summary", "summary"), ("fixed_pq", "pq"), ("fixed_full", "full")]:
        rows.append(
            _evaluate_policy(policy, rewards, actions, np.full(len(rewards), actions.index(action)), costs, ndcg=ndcg)
        )
    rows.append(_evaluate_policy("fitted_q_router_b0", rewards, actions, chosen, costs, ndcg=ndcg))
    rows.append(_cluster_log_bandit(df, rewards, ndcg, actions, costs))
    rows.append(_evaluate_policy("oracle_eval_only", rewards, actions, rewards.to_numpy().argmax(axis=1), costs, ndcg=ndcg))
    out = pd.DataFrame(rows)
    out.insert(0, "scope", "ivf_pq_semantic")
    folds.insert(0, "scope", "ivf_pq_semantic")
    return out, folds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    synth, synth_folds = run_synthetic()
    ivf, ivf_folds = run_ivf()
    summary = pd.concat([synth, ivf], ignore_index=True)
    folds = pd.concat([synth_folds, ivf_folds], ignore_index=True)

    summary.to_csv(OUT / "fitted_q_access_router_summary.csv", index=False)
    folds.to_csv(OUT / "fitted_q_access_router_folds.csv", index=False)

    report = {
        "synthetic_fitted_q_utility": float(summary[(summary.scope == "synthetic_access") & (summary.policy == "fitted_q_router")]["utility"].iloc[0]),
        "synthetic_best_fixed_utility": float(summary[(summary.scope == "synthetic_access") & (summary.policy.str.startswith("fixed_"))]["utility"].max()),
        "ivf_fitted_q_b0_utility": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy == "fitted_q_router_b0")]["utility"].iloc[0]),
        "ivf_cluster_log_b1_utility": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy == "cluster_log_bandit_b1")]["utility"].iloc[0]),
        "ivf_best_fixed_utility": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy.str.startswith("fixed_"))]["utility"].max()),
        "ivf_fitted_q_b0_ndcg": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy == "fitted_q_router_b0")]["ndcg"].iloc[0]),
        "ivf_cluster_log_b1_ndcg": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy == "cluster_log_bandit_b1")]["ndcg"].iloc[0]),
        "ivf_best_fixed_ndcg": float(summary[(summary.scope == "ivf_pq_semantic") & (summary.policy.str.startswith("fixed_"))]["ndcg"].max()),
        "interpretation": "one-step offline RL/contextual-bandit baseline; B0 uses method-visible summary/proxy features, while B1 cluster-log bandit simulates historical query/cache context as an extended summary budget; action chooses a view; reward is cost-adjusted utility",
    }
    (REPORTS / "fitted_q_access_router_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Fitted-Q Access Router Audit", ""]
    md.append("One-step offline RL/contextual-bandit router. State = method-visible summary/proxy features; action = access view; reward = cost-adjusted utility.")
    md.append("")
    md.append("## Summary")
    md.append(_markdown_table(summary.round(3)))
    md.append("")
    md.append("## Cross-validation folds")
    md.append(_markdown_table(folds.round(3)))
    (REPORTS / "fitted_q_access_router_audit.md").write_text("\n".join(md), encoding="utf-8")

    print(summary.round(4).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
