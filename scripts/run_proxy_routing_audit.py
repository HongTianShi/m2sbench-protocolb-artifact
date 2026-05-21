from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SUMMARIES = ROOT / "summaries"


SUMMARY_VISIBLE_FEATURES = [
    "risk_score",
    "event_score",
    "n_observations",
    "session_coverage",
    "link_assets_available",
    "net_return",
    "abs_net_return",
    "mean_return",
    "std_return",
    "realized_vol",
    "autocorr_1",
    "vol_cluster_score",
    "range_mean",
    "range_std",
    "range_q95",
    "volume_mean",
    "volume_std",
    "volume_burst",
    "up_share",
    "jump_intensity",
    "reversal_score",
    "monotonicity",
]


POLICIES = {
    "proxy_tree_depth3": DecisionTreeClassifier(max_depth=3, random_state=7),
    "proxy_logistic_l2": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=7),
    ),
    "proxy_rf_small": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=7, n_jobs=1
    ),
    "proxy_gbdt_small": GradientBoostingClassifier(
        n_estimators=80, learning_rate=0.05, max_depth=2, random_state=7
    ),
}


UTILITY_POLICIES = {
    "proxy_utility_rf_regressor": make_pipeline(
        SimpleImputer(strategy="median"),
        MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=6,
                random_state=11,
                n_jobs=1,
            )
        ),
    ),
    "proxy_utility_gbdt_regressor": make_pipeline(
        SimpleImputer(strategy="median"),
        MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=120, learning_rate=0.04, max_depth=2, random_state=11
            )
        ),
    ),
}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _fold_ci(mean: float, sd: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(mean) or not np.isfinite(sd) or n <= 1:
        return float("nan"), float("nan")
    # Two-sided t critical for the 5 grouped folds used here.
    tcrit = 2.776 if n == 5 and abs(alpha - 0.05) < 1e-12 else 1.96
    half = tcrit * sd / math.sqrt(n)
    return float(mean - half), float(mean + half)


def _load_financial_oracle() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(SUMMARIES / "financial_access_level_cases.csv")
    utility = cases.pivot_table(
        index="window_id", columns="access_level", values="utility", aggfunc="first"
    )
    utility = utility.reset_index()
    view_cols = [c for c in utility.columns if c != "window_id"]
    utility["oracle_best_access"] = utility[view_cols].idxmax(axis=1)
    utility["oracle_best_utility"] = utility[view_cols].max(axis=1)

    features = pd.read_parquet(SUMMARIES / "financial_summary_features.parquet")
    keep = ["window_id", "date_ny"] + [c for c in SUMMARY_VISIBLE_FEATURES if c in features.columns]
    features = features[keep].drop_duplicates("window_id")
    data = features.merge(utility, on="window_id", how="inner")
    return data, utility


def _evaluate_fixed_policy(data: pd.DataFrame, policy: str, folds: GroupKFold) -> dict[str, float | str]:
    accs: list[float] = []
    utils: list[float] = []
    regrets: list[float] = []
    n_tests: list[int] = []
    groups = data["date_ny"].astype(str).to_numpy()
    for _, test_idx in folds.split(data, data["oracle_best_access"], groups):
        test = data.iloc[test_idx]
        accs.append(float((test["oracle_best_access"] == policy).mean()))
        utils.append(float(test[policy].mean()))
        regrets.append(float((test["oracle_best_utility"] - test[policy]).mean()))
        n_tests.append(int(len(test)))
    acc_m, acc_sd = _mean_sd(accs)
    util_m, util_sd = _mean_sd(utils)
    reg_m, reg_sd = _mean_sd(regrets)
    return {
        "policy": f"fixed_{policy}",
        "folds": 5,
        "n_test_mean": float(np.mean(n_tests)),
        "top1_access_acc_mean": acc_m,
        "top1_access_acc_sd": acc_sd,
        "utility_mean": util_m,
        "utility_sd": util_sd,
        "regret_mean": reg_m,
        "regret_sd": reg_sd,
    }


def _evaluate_proxy_policy(data: pd.DataFrame, name: str, model, folds: GroupKFold) -> dict[str, float | str]:
    feature_cols = [c for c in SUMMARY_VISIBLE_FEATURES if c in data.columns]
    x = data[feature_cols]
    y = data["oracle_best_access"]
    groups = data["date_ny"].astype(str).to_numpy()

    accs: list[float] = []
    utils: list[float] = []
    regrets: list[float] = []
    n_tests: list[int] = []
    for train_idx, test_idx in folds.split(x, y, groups):
        train_x, test_x = x.iloc[train_idx], x.iloc[test_idx]
        train_y = y.iloc[train_idx]
        test = data.iloc[test_idx]

        estimator = clone(model)
        if name in {"proxy_tree_depth3", "proxy_rf_small", "proxy_gbdt_small"}:
            estimator = make_pipeline(SimpleImputer(strategy="median"), clone(model))
        estimator.fit(train_x, train_y)
        pred = np.asarray(estimator.predict(test_x))
        test_reset = test.reset_index(drop=True)
        view_cols = sorted(y.unique().tolist())
        utility_matrix = test_reset[view_cols].to_numpy(dtype=float)
        col_lookup = {col: idx for idx, col in enumerate(view_cols)}
        pred_cols = np.asarray([col_lookup[p] for p in pred], dtype=int)
        pred_utility = utility_matrix[np.arange(len(test_reset)), pred_cols]
        accs.append(float((test["oracle_best_access"].to_numpy() == pred).mean()))
        utils.append(float(np.mean(pred_utility)))
        regrets.append(float(np.mean(test["oracle_best_utility"].to_numpy() - pred_utility)))
        n_tests.append(int(len(test)))

    acc_m, acc_sd = _mean_sd(accs)
    util_m, util_sd = _mean_sd(utils)
    reg_m, reg_sd = _mean_sd(regrets)
    return {
        "policy": name,
        "folds": 5,
        "n_test_mean": float(np.mean(n_tests)),
        "top1_access_acc_mean": acc_m,
        "top1_access_acc_sd": acc_sd,
        "utility_mean": util_m,
        "utility_sd": util_sd,
        "regret_mean": reg_m,
        "regret_sd": reg_sd,
    }


def _evaluate_proxy_utility_policy(
    data: pd.DataFrame, name: str, model, folds: GroupKFold
) -> dict[str, float | str]:
    feature_cols = [c for c in SUMMARY_VISIBLE_FEATURES if c in data.columns]
    view_cols = sorted(data["oracle_best_access"].unique().tolist())
    x = data[feature_cols]
    y = data[view_cols]
    groups = data["date_ny"].astype(str).to_numpy()

    accs: list[float] = []
    utils: list[float] = []
    regrets: list[float] = []
    n_tests: list[int] = []
    for train_idx, test_idx in folds.split(x, data["oracle_best_access"], groups):
        estimator = clone(model)
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred_util = np.asarray(estimator.predict(x.iloc[test_idx]), dtype=float)
        pred_cols = pred_util.argmax(axis=1)
        real_util = y.iloc[test_idx].to_numpy(dtype=float)
        chosen = real_util[np.arange(len(test_idx)), pred_cols]
        pred_access = np.asarray(view_cols)[pred_cols]
        test = data.iloc[test_idx]
        accs.append(float((test["oracle_best_access"].to_numpy() == pred_access).mean()))
        utils.append(float(np.mean(chosen)))
        regrets.append(float(np.mean(test["oracle_best_utility"].to_numpy() - chosen)))
        n_tests.append(int(len(test)))

    acc_m, acc_sd = _mean_sd(accs)
    util_m, util_sd = _mean_sd(utils)
    reg_m, reg_sd = _mean_sd(regrets)
    return {
        "policy": name,
        "folds": 5,
        "n_test_mean": float(np.mean(n_tests)),
        "top1_access_acc_mean": acc_m,
        "top1_access_acc_sd": acc_sd,
        "utility_mean": util_m,
        "utility_sd": util_sd,
        "regret_mean": reg_m,
        "regret_sd": reg_sd,
    }


def run_bloomberg_proxy_audit() -> pd.DataFrame:
    data, _ = _load_financial_oracle()
    folds = GroupKFold(n_splits=5)
    rows: list[dict[str, float | str]] = []

    rows.append(
        {
            "policy": "oracle_best_access",
            "folds": "",
            "n_test_mean": float(len(data)),
            "top1_access_acc_mean": 1.0,
            "top1_access_acc_sd": 0.0,
            "utility_mean": float(data["oracle_best_utility"].mean()),
            "utility_sd": "",
            "regret_mean": 0.0,
            "regret_sd": 0.0,
        }
    )
    for policy in ["summary_only", "coarse_path", "microstructure_proxy", "cross_asset_path"]:
        rows.append(_evaluate_fixed_policy(data, policy, folds))
    for name, model in POLICIES.items():
        rows.append(_evaluate_proxy_policy(data, name, model, folds))
    for name, model in UTILITY_POLICIES.items():
        rows.append(_evaluate_proxy_utility_policy(data, name, model, folds))

    policy_summary = pd.read_csv(SUMMARIES / "financial_access_policy.csv")
    rule = policy_summary[
        (policy_summary["regime"] == "overall")
        & (policy_summary["policy"] == "rule_based_escalation")
    ].iloc[0]
    rows.append(
        {
            "policy": "existing_rule_based_escalation",
            "folds": "",
            "n_test_mean": float(rule["n_windows"]),
            "top1_access_acc_mean": "",
            "top1_access_acc_sd": "",
            "utility_mean": float(rule["utility"]),
            "utility_sd": "",
            "regret_mean": float(rule["regret"]),
            "regret_sd": "",
        }
    )

    out = pd.DataFrame(rows).sort_values("utility_mean", ascending=False)
    for metric in ["utility", "regret"]:
        lows: list[float] = []
        highs: list[float] = []
        for row in out.to_dict(orient="records"):
            try:
                folds_n = int(row["folds"])
                mean = float(row[f"{metric}_mean"])
                sd = float(row[f"{metric}_sd"])
            except (TypeError, ValueError):
                folds_n = 0
                mean = float("nan")
                sd = float("nan")
            low, high = _fold_ci(mean, sd, folds_n)
            lows.append(low)
            highs.append(high)
        out[f"{metric}_ci95_low"] = lows
        out[f"{metric}_ci95_high"] = highs
    out.to_csv(REPORTS / "proxy_routing_field_pilot.csv", index=False)
    _write_proxy_delta_table(out)
    return out


def _write_proxy_delta_table(field: pd.DataFrame) -> None:
    rows: list[dict[str, float | str]] = []
    base_names = ["fixed_cross_asset_path", "fixed_coarse_path", "fixed_summary_only"]
    proxy = field[field["policy"] == "proxy_utility_rf_regressor"].iloc[0]
    proxy_mean = float(proxy["utility_mean"])
    proxy_low = float(proxy["utility_ci95_low"])
    proxy_high = float(proxy["utility_ci95_high"])
    for base_name in base_names:
        base = field[field["policy"] == base_name].iloc[0]
        base_mean = float(base["utility_mean"])
        base_low = float(base["utility_ci95_low"])
        base_high = float(base["utility_ci95_high"])
        rows.append(
            {
                "comparison": f"proxy_utility_rf_regressor minus {base_name}",
                "proxy_utility_mean": proxy_mean,
                "proxy_utility_ci95_low": proxy_low,
                "proxy_utility_ci95_high": proxy_high,
                "baseline_utility_mean": base_mean,
                "baseline_utility_ci95_low": base_low,
                "baseline_utility_ci95_high": base_high,
                "mean_delta": proxy_mean - base_mean,
                "ci_overlap": bool(max(proxy_low, base_low) <= min(proxy_high, base_high)),
                "reading": "Date-held grouped-fold CIs are fold-level, not row-level bootstrap intervals.",
            }
        )
    rule = field[field["policy"] == "existing_rule_based_escalation"].iloc[0]
    rows.append(
        {
            "comparison": "proxy_utility_rf_regressor minus existing_rule_based_escalation",
            "proxy_utility_mean": proxy_mean,
            "proxy_utility_ci95_low": proxy_low,
            "proxy_utility_ci95_high": proxy_high,
            "baseline_utility_mean": float(rule["utility_mean"]),
            "baseline_utility_ci95_low": float("nan"),
            "baseline_utility_ci95_high": float("nan"),
            "mean_delta": proxy_mean - float(rule["utility_mean"]),
            "ci_overlap": "",
            "reading": "Rule baseline is an aggregate record without date-fold variance.",
        }
    )
    pd.DataFrame(rows).to_csv(REPORTS / "proxy_routing_field_pilot_deltas.csv", index=False)


def build_compact_audit(field: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(REPORTS / "recovery_only_access_predictability_split_summary.csv")
    direct = pd.read_csv(REPORTS / "recovery_only_access_predictability.csv")
    rows: list[dict[str, str]] = []

    random_auc = split.loc[
        split["split"] == "random_cell_stratified_5fold", "auc_mean"
    ].iloc[0]
    budget_auc = split.loc[
        split["split"] == "budget_group_heldout_5fold", "auc_mean"
    ].iloc[0]
    family_auc = split.loc[
        split["split"] == "family_group_heldout_5fold", "auc_mean"
    ].iloc[0]
    best_direct_auc = direct.loc[
        direct["task"] == "positive_access_gain", "metric_auc_mean"
    ].max()
    best_rf_r2 = direct.loc[direct["task"] == "access_gain_magnitude", "r2_mean"].max()

    rows.extend(
        [
            {
                "domain": "Synthetic public cells",
                "proxy_or_policy": "Recovery-only terrain proxies",
                "evaluation": "Random-cell 5-fold",
                "result": f"AUROC {random_auc:.3f}; best direct audit AUROC {best_direct_auc:.3f}",
                "reading": "Summary/recovery-visible proxies forecast access gain within the controlled contract.",
            },
            {
                "domain": "Synthetic public cells",
                "proxy_or_policy": "Recovery-only terrain proxies",
                "evaluation": "Budget-held-out 5-fold",
                "result": f"AUROC {budget_auc:.3f}",
                "reading": "Proxy signal survives summary-budget shifts.",
            },
            {
                "domain": "Synthetic public cells",
                "proxy_or_policy": "Recovery-only terrain proxies",
                "evaluation": "Family-held-out 5-fold",
                "result": f"AUROC {family_auc:.3f}; gain-magnitude R2 {best_rf_r2:.3f}",
                "reading": "OOD family transfer is weaker, so labels remain contract-local rather than universal classes.",
            },
        ]
    )

    for policy in [
        "fixed_summary_only",
        "fixed_coarse_path",
        "existing_rule_based_escalation",
        "proxy_utility_rf_regressor",
        "fixed_cross_asset_path",
    ]:
        row = field[field["policy"] == policy].iloc[0]
        rows.append(
            {
                "domain": "Bloomberg minute-bar adapter",
                "proxy_or_policy": policy.replace("_", " "),
                "evaluation": "Date-held grouped folds" if "fixed" in policy or "proxy" in policy else "Aggregate rule record",
                "result": f"utility {float(row['utility_mean']):.3f}; regret {float(row['regret_mean']):.3f}",
                "reading": "Summary-visible routing is actionable but not a solved universal router.",
            }
        )

    compact = pd.DataFrame(rows)
    compact.to_csv(REPORTS / "proxy_diagnostic_routing_audit.csv", index=False)
    with (REPORTS / "proxy_diagnostic_routing_audit.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("| Domain | Proxy/policy | Evaluation | Result | Reading |\n")
        f.write("|---|---|---|---|---|\n")
        for row in compact.to_dict(orient="records"):
            f.write(
                f"| {row['domain']} | {row['proxy_or_policy']} | {row['evaluation']} | {row['result']} | {row['reading']} |\n"
            )
    return compact


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    field = run_bloomberg_proxy_audit()
    compact = build_compact_audit(field)
    print(f"Wrote {REPORTS / 'proxy_routing_field_pilot.csv'}")
    print(f"Wrote {REPORTS / 'proxy_diagnostic_routing_audit.csv'}")
    print(f"Wrote {REPORTS / 'proxy_diagnostic_routing_audit.md'}")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
