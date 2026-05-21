"""Audit diagnostic-regime stability as a continuous terrain.

This script is intentionally lightweight: it reuses the public Protocol B
diagnostic tables and target-blind witness-neighbor records already shipped
with the artifact.  It does not read hidden targets or regenerate cells.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "summaries"
OUT_DIR = SUMMARY_DIR / "diagnostic_terrain_audit"
FIG_DIR = ROOT / "figures"


PHASE_MAP = SUMMARY_DIR / "phase_map.csv"
PAIR_FILE = (
    SUMMARY_DIR
    / "cikm_batch1_information_access_20260505"
    / "matched_summary_witness_pairs_topk.csv"
)
WITNESS_FILE = (
    SUMMARY_DIR
    / "cikm_batch3_solver_adapters_20260505"
    / "witness_driven_heuristic_search_results.csv"
)


def _pct_rank(s: pd.Series, reverse: bool = False) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    if reverse:
        values = -values
    return values.rank(method="average", pct=True).fillna(0.5)


def _minmax(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce").astype(float)
    lo = values.min()
    hi = values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def _zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        std = x.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            out[col] = 0.0
        else:
            out[col] = (x - x.mean()) / std
    return out.fillna(0.0)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 4 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).correlation)


def _auc(score: pd.Series, label: pd.Series) -> float:
    y = pd.to_numeric(label, errors="coerce").fillna(0).astype(int)
    x = pd.to_numeric(score, errors="coerce").fillna(score.median())
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def _stable_random_label(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return digest[0] % 2


def build_terrain(phase: pd.DataFrame) -> pd.DataFrame:
    terrain = phase.copy()
    terrain["cell_key"] = (
        terrain["budget"].astype(str)
        + "|"
        + terrain["family"].astype(str)
        + "|"
        + terrain["seed"].astype(str)
    )

    # Axis 1: feasible-set divergence, using only target-blind candidate-set
    # geometry/exploration summaries.
    divergence_parts = [
        _pct_rank(terrain["hypothesis_diversity"]),
        _pct_rank(terrain["feasible_set_exploration"]),
        _pct_rank(terrain["structural_mode_coverage"]),
        _pct_rank(1.0 - pd.to_numeric(terrain["posterior_concentration_proxy"])),
    ]
    terrain["feasible_set_divergence"] = _minmax(pd.concat(divergence_parts, axis=1).mean(axis=1))

    # Axis 2: blind-search barrier (inverse reachability).  High values mean the
    # blind baseline is far from target structure or misses coverage/IoU.
    barrier_parts = [
        _pct_rank(terrain["blind_norm_cd"]),
        _pct_rank(1.0 - pd.to_numeric(terrain["blind_coverage"])),
        _pct_rank(1.0 - pd.to_numeric(terrain["blind_iou"])),
        _pct_rank(terrain["early_best_cd"]),
        _pct_rank(terrain["sampler_gain"]),
        _pct_rank(1.0 - pd.to_numeric(terrain["blind_success"])),
    ]
    terrain["blind_search_barrier"] = _minmax(pd.concat(barrier_parts, axis=1).mean(axis=1))
    terrain["search_reachability"] = 1.0 - terrain["blind_search_barrier"]

    # Axis 3: task sensitivity / marginal information value.  This keeps the
    # benchmark action contract explicit and combines action flips, transfer
    # drop, access value magnitude, escalation need, and view-choice mismatch.
    view_choice_gap = (
        pd.to_numeric(terrain["point_correct"], errors="coerce")
        - pd.to_numeric(terrain["moments_correct"], errors="coerce")
    ).abs()
    sensitivity_parts = [
        _pct_rank(terrain["downstream_fragility"]),
        _pct_rank(pd.to_numeric(terrain["transfer_drop"]).abs()),
        _pct_rank(pd.to_numeric(terrain["access_value_gain"]).abs()),
        _pct_rank(terrain["need_escalation"]),
        _pct_rank(view_choice_gap),
    ]
    terrain["task_sensitivity"] = _minmax(pd.concat(sensitivity_parts, axis=1).mean(axis=1))

    terrain["internal_divergence_proxy"] = terrain["feasible_set_divergence"]
    return terrain


def terrain_summary(terrain: pd.DataFrame) -> pd.DataFrame:
    axis_cols = ["feasible_set_divergence", "blind_search_barrier", "task_sensitivity"]
    X = terrain[axis_cols].to_numpy()
    labels = terrain["phase_name"].astype(str).to_numpy()
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")

    phase_summary = (
        terrain.groupby("phase_name", as_index=False)
        .agg(
            n=("cell_key", "size"),
            feasible_set_divergence_mean=("feasible_set_divergence", "mean"),
            blind_search_barrier_mean=("blind_search_barrier", "mean"),
            task_sensitivity_mean=("task_sensitivity", "mean"),
            access_value_gain_mean=("access_value_gain", "mean"),
            downstream_fragility_mean=("downstream_fragility", "mean"),
        )
        .sort_values("task_sensitivity_mean", ascending=False)
    )
    phase_summary.to_csv(OUT_DIR / "continuous_terrain_by_diagnostic_label.csv", index=False)

    # A compact grid view: where does positive cost-adjusted access value
    # concentrate when using the continuous terrain rather than hard labels?
    grid = terrain.copy()
    grid["divergence_tertile"] = pd.qcut(
        grid["feasible_set_divergence"], q=3, labels=["low", "mid", "high"], duplicates="drop"
    )
    grid["task_tertile"] = pd.qcut(
        grid["task_sensitivity"], q=3, labels=["low", "mid", "high"], duplicates="drop"
    )
    terrain_grid = (
        grid.groupby(["divergence_tertile", "task_tertile"], observed=True)
        .agg(
            n=("cell_key", "size"),
            mean_access_gain=("access_value_gain", "mean"),
            positive_access_share=("access_value_gain", lambda x: float((x > 0).mean())),
            mean_blind_barrier=("blind_search_barrier", "mean"),
        )
        .reset_index()
    )
    terrain_grid.to_csv(OUT_DIR / "continuous_terrain_access_grid.csv", index=False)

    rows = [
        {
            "audit": "continuous terrain axes",
            "metric": "label silhouette on 3 axes",
            "value": sil,
            "interpretation": "low-to-moderate values mean diagnostic names summarize dense regions, not separated ground-truth mechanisms",
        },
        {
            "audit": "continuous terrain axes",
            "metric": "rho(task sensitivity, |access gain|)",
            "value": _spearman(terrain["task_sensitivity"], terrain["access_value_gain"].abs()),
            "interpretation": "marginal access value aligns with the continuous sensitivity coordinate",
        },
        {
            "audit": "continuous terrain axes",
            "metric": "rho(feasible divergence, downstream fragility)",
            "value": _spearman(terrain["feasible_set_divergence"], terrain["downstream_fragility"]),
            "interpretation": "target-blind feasible-set spread is associated with downstream action sensitivity",
        },
        {
            "audit": "continuous terrain axes",
            "metric": "rho(blind barrier, blind Norm. CD)",
            "value": _spearman(terrain["blind_search_barrier"], terrain["blind_norm_cd"]),
            "interpretation": "the reachability axis recovers blind-search difficulty",
        },
    ]
    return pd.DataFrame(rows)


def add_topology_and_geometry(terrain: pd.DataFrame) -> pd.DataFrame:
    cells = terrain.copy()
    hole_threshold = cells["hole_proxy"].median()
    cells["topology_code"] = (
        (pd.to_numeric(cells["components"], errors="coerce").round().astype("Int64").astype(str))
        + "_"
        + (pd.to_numeric(cells["hole_proxy"], errors="coerce") > hole_threshold).astype(int).astype(str)
    )
    geom_cols = ["bbox_aspect_ratio", "curvature_std", "density_entropy", "symmetry_score"]
    z = _zscore_frame(cells, geom_cols)
    cells["geometry_score"] = z.mean(axis=1)
    cells["random_label"] = cells["cell_key"].map(_stable_random_label)
    return cells


def multitask_fragility(terrain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PAIR_FILE.exists():
        return pd.DataFrame(), pd.DataFrame()

    cells = add_topology_and_geometry(terrain)
    lookup_cols = [
        "cell_key",
        "topology_code",
        "geometry_score",
        "random_label",
        "feasible_set_divergence",
        "blind_search_barrier",
        "task_sensitivity",
        "internal_divergence_proxy",
        "phase_name",
    ]
    lookup = cells[lookup_cols].copy()

    pairs = pd.read_csv(PAIR_FILE)
    pairs = pairs[(pairs["representation"] == "Base summary") & (pairs["neighbor_rank"] <= 5)].copy()
    pairs["query_seed"] = pairs["query_seed"].astype(str)
    pairs["neighbor_seed"] = pairs["neighbor_seed"].astype(str)
    pairs["query_key"] = (
        pairs["query_budget"].astype(str)
        + "|"
        + pairs["query_family"].astype(str)
        + "|"
        + pairs["query_seed"].astype(str)
    )
    pairs["neighbor_key"] = (
        pairs["neighbor_budget"].astype(str)
        + "|"
        + pairs["neighbor_family"].astype(str)
        + "|"
        + pairs["neighbor_seed"].astype(str)
    )

    q = lookup.add_prefix("q_").rename(columns={"q_cell_key": "query_key"})
    n = lookup.add_prefix("n_").rename(columns={"n_cell_key": "neighbor_key"})
    pairs = pairs.merge(q, on="query_key", how="left").merge(n, on="neighbor_key", how="left")
    pairs = pairs.dropna(subset=["q_topology_code", "n_topology_code", "q_geometry_score", "n_geometry_score"])

    pairs["topology_disagree"] = (pairs["q_topology_code"] != pairs["n_topology_code"]).astype(float)
    pairs["geometry_distance"] = (pairs["q_geometry_score"] - pairs["n_geometry_score"]).abs()
    geom_cut = pairs["geometry_distance"].quantile(0.75)
    pairs["geometry_fragile"] = (pairs["geometry_distance"] >= geom_cut).astype(float)
    pairs["random_disagree"] = (pairs["q_random_label"] != pairs["n_random_label"]).astype(float)

    per_cell = (
        pairs.groupby("query_key", as_index=False)
        .agg(
            action_fragility=("action_disagree", "mean"),
            view_fragility=("view_disagree", "mean"),
            topology_fragility=("topology_disagree", "mean"),
            geometry_fragility=("geometry_fragile", "mean"),
            geometry_shift=("geometry_distance", "mean"),
            random_fragility=("random_disagree", "mean"),
            feasible_set_divergence=("q_feasible_set_divergence", "first"),
            blind_search_barrier=("q_blind_search_barrier", "first"),
            task_sensitivity=("q_task_sensitivity", "first"),
            internal_divergence_proxy=("q_internal_divergence_proxy", "first"),
            phase_name=("q_phase_name", "first"),
        )
        .rename(columns={"query_key": "cell_key"})
    )

    def high_flag(col: str) -> pd.Series:
        cut = per_cell[col].quantile(0.75)
        return ((per_cell[col] >= cut) & (per_cell[col] > 0)).astype(int)

    for col in ["action_fragility", "topology_fragility", "geometry_fragility", "random_fragility"]:
        per_cell[f"high_{col}"] = high_flag(col)

    def jaccard(a: str, b: str) -> float:
        aa = per_cell[f"high_{a}"].astype(bool)
        bb = per_cell[f"high_{b}"].astype(bool)
        union = (aa | bb).sum()
        if union == 0:
            return float("nan")
        return float((aa & bb).sum() / union)

    rows = [
        {
            "audit": "multi-task fragility",
            "metric": "mean action disagreement@5",
            "value": float(per_cell["action_fragility"].mean()),
            "interpretation": "benchmark downstream-action sensitivity among matched-summary neighbors",
        },
        {
            "audit": "multi-task fragility",
            "metric": "mean topology disagreement@5",
            "value": float(per_cell["topology_fragility"].mean()),
            "interpretation": "hole/component task sensitivity under the same summary-neighbor audit",
        },
        {
            "audit": "multi-task fragility",
            "metric": "mean geometry high-shift@5",
            "value": float(per_cell["geometry_fragility"].mean()),
            "interpretation": "bbox/curvature/density/symmetry regression sensitivity",
        },
        {
            "audit": "multi-task fragility",
            "metric": "mean random-label disagreement@5",
            "value": float(per_cell["random_fragility"].mean()),
            "interpretation": "negative-control task; it should not localize a meaningful diagnostic region",
        },
        {
            "audit": "multi-task fragility",
            "metric": "Jaccard high action vs topology",
            "value": jaccard("action_fragility", "topology_fragility"),
            "interpretation": "partial overlap means decision fragility is task-conditioned rather than universal",
        },
        {
            "audit": "multi-task fragility",
            "metric": "Jaccard high action vs geometry",
            "value": jaccard("action_fragility", "geometry_fragility"),
            "interpretation": "partial overlap means different downstream tasks expose different boundaries",
        },
        {
            "audit": "multi-task fragility",
            "metric": "Jaccard high action vs random",
            "value": jaccard("action_fragility", "random_fragility"),
            "interpretation": "random control overlap is reported as a sanity check",
        },
    ]

    per_cell.to_csv(OUT_DIR / "multitask_fragility_by_cell.csv", index=False)
    pairs.to_csv(OUT_DIR / "multitask_fragility_pairs_top5.csv", index=False)
    return per_cell, pd.DataFrame(rows)


def internal_divergence_audit(terrain: pd.DataFrame, per_cell: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not per_cell.empty:
        for target in ["action_fragility", "topology_fragility", "geometry_fragility", "random_fragility"]:
            rows.append(
                {
                    "audit": "internal divergence proxy",
                    "metric": f"rho(internal divergence, {target})",
                    "value": _spearman(per_cell["internal_divergence_proxy"], per_cell[target]),
                    "interpretation": "task-independent feasible-set spread versus multi-task fragility",
                }
            )
            rows.append(
                {
                    "audit": "internal divergence proxy",
                    "metric": f"AUC internal divergence predicts high {target}",
                    "value": _auc(per_cell["internal_divergence_proxy"], per_cell[f"high_{target}"]),
                    "interpretation": "0.5 is random; values above 0.5 indicate structural spread localizes fragile cells",
                }
            )

    if WITNESS_FILE.exists():
        witness = pd.read_csv(WITNESS_FILE)
        geo = witness[witness["policy"] == "geometric-spread candidate set"].copy()
        if not geo.empty:
            rows.extend(
                [
                    {
                        "audit": "sampled candidate-set divergence",
                        "metric": "rho(mean pairwise candidate CD, action coverage)",
                        "value": _spearman(geo["mean_candidate_norm_cd"], geo["action_disagreeing_coverage"]),
                        "interpretation": "direct sampled-candidate check on 50 Protocol B probe cells",
                    },
                    {
                        "audit": "sampled candidate-set divergence",
                        "metric": "rho(mean pairwise candidate CD, distinct actions)",
                        "value": _spearman(geo["mean_candidate_norm_cd"], geo["distinct_actions"]),
                        "interpretation": "direct sampled-candidate check on 50 Protocol B probe cells",
                    },
                    {
                        "audit": "sampled candidate-set divergence",
                        "metric": "rho(mean pairwise candidate CD, set regret)",
                        "value": _spearman(geo["mean_candidate_norm_cd"], geo["set_regret"]),
                        "interpretation": "direct sampled-candidate check on 50 Protocol B probe cells",
                    },
                ]
            )

            cut = geo["mean_candidate_norm_cd"].quantile(0.75)
            high = geo[geo["mean_candidate_norm_cd"] >= cut]
            low = geo[geo["mean_candidate_norm_cd"] <= geo["mean_candidate_norm_cd"].quantile(0.25)]
            rows.extend(
                [
                    {
                        "audit": "sampled candidate-set divergence",
                        "metric": "high-divergence action coverage mean",
                        "value": float(high["action_disagreeing_coverage"].mean()),
                        "interpretation": "top quartile by sampled candidate CD",
                    },
                    {
                        "audit": "sampled candidate-set divergence",
                        "metric": "low-divergence action coverage mean",
                        "value": float(low["action_disagreeing_coverage"].mean()),
                        "interpretation": "bottom quartile by sampled candidate CD",
                    },
                ]
            )
            geo.to_csv(OUT_DIR / "sampled_candidate_divergence_probe50.csv", index=False)

    return pd.DataFrame(rows)


def plot_terrain(terrain: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "optimization-limited zone": "#4C78A8",
        "effective-collapse zone": "#54A24B",
        "ambiguity zone": "#7A57C1",
        "decision-fragile zone": "#F58518",
        "information-limited zone": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=180)
    for name, group in terrain.groupby("phase_name"):
        ax.scatter(
            group["feasible_set_divergence"],
            group["blind_search_barrier"],
            s=22 + 45 * group["task_sensitivity"],
            alpha=0.72,
            c=colors.get(name, "#777777"),
            label=name.replace(" zone", ""),
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set_xlabel("Feasible-set divergence")
    ax.set_ylabel("Blind-search barrier")
    ax.set_title("Continuous diagnostic terrain (size = task sensitivity)")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagnostic_terrain_continuous.png")
    fig.savefig(FIG_DIR / "diagnostic_terrain_continuous.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(PHASE_MAP)
    terrain = build_terrain(phase)
    terrain.to_csv(OUT_DIR / "continuous_terrain_axes_by_cell.csv", index=False)

    summary_parts = [terrain_summary(terrain)]
    per_cell, mt_summary = multitask_fragility(terrain)
    if not mt_summary.empty:
        summary_parts.append(mt_summary)
    summary_parts.append(internal_divergence_audit(terrain, per_cell))

    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(OUT_DIR / "diagnostic_terrain_audit_summary.csv", index=False)
    plot_terrain(terrain)

    with (OUT_DIR / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Diagnostic Terrain Audit\n\n"
            "This audit re-expresses the five named diagnostic strata as summaries over "
            "three continuous coordinates: feasible-set divergence, blind-search barrier, "
            "and task sensitivity.  It also checks multi-task fragility and a task-independent "
            "internal-divergence proxy using target-blind witness-neighbor records.\n"
        )

    print(summary.to_string(index=False))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
