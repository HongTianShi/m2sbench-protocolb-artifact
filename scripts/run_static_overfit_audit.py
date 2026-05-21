"""Audit public-static benchmark overfitting risk.

This script treats the released 320 Protocol B cells as a public development
slice, then checks whether the blind-baseline ordering survives resampling and
private-seed regeneration under the same cell-generation contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2sbench.research.data import generate_budget_catalog, make_case
from m2sbench.research.story import (
    ResearchStoryConfig,
    _apply_case_corruption,
    _measure_case_run,
)


SUMMARY_DIR = ROOT / "summaries"
AUDIT_DIR = SUMMARY_DIR / "static_overfit_audit"
PUBLIC_METHODS = [
    "evolutionary-blind",
    "blind-sa",
    "best-of-k-restart-blind",
    "ambiguity-set-sampler",
    "subset-affine-blind",
    "projected-gradient-blind",
]


def _cell_key(frame: pd.DataFrame) -> pd.Series:
    return frame["budget"].astype(str) + "|" + frame["family"].astype(str) + "|" + frame["seed"].astype(str)


def _leaderboard(runs: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    methods = methods or PUBLIC_METHODS
    data = runs[runs["method"].isin(methods)].copy()
    if "protocol" in data.columns:
        data = data[data["protocol"].fillna("protocol_b") == "protocol_b"]
    if "case_id" in data.columns:
        key_cols = ["case_id"]
    else:
        key_cols = [col for col in ["budget", "family", "seed"] if col in data.columns]
    winners = data.loc[data.groupby(key_cols)["norm_cd"].idxmin()].copy()
    win_counts = winners["method"].value_counts().rename("winners")
    rows = []
    for method, group in data.groupby("method"):
        rows.append(
            {
                "method": method,
                "cells": int(group[key_cols].drop_duplicates().shape[0]),
                "mean_norm_cd": float(group["norm_cd"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_iou": float(group["iou"].mean()),
                "winners": int(win_counts.get(method, 0)),
            }
        )
    out = pd.DataFrame(rows)
    out["winner_rate"] = out["winners"] / out["cells"].clip(lower=1)
    return out.sort_values(["mean_norm_cd", "winners"], ascending=[True, False]).reset_index(drop=True)


def _bootstrap_rank_stability(runs: pd.DataFrame, reps: int = 500, seed: int = 17) -> pd.DataFrame:
    data = runs[runs["method"].isin(PUBLIC_METHODS)].copy()
    keys = data[["budget", "family", "seed"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rank_records: dict[str, list[int]] = {method: [] for method in PUBLIC_METHODS}
    top1_records: dict[str, int] = {method: 0 for method in PUBLIC_METHODS}
    top2_records: dict[str, int] = {method: 0 for method in PUBLIC_METHODS}
    for _ in range(reps):
        idx = rng.integers(0, len(keys), size=len(keys))
        parts = []
        for draw_id, pos in enumerate(idx):
            key = keys.iloc[int(pos)]
            mask = (
                (data["budget"] == key["budget"])
                & (data["family"] == key["family"])
                & (data["seed"] == key["seed"])
            )
            part = data.loc[mask].copy()
            part["_draw_id"] = draw_id
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        scores = sample.groupby("method")["norm_cd"].mean().sort_values()
        ordered = scores.index.tolist()
        for rank, method in enumerate(ordered, start=1):
            rank_records[method].append(rank)
        if ordered:
            top1_records[ordered[0]] += 1
        for method in ordered[:2]:
            top2_records[method] += 1
    rows = []
    for method in PUBLIC_METHODS:
        ranks = np.asarray(rank_records[method], dtype=float)
        rows.append(
            {
                "method": method,
                "mean_rank": float(ranks.mean()),
                "median_rank": float(np.median(ranks)),
                "top1_probability": top1_records[method] / reps,
                "top2_probability": top2_records[method] / reps,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_rank", "top1_probability"], ascending=[True, False]).reset_index(drop=True)


def _method_overrides(method_name: str) -> dict[str, object]:
    if method_name == "best-of-k-restart-blind":
        return {"base_method_name": "blind-sa", "k": 4}
    if method_name == "ambiguity-set-sampler":
        return {"chains": 6, "iterations": 80}
    return {}


def _run_private_seed_audit(seed_offset: int = 1000, max_cells: int | None = None) -> pd.DataFrame:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ResearchStoryConfig()
    budget_frame = generate_budget_catalog(master=True)
    challenge = pd.read_csv(SUMMARY_DIR / "challenge_protocol_b.csv").copy()
    challenge = challenge.sort_values(["budget", "family", "seed"]).reset_index(drop=True)
    if max_cells is not None:
        challenge = challenge.head(max_cells).copy()
    rows = []
    total = len(challenge) * len(PUBLIC_METHODS)
    done = 0
    for _, row in challenge.iterrows():
        private_seed = int(row["seed"]) + seed_offset
        case = make_case(
            row["family"],
            row["budget"],
            private_seed,
            protocol="protocol_b",
            n_points=cfg.n_points,
            dense_support_points=cfg.support_points,
            support_pool_size=cfg.support_pool_size,
            budget_frame=budget_frame,
        )
        case = _apply_case_corruption(case, row["corruption_type"], seed=private_seed)
        cell_meta = {
            "suite": "private_seed_audit",
            "budget": row["budget"],
            "family": row["family"],
            "seed": private_seed,
            "public_seed_source": int(row["seed"]),
            "budget_difficulty": row["budget_difficulty"],
            "family_surface": row["family_surface"],
            "family_connectivity": row["family_connectivity"],
            "ambiguity_stratum": row["ambiguity_stratum"],
            "corruption_type": row["corruption_type"],
        }
        for method_name in PUBLIC_METHODS:
            result, metrics, runtime = _measure_case_run(
                method_name,
                case,
                budget_frame,
                **_method_overrides(method_name),
            )
            rows.append(
                {
                    **cell_meta,
                    "method": method_name,
                    "protocol": "protocol_b",
                    **metrics,
                    "runtime_s": runtime,
                    "acceptance_rate": float(result.get("acceptance_rate", np.nan)),
                    "n_hypotheses": int(len(result.get("hypotheses", []))) if "hypotheses" in result else 1,
                }
            )
            done += 1
            if done % 120 == 0 or done == total:
                print(f"private-seed audit progress: {done}/{total}", flush=True)
    out = pd.DataFrame(rows).sort_values(["budget", "family", "seed", "method"]).reset_index(drop=True)
    out.to_csv(AUDIT_DIR / "private_seed_320_runs.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-private-seed-audit", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stress = pd.read_csv(SUMMARY_DIR / "track1_blind_stress_runs.csv")
    phase = pd.read_csv(SUMMARY_DIR / "phase_map.csv")
    public_keys = set(_cell_key(phase[["budget", "family", "seed"]].drop_duplicates()))
    public = stress[
        (stress["suite"] == "challenge")
        & (stress["protocol"] == "protocol_b")
        & (_cell_key(stress).isin(public_keys))
        & (stress["method"].isin(PUBLIC_METHODS))
    ].copy()
    public_leader = _leaderboard(public)
    public_leader.to_csv(AUDIT_DIR / "public_320_leaderboard.csv", index=False)

    bootstrap = _bootstrap_rank_stability(public, reps=args.bootstrap_reps)
    bootstrap.to_csv(AUDIT_DIR / "public_320_bootstrap_rank_stability.csv", index=False)

    reference = stress[
        (stress["suite"] == "reference")
        & (stress["protocol"] == "protocol_b")
        & (stress["method"].isin(PUBLIC_METHODS))
    ].copy()
    reference_leader = _leaderboard(reference) if len(reference) else pd.DataFrame()
    if len(reference_leader):
        reference_leader.to_csv(AUDIT_DIR / "reference_32_leaderboard.csv", index=False)

    private_path = AUDIT_DIR / "private_seed_320_runs.csv"
    if args.generate_private_seed_audit or not private_path.exists():
        if args.generate_private_seed_audit:
            private = _run_private_seed_audit(args.seed_offset, args.max_cells)
        else:
            private = pd.DataFrame()
    else:
        private = pd.read_csv(private_path)
    private_leader = _leaderboard(private) if len(private) else pd.DataFrame()
    if len(private_leader):
        private_leader.to_csv(AUDIT_DIR / "private_seed_320_leaderboard.csv", index=False)
        if len(public_leader):
            delta = public_leader[["method", "mean_norm_cd"]].merge(
                private_leader[["method", "mean_norm_cd"]],
                on="method",
                suffixes=("_public", "_private_seed"),
            )
            delta["abs_delta"] = (delta["mean_norm_cd_private_seed"] - delta["mean_norm_cd_public"]).abs()
            delta.to_csv(AUDIT_DIR / "public_private_seed_method_delta.csv", index=False)

    constructor_leader = pd.DataFrame()
    constructor_path = SUMMARY_DIR / "constructor_full_blind_runs.csv"
    if constructor_path.exists():
        constructor = pd.read_csv(constructor_path)
        constructor_methods = [m for m in PUBLIC_METHODS if m in set(constructor["method"])]
        constructor_leader = _leaderboard(constructor, constructor_methods)
        constructor_leader.to_csv(AUDIT_DIR / "constructor_240_leaderboard.csv", index=False)

    summary_rows = []
    if len(public_leader):
        summary_rows.append(
            {
                "check": "public_320_development_slice",
                "cells": int(public_leader["cells"].max()),
                "finding": f"{public_leader.iloc[0]['method']} has the lowest mean Norm. CD ({public_leader.iloc[0]['mean_norm_cd']:.3f}); top winner share is {public_leader['winner_rate'].max():.3f}.",
            }
        )
    if len(bootstrap):
        top = bootstrap.iloc[0]
        top2 = bootstrap.sort_values("top2_probability", ascending=False).iloc[0]
        summary_rows.append(
            {
                "check": f"public_320_bootstrap_{args.bootstrap_reps}",
                "cells": int(public_leader["cells"].max()) if len(public_leader) else 0,
                "finding": f"{top['method']} has mean rank {top['mean_rank']:.2f}; {top2['method']} appears in the top two with probability {top2['top2_probability']:.3f}.",
            }
        )
    if len(private_leader):
        top = private_leader.iloc[0]
        public_top = public_leader.iloc[0]["method"] if len(public_leader) else "n/a"
        shared_top2 = ""
        if len(public_leader):
            pub_top2 = set(public_leader["method"].head(2))
            prv_top2 = set(private_leader["method"].head(2))
            shared_top2 = f"; top-two set preserved={pub_top2 == prv_top2}"
        summary_rows.append(
            {
                "check": f"private_seed_320_offset_{args.seed_offset}",
                "cells": int(private_leader["cells"].max()),
                "finding": f"Private-seed top method is {top['method']} (mean Norm. CD {top['mean_norm_cd']:.3f}); public top was {public_top}{shared_top2}.",
            }
        )
    if len(reference_leader):
        summary_rows.append(
            {
                "check": "reference_32_same_generator",
                "cells": int(reference_leader["cells"].max()),
                "finding": f"Reference-slice top method is {reference_leader.iloc[0]['method']} (mean Norm. CD {reference_leader.iloc[0]['mean_norm_cd']:.3f}).",
            }
        )
    if len(constructor_leader):
        top = constructor_leader.iloc[0]
        summary_rows.append(
            {
                "check": "constructor_240_independent_cells",
                "cells": int(constructor_leader["cells"].max()),
                "finding": f"Constructor slice top method is {top['method']} (mean Norm. CD {top['mean_norm_cd']:.3f}); this is an independent generator stress slice, not the public 320.",
            }
        )
    summary_rows.append(
        {
            "check": "anti_overfit_contract",
            "cells": 0,
            "finding": "Released cells are a public development/review slice; the same generator and evaluator support private seed regeneration with evaluator-only targets.",
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(AUDIT_DIR / "static_overfit_audit_summary.csv", index=False)
    (AUDIT_DIR / "static_overfit_audit_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
