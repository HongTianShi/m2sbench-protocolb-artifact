"""Audit explicit downstream action definitions for the M2S-Bench paper.

The paper reports cost-adjusted utility, grouped actions, and
decision-fragile cells.  This script makes the synthetic action contract
white-box: topology, geometry, route, and continuous structural-fidelity
actions are deterministic functions of evaluator-side structural fields.

Inputs are existing release summaries; no hidden vendor data are used.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "summaries"
OUT = SUMMARIES / "action_contract_audit"
PHASE_MAP = SUMMARIES / "phase_map.csv"
PAIR_FILE = (
    SUMMARIES
    / "cikm_batch1_information_access_20260505"
    / "matched_summary_witness_pairs_topk.csv"
)
PAIR_FILE_FALLBACK = (
    SUMMARIES / "diagnostic_terrain_audit" / "multitask_fragility_pairs_top5.csv"
)
TRACK3_DECISION = SUMMARIES / "track3_decision_table.csv"

LAMBDA = 0.08
POINT_COST = 0.58
HOLE_THRESHOLD = 40.0
ASPECT_THRESHOLD = 1.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def key(row: dict[str, str], prefix: str = "") -> tuple[str, str, int]:
    return (
        row[f"{prefix}budget"],
        row[f"{prefix}family"],
        int(float(row[f"{prefix}seed"])),
    )


def f(row: dict[str, str], col: str) -> float:
    return float(row[col])


def bool_int(value: bool) -> int:
    return 1 if value else 0


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase_rows = read_csv(PHASE_MAP)
    pair_path = PAIR_FILE if PAIR_FILE.exists() else PAIR_FILE_FALLBACK
    pair_rows = read_csv(pair_path)
    decision_rows = read_csv(TRACK3_DECISION)

    cells: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in phase_rows:
        k = key(row)
        blind_cd = f(row, "blind_norm_cd")
        point_cd = f(row, "sampler_point_norm_cd")
        point_raw_gain = blind_cd - point_cd
        point_cost_adjusted_gain = point_raw_gain - LAMBDA * POINT_COST
        topology_hole_rich = bool_int(f(row, "hole_proxy") >= HOLE_THRESHOLD)
        geometry_elongated = bool_int(f(row, "bbox_aspect_ratio") >= ASPECT_THRESHOLD)
        route_escalate = bool_int(f(row, "access_value_gain") > 0.0)
        cells[k] = {
            "budget": row["budget"],
            "family": row["family"],
            "seed": int(float(row["seed"])),
            "phase_name": row["phase_name"],
            "hole_proxy": f(row, "hole_proxy"),
            "topology_hole_rich": topology_hole_rich,
            "bbox_aspect_ratio": f(row, "bbox_aspect_ratio"),
            "geometry_elongated": geometry_elongated,
            "access_value_gain": f(row, "access_value_gain"),
            "route_escalate": route_escalate,
            "blind_norm_cd": blind_cd,
            "point_norm_cd": point_cd,
            "structural_fidelity_raw_gain": point_raw_gain,
            "structural_fidelity_cost_adjusted_gain": point_cost_adjusted_gain,
        }

    pair_audit: list[dict[str, object]] = []
    for row in pair_rows:
        if int(float(row["neighbor_rank"])) > 5:
            continue
        qk = key(row, "query_")
        nk = key(row, "neighbor_")
        if qk not in cells or nk not in cells:
            continue
        q = cells[qk]
        n = cells[nk]
        pair_audit.append(
            {
                "representation": row["representation"],
                "query_key": "|".join(map(str, qk)),
                "neighbor_key": "|".join(map(str, nk)),
                "neighbor_rank": int(float(row["neighbor_rank"])),
                "summary_distance": float(row["distance"]),
                "topology_disagree": bool_int(
                    q["topology_hole_rich"] != n["topology_hole_rich"]
                ),
                "geometry_disagree": bool_int(
                    q["geometry_elongated"] != n["geometry_elongated"]
                ),
                "routing_disagree": bool_int(
                    q["route_escalate"] != n["route_escalate"]
                ),
                "continuous_gain_gap": abs(
                    float(q["structural_fidelity_cost_adjusted_gain"])
                    - float(n["structural_fidelity_cost_adjusted_gain"])
                ),
                "original_action_disagree": float(row["action_disagree"]),
                "original_view_disagree": float(row["view_disagree"]),
                "family_disagree": float(row["family_disagree"]),
            }
        )

    def avg(col: str, rows: list[dict[str, object]] = pair_audit) -> float:
        return mean(float(r[col]) for r in rows)

    phase_counts = Counter(str(v["phase_name"]) for v in cells.values())
    by_phase = defaultdict(list)
    for v in cells.values():
        by_phase[str(v["phase_name"])].append(v)

    by_phase_rows: list[dict[str, object]] = []
    for phase_name, rows in sorted(by_phase.items()):
        by_phase_rows.append(
            {
                "diagnostic_label": phase_name,
                "n": len(rows),
                "topology_hole_rich_share": mean(
                    float(r["topology_hole_rich"]) for r in rows
                ),
                "geometry_elongated_share": mean(
                    float(r["geometry_elongated"]) for r in rows
                ),
                "route_escalate_share": mean(float(r["route_escalate"]) for r in rows),
                "mean_structural_fidelity_raw_gain": mean(
                    float(r["structural_fidelity_raw_gain"]) for r in rows
                ),
                "mean_structural_fidelity_cost_adjusted_gain": mean(
                    float(r["structural_fidelity_cost_adjusted_gain"]) for r in rows
                ),
            }
        )

    synthetic_decision = {
        r["access_level"]: r
        for r in decision_rows
        if r["split"] == "synthetic_heldout"
    }
    utility_rows = []
    for access_level in [
        "moments_only",
        "occupancy_grid",
        "raster",
        "point_cloud",
        "family_aware_oracle",
    ]:
        if access_level in synthetic_decision:
            row = synthetic_decision[access_level]
            utility_rows.append(
                {
                    "access_level": access_level,
                    "balanced_action_accuracy": float(row["balanced_accuracy"]),
                    "lambda_cost_adjusted_utility": float(row["utility_lambda"]),
                    "access_cost": float(row["access_cost"]),
                }
            )

    cell_rows = [dict(v) for v in cells.values()]
    summary_rows = [
        {
            "audit_item": "topology_action",
            "white_box_definition": f"hole-rich support indicator 1[hole_proxy >= {HOLE_THRESHOLD:g}]",
            "positive_share": mean(float(v["topology_hole_rich"]) for v in cells.values()),
            "matched_summary_disagree_at5": avg("topology_disagree"),
            "paper_reading": "tests whether matched summaries can hide support/topology differences",
        },
        {
            "audit_item": "geometry_action",
            "white_box_definition": f"elongated support indicator 1[bbox_aspect_ratio >= {ASPECT_THRESHOLD:g}]",
            "positive_share": mean(float(v["geometry_elongated"]) for v in cells.values()),
            "matched_summary_disagree_at5": avg("geometry_disagree"),
            "paper_reading": "tests whether matched summaries can hide geometry-scale decisions",
        },
        {
            "audit_item": "routing_action",
            "white_box_definition": "escalate to structural view if cost-adjusted access_value_gain > 0",
            "positive_share": mean(float(v["route_escalate"]) for v in cells.values()),
            "matched_summary_disagree_at5": avg("routing_disagree"),
            "paper_reading": "turns the access menu into a summary-only versus structural-view routing decision",
        },
        {
            "audit_item": "continuous_fidelity_utility",
            "white_box_definition": f"A_fid=-NormCD; reported gain is blind_cd - point_cd - {LAMBDA:g}*{POINT_COST:g}",
            "positive_share": mean(
                1.0
                if float(v["structural_fidelity_cost_adjusted_gain"]) > 0.0
                else 0.0
                for v in cells.values()
            ),
            "matched_summary_disagree_at5": mean(
                1.0 if float(r["continuous_gain_gap"]) > 0.05 else 0.0
                for r in pair_audit
            ),
            "paper_reading": "removes discrete labels and scores the value of access as structure-error reduction under cost",
        },
        {
            "audit_item": "grouped_policy_context",
            "white_box_definition": "balanced action accuracy over deterministic access/interface choices",
            "positive_share": "",
            "matched_summary_disagree_at5": "",
            "paper_reading": (
                "synthetic held-out: occupancy utility "
                f"{float(synthetic_decision['occupancy_grid']['utility_lambda']):.3f}, "
                "moments "
                f"{float(synthetic_decision['moments_only']['utility_lambda']):.3f}, "
                "point "
                f"{float(synthetic_decision['point_cloud']['utility_lambda']):.3f}"
            ),
        },
    ]

    write_csv(
        OUT / "white_box_action_by_cell.csv",
        cell_rows,
        [
            "budget",
            "family",
            "seed",
            "phase_name",
            "hole_proxy",
            "topology_hole_rich",
            "bbox_aspect_ratio",
            "geometry_elongated",
            "access_value_gain",
            "route_escalate",
            "blind_norm_cd",
            "point_norm_cd",
            "structural_fidelity_raw_gain",
            "structural_fidelity_cost_adjusted_gain",
        ],
    )
    write_csv(
        OUT / "white_box_action_pair_audit.csv",
        pair_audit,
        [
            "representation",
            "query_key",
            "neighbor_key",
            "neighbor_rank",
            "summary_distance",
            "topology_disagree",
            "geometry_disagree",
            "routing_disagree",
            "continuous_gain_gap",
            "original_action_disagree",
            "original_view_disagree",
            "family_disagree",
        ],
    )
    write_csv(
        OUT / "white_box_action_by_diagnostic_label.csv",
        by_phase_rows,
        [
            "diagnostic_label",
            "n",
            "topology_hole_rich_share",
            "geometry_elongated_share",
            "route_escalate_share",
            "mean_structural_fidelity_raw_gain",
            "mean_structural_fidelity_cost_adjusted_gain",
        ],
    )
    write_csv(
        OUT / "white_box_action_utility_rows.csv",
        utility_rows,
        [
            "access_level",
            "balanced_action_accuracy",
            "lambda_cost_adjusted_utility",
            "access_cost",
        ],
    )
    write_csv(
        OUT / "white_box_action_summary.csv",
        summary_rows,
        [
            "audit_item",
            "white_box_definition",
            "positive_share",
            "matched_summary_disagree_at5",
            "paper_reading",
        ],
    )

    summary_json = {
        "n_cells": len(cells),
        "n_pairs_top5": len(pair_audit),
        "lambda": LAMBDA,
        "point_cost": POINT_COST,
        "pair_source": str(pair_path.relative_to(ROOT)),
        "thresholds": {
            "hole_proxy": HOLE_THRESHOLD,
            "bbox_aspect_ratio": ASPECT_THRESHOLD,
        },
        "phase_counts": dict(phase_counts),
        "topology_disagree_at5": avg("topology_disagree"),
        "geometry_disagree_at5": avg("geometry_disagree"),
        "routing_disagree_at5": avg("routing_disagree"),
        "continuous_gain_gap_gt_005_at5": mean(
            1.0 if float(r["continuous_gain_gap"]) > 0.05 else 0.0
            for r in pair_audit
        ),
        "continuous_fidelity_mean_raw_gain": mean(
            float(v["structural_fidelity_raw_gain"]) for v in cells.values()
        ),
        "continuous_fidelity_mean_cost_adjusted_gain": mean(
            float(v["structural_fidelity_cost_adjusted_gain"]) for v in cells.values()
        ),
        "continuous_fidelity_positive_cost_adjusted_share": mean(
            1.0 if float(v["structural_fidelity_cost_adjusted_gain"]) > 0.0 else 0.0
            for v in cells.values()
        ),
        "synthetic_heldout_utilities": {
            k: float(v["utility_lambda"]) for k, v in synthetic_decision.items()
        },
    }
    (OUT / "white_box_action_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8"
    )

    readme = f"""# White-box action contract audit

This audit turns the synthetic downstream actions used by M2S-Bench into
deterministic evaluator-side tasks.

- Topology action: hole-rich support indicator, `hole_proxy >= {HOLE_THRESHOLD:g}`.
- Geometry action: elongated support indicator, `bbox_aspect_ratio >= {ASPECT_THRESHOLD:g}`.
- Routing action: escalate to a structural view when the cost-adjusted access gain is positive.
- Continuous fidelity utility: `A_fid = -NormCD`, with point-view gain computed as
  `blind_norm_cd - point_norm_cd - {LAMBDA:g} * {POINT_COST:g}`.

The matched-summary disagreement rates are computed on top-5 nearest neighbors
under the released summary representation, so they ask whether cells that look
equivalent through the compact summary can induce different white-box actions.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
