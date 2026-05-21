#!/usr/bin/env python
"""Score a Protocol B public-dev route submission.

This is the paper-facing submission loop for lightweight official public-dev
packs.  It validates the submitted route JSONL against the declared view/cost
menus and released manifest, then joins evaluator-held public-dev references to
emit per-row raw quality, charged cost, utility, regret, and a leaderboard row.
Private-test scoring uses the same route schema and menu ids while withholding
the reference file and returning aggregate leaderboard feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_protocol_b_route import (  # noqa: E402
    action_rules,
    collect_forbidden_keys,
    get_cost_menu_id,
    get_declared_views,
    load_json,
    load_jsonl,
    load_manifest,
    validate_menu_pair,
    validate_row,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0


def _costs(cost_menu: dict[str, Any]) -> dict[str, float]:
    return {key: float(value["declared_cost"]) for key, value in cost_menu["view_costs"].items()}


def _lambda(cost_menu: dict[str, Any]) -> float:
    return float(cost_menu.get("lambda", 0.0))


def _select(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("abstain") is True:
        return "abstain", []
    route = row.get("route")
    if isinstance(route, str) and route:
        return route, [route]
    ranked = row.get("ranked_views")
    if isinstance(ranked, list) and ranked:
        return str(ranked[0]), [str(item) for item in ranked]
    view_set = row.get("view_set")
    if isinstance(view_set, list) and view_set:
        raise ValueError("view_set is not leaderboard-scored unless a menu declares a non-oracle set rule")
    return "abstain", []


def _score_row(
    manifest_row: dict[str, Any],
    submission_row: dict[str, Any] | None,
    reference_row: dict[str, Any],
    costs: dict[str, float],
    lam: float,
    method: str,
) -> dict[str, Any]:
    scores = {key: float(value) for key, value in reference_row["view_scores"].items()}
    utilities = {view: score - lam * costs.get(view, 0.0) for view, score in scores.items()}
    oracle_view = max(utilities, key=lambda view: utilities[view])
    oracle_utility = utilities[oracle_view]
    if submission_row is None:
        selected, selected_set = "missing", []
        raw_quality = 0.0
        charged_cost = 0.0
        utility = 0.0
    else:
        selected, selected_set = _select(submission_row)
        if selected in scores:
            raw_quality = scores[selected]
            charged_cost = costs.get(selected, 0.0)
        else:
            raw_quality = 0.0
            charged_cost = 0.0
        utility = raw_quality - lam * charged_cost
    return {
        "query_id": manifest_row["query_id"],
        "cell_id": manifest_row["cell_id"],
        "slice_id": manifest_row.get("slice_id", ""),
        "method": method,
        "selected_view": selected,
        "raw_quality": round(raw_quality, 6),
        "charged_cost": round(charged_cost, 6),
        "utility": round(utility, 6),
        "oracle_view": oracle_view,
        "oracle_utility": round(oracle_utility, 6),
        "regret": round(max(0.0, oracle_utility - utility), 6),
        "oracle_hit": float(selected == oracle_view),
        "missing_submission": float(submission_row is None),
    }


def _summary(rows: list[dict[str, Any]], method: str, cost_menu_id: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "method": method,
        "cost_menu": cost_menu_id,
        "n_rows": n,
        "mean_raw_quality": round(_mean(rows, "raw_quality"), 6),
        "mean_charged_cost": round(_mean(rows, "charged_cost"), 6),
        "mean_utility": round(_mean(rows, "utility"), 6),
        "mean_regret": round(_mean(rows, "regret"), 6),
        "oracle_hit_rate": round(_mean(rows, "oracle_hit"), 6),
        "missing_submission_rate": round(_mean(rows, "missing_submission"), 6),
        "leaderboard_sort": "ascending mean_regret, then descending mean_utility",
    }


def _write_leaderboard(path: Path, summary: dict[str, Any]) -> None:
    fields = ["method", "n_rows", "mean_utility", "mean_regret", "mean_raw_quality", "mean_charged_cost", "oracle_hit_rate", "cost_menu"]
    row = {field: summary[field] for field in fields}
    _write_csv(path, [row])
    md = "| " + " | ".join(fields) + " |\n"
    md += "| " + " | ".join(["---"] * len(fields)) + " |\n"
    md += "| " + " | ".join(str(row[field]) for field in fields) + " |\n"
    path.with_suffix(".md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--view-menu", type=Path, required=True)
    parser.add_argument("--cost-menu", type=Path, required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for default metrics/summary/leaderboard outputs.",
    )
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--leaderboard-out", type=Path, default=None)
    args = parser.parse_args()

    view_menu = load_json(args.view_menu)
    cost_menu = load_json(args.cost_menu)
    declared_views = get_declared_views(view_menu)
    forbidden_keys = collect_forbidden_keys(view_menu)
    cost_menu_id = get_cost_menu_id(cost_menu)
    menu_errors = validate_menu_pair(view_menu, cost_menu)
    if menu_errors:
        raise SystemExit("; ".join(menu_errors))
    allow_view_set, allow_abstain = action_rules(view_menu, cost_menu)
    manifest = load_manifest(args.manifest)
    manifest_rows = load_jsonl(args.manifest)
    submissions = load_jsonl(args.submission)
    allowed_tiers = set(view_menu.get("allowed_tiers", []))
    validation_errors: list[str] = []
    for row in submissions:
        validation_errors.extend(
            validate_row(
                row,
                declared_views=declared_views,
                cost_menu_id=cost_menu_id,
                allowed_tiers=allowed_tiers,
                manifest=manifest,
                forbidden_keys=forbidden_keys,
                strict_extra_keys=True,
                allow_view_set=allow_view_set,
                allow_abstain=allow_abstain,
            )
        )
    if validation_errors:
        for error in validation_errors:
            print(error, file=sys.stderr)
        raise SystemExit(f"INVALID: {len(validation_errors)} validation error(s)")

    submission_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for row in submissions:
        key = (row["query_id"], row["cell_id"])
        if key in submission_by_key:
            duplicate_keys.append(key)
        submission_by_key[key] = row
    if duplicate_keys:
        preview = ", ".join(str(key) for key in duplicate_keys[:5])
        raise SystemExit(f"Duplicate submitted route row(s): {preview}")
    reference_by_key = {(row["query_id"], row["cell_id"]): row for row in load_jsonl(args.reference)}
    method = args.method or (submissions[0].get("method") if submissions else "unknown")
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        key = (manifest_row["query_id"], manifest_row["cell_id"])
        if key not in reference_by_key:
            raise SystemExit(f"Missing evaluator reference for {key}")
        rows.append(
            _score_row(
                manifest_row,
                submission_by_key.get(key),
                reference_by_key[key],
                _costs(cost_menu),
                _lambda(cost_menu),
                str(method),
            )
        )
    out_dir = args.out_dir or (Path(tempfile.gettempdir()) / "m2sbench_public_dev_score")
    stem = args.submission.stem
    metrics_out = args.metrics_out or out_dir / f"{stem}_public_dev_metrics.csv"
    summary_out = args.summary_out or out_dir / f"{stem}_public_dev_summary.json"
    leaderboard_out = args.leaderboard_out or out_dir / f"{stem}_public_dev_leaderboard.csv"
    _write_csv(metrics_out, rows)
    summary = _summary(rows, str(method), cost_menu_id)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_leaderboard(leaderboard_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
