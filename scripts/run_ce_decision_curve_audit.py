#!/usr/bin/env python
"""Decision curve for CE-as-view purchase under Protocol B.

Reads the saved cross-encoder access-boundary CSV and evaluates thresholds on
predicted net CE gain.  This turns the CE buyer from a single-threshold result
into a decision-theoretic curve: utility, regret, buy rate, precision, recall,
and false-buy cost as the purchase threshold becomes more conservative.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
BOUNDARY = REPORT_DIR / (
    "cross_encoder_access_boundary_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2__"
    "cross_encoder_ms_marco_MiniLM_L_6_v2.csv"
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path, default=BOUNDARY)
    ap.add_argument("--lambda-cost", type=float, default=0.08)
    ap.add_argument("--ce-cost", type=float, default=0.45)
    args = ap.parse_args()

    rows = []
    with args.boundary.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows in {args.boundary}")

    bi = np.asarray([float(r["bi_ndcg"]) for r in rows], dtype="float32")
    ce = np.asarray([float(r["cross_ndcg"]) for r in rows], dtype="float32")
    pred = np.asarray([float(r["predicted_net_gain"]) for r in rows], dtype="float32")
    worth = (ce - args.lambda_cost * args.ce_cost) > bi
    oracle = np.maximum(bi, ce - args.lambda_cost * args.ce_cost)
    fixed_cheap_u = float(bi.mean())
    fixed_ce_u = float((ce - args.lambda_cost * args.ce_cost).mean())
    oracle_u = float(oracle.mean())

    thresholds = sorted(set(float(x) for x in np.quantile(pred, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))
    thresholds = [min(thresholds) - 1e-6] + thresholds + [max(thresholds) + 1e-6, 0.0, 0.02, 0.04]
    thresholds = sorted(set(round(t, 8) for t in thresholds))
    curve = []
    for thr in thresholds:
        buy = pred > thr
        raw_vec = np.where(buy, ce, bi)
        cost_vec = buy.astype("float32") * args.ce_cost
        util_vec = np.where(buy, ce - args.lambda_cost * args.ce_cost, bi)
        false_buy = buy & (~worth)
        true_buy = buy & worth
        precision = float(true_buy.sum() / max(1, buy.sum()))
        recall = float(true_buy.sum() / max(1, worth.sum()))
        curve.append(
            {
                "threshold": float(thr),
                "raw_ndcg": float(raw_vec.mean()),
                "mean_cost": float(cost_vec.mean()),
                "utility": float(util_vec.mean()),
                "regret": float((oracle - util_vec).mean()),
                "ce_buy_rate": float(buy.mean()),
                "precision": precision,
                "recall": recall,
                "false_buy_rate": float(false_buy.mean()),
                "false_buy_cost": float(false_buy.mean() * args.ce_cost),
            }
        )
    best = max(curve, key=lambda r: r["utility"])
    conservative = min((r for r in curve if r["threshold"] >= 0.0), key=lambda r: (r["regret"], -r["precision"]))

    out = {
        "task": "ce_decision_curve_audit",
        "boundary_csv": (
            args.boundary.relative_to(ROOT).as_posix()
            if args.boundary.is_relative_to(ROOT)
            else args.boundary.as_posix()
        ),
        "test_queries": len(rows),
        "lambda": args.lambda_cost,
        "ce_cost": args.ce_cost,
        "worth_buy_prevalence": float(worth.mean()),
        "fixed_cheap_utility": fixed_cheap_u,
        "fixed_ce_utility": fixed_ce_u,
        "oracle_utility": oracle_u,
        "best_threshold_row": best,
        "conservative_nonnegative_threshold_row": conservative,
        "curve": curve,
    }
    json_path = REPORT_DIR / "ce_decision_curve_audit.json"
    md_path = REPORT_DIR / "ce_decision_curve_audit.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# CE Purchase Decision Curve",
        "",
        f"- Test queries: {len(rows):,}",
        f"- Worth-buy prevalence: {out['worth_buy_prevalence']:.3f}",
        f"- Fixed cheap utility: {fixed_cheap_u:.4f}",
        f"- Fixed CE utility: {fixed_ce_u:.4f}",
        f"- Oracle utility: {oracle_u:.4f}",
        "",
        "| threshold | raw NDCG | cost | utility | regret | CE buy rate | precision | recall | false-buy cost |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in curve:
        lines.append(
            f"| {row['threshold']:.4f} | {row['raw_ndcg']:.4f} | {row['mean_cost']:.4f} | "
            f"{row['utility']:.4f} | {row['regret']:.4f} | "
            f"{row['ce_buy_rate']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['false_buy_cost']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps({"best": best, "conservative": conservative}, indent=2))


if __name__ == "__main__":
    main()
