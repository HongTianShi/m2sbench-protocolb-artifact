#!/usr/bin/env python
"""Derive a compact Protocol B solver comparison from completed reports.

This script does not rerun embedding search.  It reads the completed public
standard-qrel, HNSW-depth, CE, and evidence-cascade reports and materializes the
review-facing comparison table: existing retrieval paradigms as solvers under
one Protocol B visibility/cost contract.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DATASETS = [
    "beir_fiqa_test",
    "beir_scifact_test",
    "beir_nfcorpus_test",
    "beir_arguana",
    "antique_test",
]


def read_json(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def ci_text(ci: dict | None) -> str:
    if not ci:
        return "--"
    return f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"


def buy_rate(choices: dict) -> float:
    return 1.0 - float(choices.get("summary", 0.0))


def weighted_router(router_key: str) -> dict:
    total_q = 0
    sums = {"utility": 0.0, "regret": 0.0, "cost": 0.0}
    choices: dict[str, float] = {}
    for ds in DATASETS:
        rep = read_json(f"standard_ir_access_{ds}.json")
        router = rep["routers"][router_key]
        q = int(router["test_queries"])
        total_q += q
        sums["utility"] += q * float(router["utility@10"])
        sums["regret"] += q * float(router["regret"])
        sums["cost"] += q * float(router["cost"])
        for k, v in router.get("choices", {}).items():
            choices[k] = choices.get(k, 0.0) + q * float(v)
    return {
        "queries": total_q,
        "utility": sums["utility"] / total_q,
        "regret": sums["regret"] / total_q,
        "cost": sums["cost"] / total_q,
        "choices": {k: v / total_q for k, v in choices.items()},
    }


def main() -> None:
    cascade = read_json("evidence_cascade_router_audit.json")
    hnsw = read_json("hnsw_cascade_router_beir_fiqa_test.json")
    ce = read_json("ce_label_budget_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.json")
    ce_cal = read_json("ce_purchase_calibration_metrics.json")
    routes = {r["name"]: r for r in cascade["routes"]}
    selective = {r["name"]: r for r in cascade["selective_reranking_baselines"]}
    features = {r["name"]: r for r in cascade["feature_ablation"]}
    oracle_u = float(routes["fixed_full"]["utility"] + routes["fixed_full"]["regret"])
    best_fixed_u = max(routes[f"fixed_{v}"]["utility"] for v in ("summary", "pq", "full"))

    def row(name: str, paradigm: str, tier: str, menu: str, r: dict, source: str) -> dict:
        util = float(r["utility"])
        regret = float(r.get("regret", oracle_u - util))
        denom = max(oracle_u - best_fixed_u, 1e-12)
        return {
            "solver": name,
            "paradigm": paradigm,
            "visible_tier": tier,
            "route_menu": menu,
            "utility": util,
            "regret": regret,
            "gap_closed": (util - best_fixed_u) / denom,
            "buy_rate": buy_rate(r.get("choices", {})),
            "full_buy_rate": float(r.get("choices", {}).get("full", 0.0)),
            "ci_vs_best_fixed": ci_text(r.get("paired_diff_vs_fixed_full")) if name != "Fixed full" else "--",
            "source": source,
        }

    rows = [
        row("Fixed summary", "fixed view", "B0", "S", routes["fixed_summary"], "pooled standard qrels"),
        row("Fixed PQ", "fixed compressed view", "B1", "PQ", routes["fixed_pq"], "pooled standard qrels"),
        row("Fixed full", "fixed richest view", "paid", "F", routes["fixed_full"], "pooled standard qrels"),
        row("Cascade exit gate", "cascade", "B1", "S/PQ/F", routes["two_threshold_cascade"], "pooled standard qrels"),
        row("Selective rerank RF", "selective reranking", "B1", "PQ/F", selective["selective_rf_B1_no_dataset"], "pooled standard qrels"),
        row("QPP uncertainty gate", "QPP/uncertainty", "B0", "S/F", weighted_router("summary_qpp_b0"), "per-dataset held-out qrels"),
        row("RF utility predictor", "cost-sensitive utility prediction", "B1", "S/PQ/F", routes["rf_evidence_cascade"], "pooled standard qrels"),
        row("ET utility predictor", "cost-sensitive utility prediction", "B1", "S/PQ/F", routes["et_evidence_cascade"], "pooled standard qrels"),
        row("HistGBR utility predictor", "cost-sensitive utility prediction", "B1", "S/PQ/F", routes["hgb_evidence_cascade"], "pooled standard qrels"),
        row("Ridge visible predictor", "linear utility prediction", "B1", "S/PQ/F", routes["ridge_evidence_cascade"], "pooled standard qrels"),
        row("B0 RF feature ablation", "legal feature ablation", "B0", "S/PQ/F", features["rf_B0_no_dataset"], "pooled standard qrels"),
        row("B1 RF feature ablation", "legal feature ablation", "B1", "S/PQ/F", features["rf_B1_full"], "pooled standard qrels"),
    ]
    cls = selective["selective_hgb_cls_B1_no_dataset"]
    rows.append(row("Learning-to-defer classifier", "deferral classifier", "B1", "PQ/F", cls, "pooled standard qrels"))
    hrow = next(r for r in hnsw["routes"] if r["name"] == "agreement_gate_top1_same")
    h_oracle = float(hrow["utility@10"] + hrow["regret"])
    h_best = max(float(r["utility@10"]) for r in hnsw["routes"] if r["name"].startswith("fixed_"))
    rows.append(
        {
            "solver": "ANN-depth agreement gate",
            "paradigm": "ANN-depth routing",
            "visible_tier": "B1",
            "route_menu": "H16/H64",
            "utility": float(hrow["utility@10"]),
            "regret": float(hrow["regret"]),
            "gap_closed": (float(hrow["utility@10"]) - h_best) / max(h_oracle - h_best, 1e-12),
            "buy_rate": 1.0,
            "full_buy_rate": 0.0,
            "ci_vs_best_fixed": "+.0069 (FiQA HNSW)",
            "source": "HNSW-depth FiQA",
        }
    )
    fixed = {r["policy"]: r for r in ce["fixed_policies"]}
    ab = {r["feature_budget"]: r for r in ce["feature_ablation"]}
    rows.append(
        {
            "solver": "CE cheap-only gate",
            "paradigm": "expensive-view purchase",
            "visible_tier": "B0/B1",
            "route_menu": "cheap/CE",
            "utility": float(ab["+ margin/entropy"]["utility@10"]),
            "regret": float(ab["+ margin/entropy"]["regret"]),
            "gap_closed": float(ab["+ margin/entropy"]["oracle_gap_closed"]),
            "buy_rate": float(ab["+ margin/entropy"]["ce_buy_rate"]),
            "full_buy_rate": float(ab["+ margin/entropy"]["ce_buy_rate"]),
            "ci_vs_best_fixed": f"AUPRC {ce_cal['auprc_raw_predicted_net_gain']:.3f}; Brier {ce_cal['brier_cv_platt']:.3f}; ECE {ce_cal['ece5_cv_platt_equal_count']:.3f}",
            "source": "FiQA CE boundary",
        }
    )
    rows.append(
        {
            "solver": "Oracle route",
            "paradigm": "evaluator-only upper bound",
            "visible_tier": "invalid",
            "route_menu": "S/PQ/F",
            "utility": oracle_u,
            "regret": 0.0,
            "gap_closed": 1.0,
            "buy_rate": 1.0 - cascade["lambda_frontier"][3]["oracle_choices"]["summary"],
            "full_buy_rate": cascade["lambda_frontier"][3]["oracle_choices"]["full"],
            "ci_vs_best_fixed": "evaluator-only",
            "source": "pooled standard qrels",
        }
    )

    # Dry-run hidden-test evidence: random split stability plus held-out-domain stress.
    hidden = {
        "random_split": cascade["split_summary"],
        "leave_one_dataset_out": cascade["leave_one_dataset_out"],
        "readout": "Repeated 60/40 split is the public-dev/private-test proxy; LODO is a harder hidden-domain proxy.",
    }

    # Lambda sweep rescoring: fixed views and default-profile policy means.
    lambda_rows = []
    for lf in cascade["lambda_frontier"]:
        lam = float(lf["lambda"])
        out = {
            "lambda": lam,
            "fixed_summary": lf["fixed_summary"],
            "fixed_pq": lf["fixed_pq"],
            "fixed_full": lf["fixed_full"],
            "oracle": lf["oracle"],
        }
        for key, label in [
            ("rf_evidence_cascade", "rf_router"),
            ("two_threshold_cascade", "cascade_gate"),
            ("hgb_evidence_cascade", "histgbr_router"),
        ]:
            r = routes[key]
            out[label] = float(r["ndcg"] - lam * r["cost"])
        sr = selective["selective_rf_B1_no_dataset"]
        out["selective_rerank_rf"] = float(sr["ndcg"] - lam * sr["cost"])
        lambda_rows.append(out)

    result = {
        "task": "protocol_b_solver_comparison_from_reports",
        "source_reports": [
            "evidence_cascade_router_audit.json",
            "standard_ir_access_*.json",
            "hnsw_cascade_router_beir_fiqa_test.json",
            "ce_label_budget_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2.json",
            "ce_purchase_calibration_metrics.json",
        ],
        "rows": sorted(rows, key=lambda r: r["utility"], reverse=True),
        "hidden_test_dry_run": hidden,
        "lambda_sweep": lambda_rows,
    }
    json_path = REPORT_DIR / "protocol_b_solver_comparison_from_reports.json"
    md_path = REPORT_DIR / "protocol_b_solver_comparison_from_reports.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Existing Retrieval Pipelines as Protocol B Solvers",
        "",
        "| solver | paradigm | tier/menu | utility | regret | gap closed | buy/full | CI or diagnostic |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['solver']} | {r['paradigm']} | {r['visible_tier']} / {r['route_menu']} | "
            f"{r['utility']:.4f} | {r['regret']:.4f} | {r['gap_closed']:.3f} | "
            f"{r['buy_rate']:.2f}/{r['full_buy_rate']:.2f} | {r['ci_vs_best_fixed']} |"
        )
    ss = hidden["random_split"]
    lines += [
        "",
        "## Hidden-Test Dry Run",
        "",
        f"- Repeated split adaptive-best-fixed: {ss['mean_adaptive_minus_best_fixed']:+.4f} "
        f"[{ss['lo']:+.4f},{ss['hi']:+.4f}], positive share {ss['positive_share']:.3f}.",
        "",
        "| held-out dataset | best fixed | best adaptive | delta |",
        "| --- | --- | --- | ---: |",
    ]
    for r in hidden["leave_one_dataset_out"]:
        lines.append(
            f"| {r['heldout_dataset']} | {r['best_fixed']} {r['best_fixed_utility']:.4f} | "
            f"{r['best_adaptive']} {r['best_adaptive_utility']:.4f} | {r['adaptive_minus_best_fixed']:+.4f} |"
        )
    lines += [
        "",
        "## Lambda Sweep",
        "",
        "| lambda | fixed S | fixed PQ | fixed F | cascade | RF | selective RF | oracle |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in lambda_rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['fixed_summary']:.4f} | {r['fixed_pq']:.4f} | "
            f"{r['fixed_full']:.4f} | {r['cascade_gate']:.4f} | {r['rf_router']:.4f} | "
            f"{r['selective_rerank_rf']:.4f} | {r['oracle']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
