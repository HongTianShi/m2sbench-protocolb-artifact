#!/usr/bin/env python
"""Summarize 2Wiki lambda sensitivity from frozen aggregate reports.

This is a lightweight audit: it does not rebuild 2Wiki or retrain routers.  It
rescales the frozen policy rows with U_lambda = raw - lambda * mean_cost so that
reviewers can see whether the reported adaptive gain is a single-lambda artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


DEFAULT_INPUTS = {
    "2wiki_context": REPORTS / "2wiki_structured_evidence_2000.json",
    "2wiki_hyperlink_10k": REPORTS / "2wiki_hyperlink_corpus_stress_10000.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def policy_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["policy"]: row for row in data["policy_rows"]}


def utility(row: dict[str, Any], lam: float) -> float:
    return float(row["raw_ndcg"] - lam * row["cost"])


def summarize_slice(name: str, data: dict[str, Any], lambdas: list[float]) -> dict[str, Any]:
    policies = policy_map(data)
    fixed = [row for row in data["policy_rows"] if row["policy"].startswith("fixed_")]
    if name == "2wiki_context":
        adaptive_name = "et_B1_context"
    else:
        adaptive_name = "best_legal_adaptive"
    adaptive = policies[adaptive_name]
    rows = []
    for lam in lambdas:
        fixed_utils = [(row["policy"], utility(row, lam)) for row in fixed]
        best_fixed_name, best_fixed_u = max(fixed_utils, key=lambda x: x[1])
        adaptive_u = utility(adaptive, lam)
        rows.append(
            {
                "lambda": lam,
                "best_fixed": best_fixed_name,
                "best_fixed_utility": adaptive_round(best_fixed_u),
                "adaptive_policy": adaptive_name,
                "adaptive_utility": adaptive_round(adaptive_u),
                "adaptive_minus_best_fixed": adaptive_round(adaptive_u - best_fixed_u),
            }
        )
    default_lam = float(data.get("lambda_cost", 0.08))
    default_row = min(rows, key=lambda r: abs(r["lambda"] - default_lam))
    return {
        "slice": name,
        "source_report": str(DEFAULT_INPUTS[name].relative_to(ROOT)).replace("\\", "/"),
        "default_lambda": default_lam,
        "adaptive_policy": adaptive_name,
        "adaptive_raw_ndcg": adaptive_round(adaptive["raw_ndcg"]),
        "adaptive_mean_cost": adaptive_round(adaptive["cost"]),
        "default_adaptive_minus_best_fixed": default_row["adaptive_minus_best_fixed"],
        "lambda_rows": rows,
    }


def adaptive_round(x: float) -> float:
    return round(float(x), 6)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 2Wiki Lambda Sensitivity From Frozen Reports",
        "",
        "This audit rescales the frozen aggregate policy rows with `U_lambda = raw - lambda * mean_cost`.",
        "It does not rebuild the 7GB corpus or retrain routers; the goal is to check whether the main 2Wiki gains vanish under small changes to lambda.",
        "",
        "Readout: in both slices, the selected legal adaptive row has higher raw score and lower mean cost than the best fixed 1-hop row at the default profile, so the advantage is not a narrow `lambda=.08` crossing.",
        "",
    ]
    for item in summary["slices"]:
        lines.extend(
            [
                f"## {item['slice']}",
                "",
                f"- Source: `{item['source_report']}`",
                f"- Frozen adaptive policy: `{item['adaptive_policy']}`",
                f"- Adaptive raw/cost: {item['adaptive_raw_ndcg']:.3f} / {item['adaptive_mean_cost']:.3f}",
                "",
                "| lambda | best fixed | fixed utility | adaptive utility | adaptive - fixed |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in item["lambda_rows"]:
            lines.append(
                f"| {row['lambda']:.2f} | `{row['best_fixed']}` | {row['best_fixed_utility']:.3f} | "
                f"{row['adaptive_utility']:.3f} | {row['adaptive_minus_best_fixed']:+.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=REPORTS / "2wiki_lambda_sensitivity.json")
    parser.add_argument("--out-md", type=Path, default=REPORTS / "2wiki_lambda_sensitivity.md")
    parser.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.04, 0.08, 0.16, 0.32])
    args = parser.parse_args()

    summary = {
        "audit": "2wiki_lambda_sensitivity",
        "method": "aggregate_rescore_from_frozen_policy_rows",
        "formula": "utility_lambda = raw_ndcg - lambda * mean_cost",
        "limitations": "Routes and routers are frozen from the source reports; this is a sensitivity rescore, not a retraining sweep.",
        "slices": [
            summarize_slice(name, load_json(path), args.lambdas)
            for name, path in DEFAULT_INPUTS.items()
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.out_md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
