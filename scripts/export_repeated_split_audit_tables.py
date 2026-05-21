#!/usr/bin/env python
"""Export compact repeated-split audit tables from frozen report JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _repeat_summary(report: dict[str, Any], *, name: str, file: str, family: str, rebuild: str) -> dict[str, Any]:
    rep = report.get("repeat_summary") or report.get("repeated_split_summary") or report.get("repeated_602020") or {}
    if "mean_learned_minus_best_fixed" in rep:
        delta = rep.get("mean_learned_minus_best_fixed")
    elif "learned_minus_best_fixed_mean" in rep:
        delta = rep.get("learned_minus_best_fixed_mean")
    elif "mean_adaptive_minus_best_fixed" in rep:
        delta = rep.get("mean_adaptive_minus_best_fixed")
    else:
        delta = rep.get("best_learned_minus_best_fixed_mean")
    ci = rep.get("ci95") or [rep.get("lo"), rep.get("hi")] or rep.get("best_learned_minus_best_fixed_ci") or []
    if isinstance(ci, dict):
        ci = [ci.get("lo"), ci.get("hi")]
    positive = rep.get("positive_share") or rep.get("positive_split_share")
    repeats = rep.get("repeats") or rep.get("reps") or rep.get("split_reps") or report.get("config", {}).get("repeats")
    return {
        "report": name,
        "file": file,
        "family": family,
        "n_rows": report.get("n_rows") or report.get("total_queries") or report.get("boundary_queries"),
        "repeats": repeats,
        "delta_mean": delta,
        "ci95_low": ci[0] if len(ci) > 0 else "",
        "ci95_high": ci[1] if len(ci) > 1 else "",
        "positive_share": positive,
        "per_repeat_rows_shipped": "yes" if report.get("repeated_split_rows") else "no",
        "rebuild_script": rebuild,
    }


def _ci_pair(row: dict[str, Any]) -> tuple[Any, Any]:
    ci = row.get("delta_vs_best_fixed_ci") or row.get("delta_vs_fixed_full_ci") or ["", ""]
    if isinstance(ci, dict):
        return ci.get("lo", ""), ci.get("hi", "")
    if isinstance(ci, list) and len(ci) >= 2:
        return ci[0], ci[1]
    return "", ""


def main() -> None:
    standard = _load("protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.json")
    standard_rows = standard.get("repeated_split_rows", [])
    _write_csv(REPORTS / "protocol_b_standard_qrel_602020_repeated_split_rows.csv", standard_rows)

    dense_legal = _load("dense_legal_learner_sweep.json")
    dense_additional = _load("dense_additional_learner_sweep.json")
    dense_rows = []
    for source, report in [("dense_legal_learner_sweep", dense_legal), ("dense_additional_learner_sweep", dense_additional)]:
        for row in report.get("summaries", []):
            ci_low, ci_high = _ci_pair(row)
            dense_rows.append(
                {
                    "source": source,
                    "model": row.get("model"),
                    "status": row.get("status"),
                    "n_reps": row.get("n_reps"),
                    "utility_mean": row.get("utility_mean"),
                    "delta_vs_fixed_full_mean": row.get("delta_vs_best_fixed_mean", row.get("delta_vs_fixed_full_mean")),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "positive_split_share": row.get("positive_split_share"),
                    "gap_closed_mean": row.get("gap_closed_mean"),
                }
            )
    _write_csv(REPORTS / "dense_learner_family_sweep_summary.csv", dense_rows)

    structured_specs = [
        ("2Wiki structured", "2wiki_structured_evidence_2000.json", "validated core", "scripts/run_2wiki_structured_evidence_audit.py"),
        ("2Wiki hyperlink 10k", "2wiki_hyperlink_corpus_stress_10000.json", "validated structured stress", "scripts/run_2wiki_hyperlink_corpus_stress.py"),
        ("HotpotQA 5k", "hotpot_structured_evidence_5000_noce.json", "supporting generalization", "scripts/run_hotpot_structured_evidence_audit.py"),
        ("MuSiQue 10k", "musique_structured_evidence_10000_noce.json", "supporting generalization", "scripts/run_musique_structured_evidence_audit.py"),
        ("StrategyQA 2k", "strategyqa_reasoning_evidence_2000_noce.json", "exploratory boundary", "scripts/run_strategyqa_reasoning_evidence_audit.py"),
        ("FEVER 100", "fever_claim_evidence_100_noce.json", "stopped smoke boundary", "scripts/run_fever_claim_evidence_audit.py"),
    ]
    structured_rows = [
        _repeat_summary(_load(file), name=name, file=file, family=family, rebuild=rebuild)
        for name, file, family, rebuild in structured_specs
        if (REPORTS / file).exists()
    ]
    _write_csv(REPORTS / "structured_repeat_summary.csv", structured_rows)

    index_rows = [
        _repeat_summary(
            standard,
            name="standard-qrel dense 60/20/20",
            file="protocol_b_standard_qrel_602020_sentence_transformers_all_MiniLM_L6_v2.json",
            family="validated dense stress",
            rebuild="scripts/run_protocol_b_standard_qrel_602020_audit.py",
        ),
        {
            "report": "dense learner-family sweep",
            "file": "dense_legal_learner_sweep.json; dense_additional_learner_sweep.json",
            "family": "supporting dense weak-signal learner sensitivity",
            "n_rows": "",
            "repeats": dense_legal.get("reps"),
            "delta_mean": "model-specific; see dense_learner_family_sweep_summary.csv",
            "ci95_low": "",
            "ci95_high": "",
            "positive_share": "model-specific",
            "per_repeat_rows_shipped": "no; aggregate model summaries shipped",
            "rebuild_script": "scripts/run_learner_family_sweep.py --slice dense_semantic --config configs/learner_sweep_dense.yaml --out reports/dense_learner_family_sweep_summary.csv",
        },
        *structured_rows,
    ]
    _write_csv(REPORTS / "repeated_split_audit_index.csv", index_rows)

    md = [
        "# Repeated-Split Audit Index",
        "",
        "This index makes the interval evidence inspectable without rerunning heavy audits. "
        "Some frozen reports ship per-repeat rows; others ship aggregate repeated-split summaries only because local scratch folds were not retained in the clean release.",
        "",
        "| Report | Family | Repeats | Delta mean | 95% interval | Positive share | Per-repeat rows shipped | Rebuild script |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in index_rows:
        interval = f"[{row.get('ci95_low')}, {row.get('ci95_high')}]" if row.get("ci95_low") != "" else ""
        md.append(
            "| {report} | {family} | {repeats} | {delta_mean} | {interval} | {positive_share} | {per_repeat_rows_shipped} | `{rebuild_script}` |".format(
                interval=interval,
                **row,
            )
        )
    md.extend(
        [
            "",
            "Companion CSVs:",
            "",
            "- `reports/protocol_b_standard_qrel_602020_repeated_split_rows.csv`: per-repeat rows for the standard-qrel dense 60/20/20 audit.",
            "- `reports/dense_learner_family_sweep_summary.csv`: model-level legal learner sweep summaries, including negative neural/ranker rows.",
            "- `reports/structured_repeat_summary.csv`: compact repeated-split summaries for 2Wiki core/stress plus supporting/boundary structured checks.",
        ]
    )
    (REPORTS / "repeated_split_audit_index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Exported repeated-split audit tables.")


if __name__ == "__main__":
    main()
