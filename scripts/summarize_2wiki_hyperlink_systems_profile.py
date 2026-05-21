from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_INPUT = REPORTS / "2wiki_hyperlink_corpus_stress_2000.json"


def fmt_float(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(val).replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def row_by_policy(report: dict[str, Any], policy: str) -> dict[str, Any]:
    for row in report["policy_rows"]:
        if row["policy"] == policy:
            return row
    raise KeyError(policy)


def row_by_control(report: dict[str, Any], control: str) -> dict[str, Any]:
    for row in report["control_rows"]:
        if row["control"] == control:
            return row
    raise KeyError(control)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stem", default=None, help="Output stem; defaults to the input stem with '_systems_profile'.")
    args = parser.parse_args()

    report = json.load(args.input.open("r", encoding="utf-8"))
    profile = report["corpus_profile"]
    passes = [profile["seed_pass"], profile["first_hop_pass"], profile["second_hop_pass"]]
    total_bytes = sum(p["bytes_scanned"] for p in passes)
    total_seconds = sum(p["seconds"] for p in passes)
    total_lines = sum(p["lines_scanned"] for p in passes)
    n_rows = report["n_rows"]
    cold_profile = {
        "corpus": profile["corpus"],
        "n_rows": n_rows,
        "passes": len(passes),
        "total_bytes_streamed": total_bytes,
        "total_gb_streamed": fmt_float(total_bytes / 1e9, 3),
        "total_lines_scanned": total_lines,
        "total_scan_seconds": fmt_float(total_seconds, 3),
        "mean_scan_mb_per_s": fmt_float(total_bytes / max(total_seconds, 1e-9) / 1e6, 3),
        "cold_materialization_s_per_query": fmt_float(total_seconds / max(n_rows, 1), 4),
        "cold_streamed_mb_per_query": fmt_float(total_bytes / max(n_rows, 1) / 1e6, 3),
        "context_titles": profile["seed_pass"]["needed"],
        "selected_1hop_unique": profile["selected_1hop_unique"],
        "selected_2hop_unique": profile["selected_2hop_unique"],
        "doc_map_size": profile["doc_map_size"],
        "mean_1hop_per_query": fmt_float(profile["mean_1hop_per_query"], 3),
        "mean_2hop_per_query": fmt_float(profile["mean_2hop_per_query"], 3),
        "found_1hop": profile["first_hop_pass"]["found"],
        "found_2hop": profile["second_hop_pass"]["found"],
    }

    summary_rows = [
        {
            "item": "Cold corpus materialization",
            "value": f"{cold_profile['total_gb_streamed']} GB in {cold_profile['total_scan_seconds']} s",
            "readout": f"{cold_profile['mean_scan_mb_per_s']} MB/s; {cold_profile['cold_materialization_s_per_query']} s/query amortized",
        },
        {
            "item": "Local evidence map",
            "value": f"{cold_profile['doc_map_size']} docs",
            "readout": f"{cold_profile['context_titles']} context titles; {cold_profile['selected_1hop_unique']} 1-hop; {cold_profile['selected_2hop_unique']} 2-hop",
        },
        {
            "item": "Best fixed vs adaptive",
            "value": f"{row_by_policy(report, 'fixed_hyperlink_1hop')['utility']:.3f} -> {row_by_policy(report, 'best_legal_adaptive')['utility']:.3f}",
            "readout": f"oracle {row_by_policy(report, 'oracle_route')['utility']:.3f}; repeated delta {report['repeat_summary']['learned_minus_best_fixed_mean']:.4f}",
        },
        {
            "item": "Graph controls",
            "value": f"real 1-hop {row_by_control(report, 'real_1hop')['utility']:.3f}",
            "readout": f"shuffled {row_by_control(report, 'shuffled_1hop')['utility']:.3f}; degree-random {row_by_control(report, 'degree_random_1hop')['utility']:.3f}; random 2-hop {row_by_control(report, 'random_2hop')['utility']:.3f}",
        },
    ]

    payload = {
        "task": "2wiki_hyperlink_corpus_systems_profile",
        "source_report": str(args.input),
        "cold_profile": cold_profile,
        "policy_excerpt": {
            "best_fixed": row_by_policy(report, "fixed_hyperlink_1hop"),
            "best_legal_adaptive": row_by_policy(report, "best_legal_adaptive"),
            "oracle_route": row_by_policy(report, "oracle_route"),
            "repeat_summary": report["repeat_summary"],
        },
        "control_excerpt": {
            name: row_by_control(report, name)
            for name in ["real_1hop", "shuffled_1hop", "degree_random_1hop", "real_2hop", "random_2hop"]
        },
        "summary_rows": summary_rows,
        "scope_note": "This is a cold streaming/materialization profile for a query-local 2Wiki hyperlink evidence acquisition audit, not a full RAG serving benchmark.",
    }

    stem = args.stem or args.input.stem.replace("2wiki_hyperlink_corpus_stress", "2wiki_hyperlink_systems_profile")
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)
    md = [
        "# 2Wiki Hyperlink-Corpus Systems Profile",
        "",
        f"Derived from `{args.input.as_posix()}`. The profile records cold streaming/materialization of the 7GB hyperlink paragraph corpus for the structured evidence-acquisition slice. It is not an answer-generation or GraphRAG serving benchmark.",
        "",
        markdown_table(summary_rows, ["item", "value", "readout"]),
        "",
        "```json",
        json.dumps(cold_profile, indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
