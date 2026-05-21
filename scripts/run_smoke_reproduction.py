#!/usr/bin/env python
"""Run lightweight smoke reproduction for the paper-facing Protocol B artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_DEV_JOBS = {
    "dense": {
        "submission": "submission_template/protocol_b_public_dev_dense_semantic_router.jsonl",
        "manifest": "public_dev/dense_semantic.manifest.jsonl",
        "reference": "public_dev/evaluator_only/dense_semantic.reference.jsonl",
        "view_menu": "menus/dense_semantic.view_menu.json",
        "cost_menu": "menus/dense_semantic.cost_menu.json",
        "summary_name": "dense_smoke.json",
    },
    "2wiki_structured": {
        "submission": "submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl",
        "manifest": "public_dev/2wiki_structured.manifest.jsonl",
        "reference": "public_dev/evaluator_only/2wiki_structured.reference.jsonl",
        "view_menu": "menus/2wiki_structured.view_menu.json",
        "cost_menu": "menus/2wiki_structured.cost_menu.json",
        "summary_name": "2wiki_structured_smoke.json",
    },
    "2wiki_hyperlink": {
        "submission": "submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl",
        "manifest": "public_dev/2wiki_hyperlink.manifest.jsonl",
        "reference": "public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl",
        "view_menu": "menus/2wiki_hyperlink.view_menu.json",
        "cost_menu": "menus/2wiki_hyperlink.cost_menu.json",
        "summary_name": "2wiki_hyperlink_smoke.json",
    },
}


def _run(cmd: list[str]) -> None:
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_public_dev(name: str, out_dir: Path) -> dict[str, Any]:
    job = PUBLIC_DEV_JOBS[name]
    details_dir = out_dir / "details"
    prefix = details_dir / name
    summary_out = out_dir / job["summary_name"]
    generated_detail = [
        prefix.with_suffix(".metrics.csv"),
        prefix.with_suffix(".leaderboard.csv"),
        prefix.with_suffix(".leaderboard.md"),
        summary_out,
    ]
    _run(
        [
            "scripts/validate_public_dev_pack.py",
            "--manifest",
            job["manifest"],
            "--reference",
            job["reference"],
        ]
    )
    _run(
        [
            "scripts/score_protocol_b_public_dev.py",
            job["submission"],
            "--manifest",
            job["manifest"],
            "--reference",
            job["reference"],
            "--view-menu",
            job["view_menu"],
            "--cost-menu",
            job["cost_menu"],
            "--metrics-out",
            str(prefix.with_suffix(".metrics.csv")),
            "--summary-out",
            str(summary_out),
            "--leaderboard-out",
            str(prefix.with_suffix(".leaderboard.csv")),
        ]
    )
    summary = _load_json(summary_out)
    summary["generated_detail_files"] = [
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in generated_detail
    ]
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_cost_smoke(out_dir: Path) -> dict[str, Any]:
    systems = _load_json(ROOT / "reports" / "unified_systems_profile_audit.json")
    derivation = _load_json(ROOT / "docs" / "cost_profile_derivation.json")
    rows = systems.get("rows", [])
    compact_rows = [
        {
            "view": row["view"],
            "C_op": row.get("C_op"),
            "C_mem": row.get("C_mem"),
            "C_lat": row.get("C_lat"),
            "p95_us": row.get("p95_us"),
            "bytes_touched": row.get("bytes_touched"),
        }
        for row in rows
    ]
    result = {
        "status": "COST PROFILE SMOKE PASSED",
        "task": systems.get("task"),
        "dataset": systems.get("dataset"),
        "profiles": [profile.get("profile_id") for profile in derivation.get("profiles", [])],
        "ce_accounting": derivation.get("ce_accounting", {}),
        "profile_winners": systems.get("profile_winners", {}),
        "n_profile_rows": len(compact_rows),
        "sample_rows": compact_rows[:3],
    }
    out_path = out_dir / "cost_smoke.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "m2sbench_smoke_reproduction",
        help="Directory for generated smoke files. Defaults to the system temp directory.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["dense", "2wiki", "cost"],
        choices=["dense", "2wiki", "cost"],
        help="Smoke components to run.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated_details: list[str] = []
    summary: dict[str, Any] = {"status": "SMOKE REPRODUCTION PASSED", "components": args.components}

    if "dense" in args.components:
        summary["dense"] = _score_public_dev("dense", out_dir)
        generated.append(out_dir / "dense_smoke.json")
        generated_details.extend(summary["dense"].get("generated_detail_files", []))
    if "2wiki" in args.components:
        structured = _score_public_dev("2wiki_structured", out_dir)
        hyperlink = _score_public_dev("2wiki_hyperlink", out_dir)
        combined = {
            "status": "2WIKI SMOKE PASSED",
            "structured": structured,
            "hyperlink": hyperlink,
        }
        (out_dir / "2wiki_smoke.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
        summary["2wiki"] = combined
        generated.append(out_dir / "2wiki_smoke.json")
        generated.append(out_dir / "2wiki_structured_smoke.json")
        generated.append(out_dir / "2wiki_hyperlink_smoke.json")
        generated_details.extend(structured.get("generated_detail_files", []))
        generated_details.extend(hyperlink.get("generated_detail_files", []))
    if "cost" in args.components:
        summary["cost"] = _write_cost_smoke(out_dir)
        generated.append(out_dir / "cost_smoke.json")

    summary_path = out_dir / "smoke_reproduction_summary.json"
    summary["generated"] = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in generated]
    summary["generated_detail_files"] = generated_details
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    generated.append(summary_path)

    print("SMOKE REPRODUCTION PASSED")
    print("Generated:")
    for path in generated:
        display = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"  {display}")


if __name__ == "__main__":
    main()
