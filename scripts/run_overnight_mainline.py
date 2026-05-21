#!/usr/bin/env python
"""Run the mainline overnight robustness jobs.

The runner is intentionally narrow: 2Wiki hyperlink-corpus robustness, dense
joint-menu profile checks, and final artifact consistency.  It writes logs and
a compact summary under reports/ so partial results remain useful if a later
job fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", "data/external/hf_cache"))
DEFAULT_HYPERLINK_CORPUS = Path(
    os.environ.get(
        "M2SBENCH_2WIKI_HYPERLINK_CORPUS",
        "data/external/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
    )
)


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_job(name: str, cmd: list[str], env: dict[str, str], log_dir: Path, cwd: Path = ROOT) -> dict[str, Any]:
    log_path = log_dir / f"{name}.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {shell_join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "name": name,
        "cmd": shell_join(cmd),
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 3),
        "log": str(log_path.relative_to(ROOT)),
        "cwd": str(cwd.relative_to(ROOT)) if cwd == ROOT or ROOT in cwd.parents else str(cwd),
        "ok": proc.returncode == 0,
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def policy(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in report.get("policy_rows", []):
        if row.get("policy") == name:
            return row
    return None


def control(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in report.get("control_rows", []):
        if row.get("control") == name:
            return row
    return None


def summarize_2wiki(n: int) -> dict[str, Any] | None:
    path = REPORTS / f"2wiki_hyperlink_corpus_stress_{n}.json"
    report = load_json(path)
    if not report:
        return None
    best = policy(report, "best_legal_adaptive") or {}
    fixed = policy(report, "fixed_hyperlink_1hop") or {}
    oracle = policy(report, "oracle_route") or {}
    rep = report.get("repeat_summary", {})
    real_1 = control(report, "real_1hop") or {}
    shuffled = control(report, "shuffled_1hop") or {}
    degree = control(report, "degree_random_1hop") or {}
    real_2 = control(report, "real_2hop") or {}
    random_2 = control(report, "random_2hop") or {}
    return {
        "n": n,
        "report": str(path.relative_to(ROOT)),
        "best_fixed_1hop_utility": fixed.get("utility"),
        "adaptive_utility": best.get("utility"),
        "oracle_utility": oracle.get("utility"),
        "adaptive_minus_fixed_1hop": None if not best or not fixed else best.get("utility", 0.0) - fixed.get("utility", 0.0),
        "repeat_delta": rep.get("learned_minus_best_fixed_mean"),
        "repeat_ci95": rep.get("ci95"),
        "positive_share": rep.get("positive_share"),
        "view_share": best.get("view_share"),
        "real_1hop_utility": real_1.get("utility"),
        "shuffled_1hop_utility": shuffled.get("utility"),
        "degree_random_1hop_utility": degree.get("utility"),
        "real_2hop_utility": real_2.get("utility"),
        "random_2hop_utility": random_2.get("utility"),
        "success_criteria": {
            "delta_gt_0p015": bool((rep.get("learned_minus_best_fixed_mean") or 0.0) > 0.015),
            "ci_lower_gt_0": bool(rep.get("ci95") and rep["ci95"][0] > 0.0),
            "positive_share_ge_0p9": bool((rep.get("positive_share") or 0.0) >= 0.9),
            "real_controls_beat_random": bool(
                (real_1.get("utility") or -1.0) > (shuffled.get("utility") or 9.0)
                and (real_1.get("utility") or -1.0) > (degree.get("utility") or 9.0)
                and (real_2.get("utility") or -1.0) > (random_2.get("utility") or 9.0)
            ),
        },
    }


def summarize_dense(path: Path) -> dict[str, Any] | None:
    report = load_json(path)
    if not report:
        return None
    repeated = report.get("repeated_602020", {})
    return {
        "report": str(path.relative_to(ROOT)),
        "cost_profile": report.get("cost_profile", "C_op"),
        "lambda": report.get("lambda"),
        "best_fixed": (report.get("best_fixed") or {}).get("name"),
        "best_fixed_utility": (report.get("best_fixed") or {}).get("utility"),
        "best_restricted": (report.get("best_restricted_pipeline") or {}).get("name"),
        "best_restricted_utility": (report.get("best_restricted_pipeline") or {}).get("utility"),
        "best_joint_learner": (report.get("best_joint_learner") or {}).get("name"),
        "best_joint_learner_utility": (report.get("best_joint_learner") or {}).get("utility"),
        "oracle_utility": (report.get("joint_oracle") or {}).get("utility"),
        "repeat_delta": repeated.get("best_learned_minus_best_fixed_mean"),
        "repeat_ci": repeated.get("best_learned_minus_best_fixed_ci"),
        "positive_share": repeated.get("positive_split_share"),
        "oracle_view_share": (report.get("joint_oracle") or {}).get("view_share"),
    }


def write_summary(jobs: list[dict[str, Any]], started: str, finished: str, args: argparse.Namespace) -> Path:
    two = [x for x in [summarize_2wiki(2000), summarize_2wiki(args.primary_rows), summarize_2wiki(args.stretch_rows)] if x]
    dense_paths = sorted(REPORTS.glob("joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2*.json"))
    dense = [x for x in (summarize_dense(p) for p in dense_paths) if x]
    changed = []
    for p in [
        *(REPORTS.glob("2wiki_hyperlink_corpus_stress_*.json")),
        *(REPORTS.glob("2wiki_hyperlink_systems_profile_*.json")),
        *(REPORTS.glob("joint_dense_access_menu_beir_fiqa_test__sentence_transformers_all_MiniLM_L6_v2*.json")),
    ]:
        changed.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})

    payload = {
        "started": started,
        "finished": finished,
        "jobs": jobs,
        "two_wiki": two,
        "dense": dense,
        "changed_report_hashes": changed,
    }
    json_path = REPORTS / "overnight_mainline_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(v: Any, digits: int = 4) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    lines = [
        "# Overnight Mainline Summary",
        "",
        f"- Started: {started}",
        f"- Finished: {finished}",
        "- Scope: mainline only: 2Wiki hyperlink-corpus robustness, dense joint-menu robustness, artifact consistency.",
        "- Non-goals: no new benchmark family, no end-to-end RAG claim, no deployment-cost truth claim.",
        "",
        "## Job Status",
        "",
        "| job | ok | seconds | log |",
        "| --- | ---: | ---: | --- |",
    ]
    for job in jobs:
        lines.append(f"| {job['name']} | {job['ok']} | {job['seconds']:.1f} | `{job['log']}` |")

    lines += [
        "",
        "## 2Wiki Hyperlink-Corpus Robustness",
        "",
        "| N | adaptive | fixed 1-hop | oracle | repeat delta | CI | pos. share | controls readout | paper use |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in two:
        ci = row.get("repeat_ci95") or []
        criteria = row.get("success_criteria", {})
        paper_use = "candidate main-table update" if all(criteria.values()) else "artifact robustness only unless manually approved"
        controls = (
            f"real1 {fmt(row.get('real_1hop_utility'))}; "
            f"shuf {fmt(row.get('shuffled_1hop_utility'))}; "
            f"deg {fmt(row.get('degree_random_1hop_utility'))}; "
            f"real2 {fmt(row.get('real_2hop_utility'))}; "
            f"rand2 {fmt(row.get('random_2hop_utility'))}"
        )
        lines.append(
            f"| {row['n']} | {fmt(row.get('adaptive_utility'))} | {fmt(row.get('best_fixed_1hop_utility'))} | "
            f"{fmt(row.get('oracle_utility'))} | {fmt(row.get('repeat_delta'))} | "
            f"[{fmt(ci[0]) if ci else '-'}, {fmt(ci[1]) if ci else '-'}] | {fmt(row.get('positive_share'), 3)} | "
            f"{controls} | {paper_use} |"
        )

    lines += [
        "",
        "## Dense Joint-Menu Robustness",
        "",
        "| profile | lambda | best fixed | fixed utility | best legal learner | learner utility | oracle | repeat delta | pos. share |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in dense:
        lines.append(
            f"| {row.get('cost_profile')} | {fmt(row.get('lambda'), 3)} | {row.get('best_fixed')} | "
            f"{fmt(row.get('best_fixed_utility'))} | {row.get('best_joint_learner')} | "
            f"{fmt(row.get('best_joint_learner_utility'))} | {fmt(row.get('oracle_utility'))} | "
            f"{fmt(row.get('repeat_delta'))} | {fmt(row.get('positive_share'), 3)} |"
        )

    lines += [
        "",
        "## Candidate Paper Updates",
        "",
        "- The 10k 7GB hyperlink-corpus row is the paper-facing robustness number when it satisfies the success criteria; 5k and 2k rows remain artifact consistency checks.",
        "- Dense rows should be used as stability/profile evidence, not as a strong-router claim.",
        "- Any failed or negative run remains in the report index as `should NOT be used` for main-text strengthening.",
        "",
        "## Changed Report Hashes",
        "",
        "| report | sha256 | bytes |",
        "| --- | --- | ---: |",
    ]
    for item in changed:
        lines.append(f"| `{item['path']}` | `{item['sha256']}` | {item['bytes']} |")

    md_path = REPORTS / "overnight_mainline_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-rows", type=int, default=5000)
    parser.add_argument("--stretch-rows", type=int, default=10000)
    parser.add_argument("--primary-repeats", type=int, default=20)
    parser.add_argument("--stretch-repeats", type=int, default=10)
    parser.add_argument("--run-stretch", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-artifact-checks", action="store_true")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--hyperlink-corpus", type=Path, default=DEFAULT_HYPERLINK_CORPUS)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    log_dir = REPORTS / f"overnight_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    env["HF_DATASETS_CACHE"] = str(args.hf_home / "datasets")
    env["M2SBENCH_2WIKI_HYPERLINK_CORPUS"] = str(args.hyperlink_corpus)
    env["TOKENIZERS_PARALLELISM"] = "false"
    jobs: list[dict[str, Any]] = []

    def run(name: str, cmd: list[str], cwd: Path = ROOT) -> bool:
        job = run_job(name, cmd, env, log_dir, cwd=cwd)
        jobs.append(job)
        return job["ok"]

    py = sys.executable
    run(
        f"2wiki_hyperlink_{args.primary_rows}",
        [
            py,
            "scripts/run_2wiki_hyperlink_corpus_stress.py",
            "--max-rows",
            str(args.primary_rows),
            "--max-1hop",
            "16",
            "--max-2hop",
            "16",
            "--repeats",
            str(args.primary_repeats),
        ],
    )
    run(
        f"2wiki_profile_{args.primary_rows}",
        [
            py,
            "scripts/summarize_2wiki_hyperlink_systems_profile.py",
            "--input",
            f"reports/2wiki_hyperlink_corpus_stress_{args.primary_rows}.json",
        ],
    )

    if args.run_stretch:
        run(
            f"2wiki_hyperlink_{args.stretch_rows}",
            [
                py,
                "scripts/run_2wiki_hyperlink_corpus_stress.py",
                "--max-rows",
                str(args.stretch_rows),
                "--max-1hop",
                "16",
                "--max-2hop",
                "16",
                "--repeats",
                str(args.stretch_repeats),
            ],
        )
        run(
            f"2wiki_profile_{args.stretch_rows}",
            [
                py,
                "scripts/summarize_2wiki_hyperlink_systems_profile.py",
                "--input",
                f"reports/2wiki_hyperlink_corpus_stress_{args.stretch_rows}.json",
            ],
        )

    if not args.skip_dense:
        for profile in ["C_op", "C_mem", "C_lat"]:
            run(
                f"dense_joint_{profile}",
                [
                    py,
                    "scripts/run_joint_dense_access_menu_audit.py",
                    "--cost-profile",
                    profile,
                    "--lambda-cost",
                    "0.08",
                    "--split-reps",
                    "30",
                ],
            )
        for lam in ["0.04", "0.16"]:
            run(
                f"dense_joint_C_op_lambda_{lam.replace('.', 'p')}",
                [
                    py,
                    "scripts/run_joint_dense_access_menu_audit.py",
                    "--cost-profile",
                    "C_op",
                    "--lambda-cost",
                    lam,
                    "--split-reps",
                    "10",
                ],
            )

    if not args.skip_artifact_checks:
        run("check_pipeline_demo", [py, "scripts/check_pipeline_demo.py"])
        run(
            "check_submission_kit",
            [py, "scripts/check_submission_kit.py", "--out-dir", "outputs/check_submission_kit"],
        )
        run("pytest", [py, "-m", "pytest", "-q", "tests"])
        validation_jobs = [
            (
                "validate_dense",
                "submission_template/protocol_b_valid_dense.jsonl",
                "menus/dense_semantic.view_menu.json",
                "menus/dense_semantic.cost_menu.json",
                "manifests/dense_semantic.dev_manifest.jsonl",
                True,
            ),
            (
                "validate_2wiki_hyperlink",
                "submission_template/protocol_b_valid_2wiki_hyperlink.jsonl",
                "menus/2wiki_hyperlink.view_menu.json",
                "menus/2wiki_hyperlink.cost_menu.json",
                "manifests/2wiki_hyperlink.dev_manifest.jsonl",
                True,
            ),
            (
                "validate_invalid_leak",
                "submission_template/protocol_b_invalid_leak.jsonl",
                "menus/dense_semantic.view_menu.json",
                "menus/dense_semantic.cost_menu.json",
                "manifests/dense_semantic.dev_manifest.jsonl",
                False,
            ),
        ]
        for name, route, view_menu, cost_menu, manifest, should_pass in validation_jobs:
            ok = run(
                name,
                [
                    py,
                    "scripts/validate_protocol_b_route.py",
                    route,
                    "--view-menu",
                    view_menu,
                    "--cost-menu",
                    cost_menu,
                    "--manifest",
                    manifest,
                ],
            )
            if not should_pass:
                jobs[-1]["ok"] = not ok
                jobs[-1]["expected_failure"] = True
        if run("build_release", [py, "scripts/build_anonymous_release.py"]):
            dist = ROOT / "dist" / "m2sbench_anonymous_release"
            run(
                "dist_check_submission_kit",
                [py, "scripts/check_submission_kit.py", "--out-dir", "outputs/check_submission_kit"],
                cwd=dist,
            )
            run("dist_pytest", [py, "-m", "pytest", "-q", "tests"], cwd=dist)

    finished = datetime.now().isoformat(timespec="seconds")
    summary_path = write_summary(jobs, started, finished, args)
    print(summary_path)
    return 0 if all(j["ok"] for j in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
