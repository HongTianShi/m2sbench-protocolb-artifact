from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "challenge_cells.jsonl",
    "visibility_manifest.json",
    "evaluator_only/hidden_reference.jsonl",
    "public_dev/dense_semantic.manifest.jsonl",
    "public_dev/evaluator_only/dense_semantic.reference.jsonl",
    "public_dev/2wiki_structured.manifest.jsonl",
    "public_dev/evaluator_only/2wiki_structured.reference.jsonl",
    "public_dev/2wiki_hyperlink.manifest.jsonl",
    "public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl",
    "submission_template/solver.py",
    "submission_template/protocol_b_public_dev_dense_semantic_router.jsonl",
    "submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl",
    "submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl",
    "scripts/run_baseline.py",
    "scripts/evaluate_submission.py",
    "scripts/score_protocol_b_public_dev.py",
    "scripts/make_leaderboard.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight submission-kit checks.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "m2sbench_submission_kit_check",
        help="Directory for generated check files. Defaults to the system temp directory.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    if missing:
        raise SystemExit("Missing submission-kit files: " + ", ".join(missing))

    legacy_submission = out_dir / "check_submission.jsonl"
    subprocess.run(
        [sys.executable, "scripts/run_baseline.py", "--out", str(legacy_submission)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "scripts/validate_submission.py", str(legacy_submission)], cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_submission.py",
            str(legacy_submission),
            "--metrics-out",
            str(out_dir / "check_metrics.csv"),
            "--summary-out",
            str(out_dir / "check_metrics_summary.json"),
            "--leaderboard-out",
            str(out_dir / "check_leaderboard.csv"),
        ],
        cwd=ROOT,
        check=True,
    )

    public_dev_jobs = [
        (
            "dense_semantic",
            "submission_template/protocol_b_public_dev_dense_semantic_router.jsonl",
            "public_dev/dense_semantic.manifest.jsonl",
            "public_dev/evaluator_only/dense_semantic.reference.jsonl",
            "menus/dense_semantic.view_menu.json",
            "menus/dense_semantic.cost_menu.json",
        ),
        (
            "2wiki_structured",
            "submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl",
            "public_dev/2wiki_structured.manifest.jsonl",
            "public_dev/evaluator_only/2wiki_structured.reference.jsonl",
            "menus/2wiki_structured.view_menu.json",
            "menus/2wiki_structured.cost_menu.json",
        ),
        (
            "2wiki_hyperlink",
            "submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl",
            "public_dev/2wiki_hyperlink.manifest.jsonl",
            "public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl",
            "menus/2wiki_hyperlink.view_menu.json",
            "menus/2wiki_hyperlink.cost_menu.json",
        ),
    ]
    public_dev_out = out_dir / "public_dev"
    public_dev_out.mkdir(parents=True, exist_ok=True)
    for name, submission, manifest, reference, view_menu, cost_menu in public_dev_jobs:
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_public_dev_pack.py",
                "--manifest",
                manifest,
                "--reference",
                reference,
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/score_protocol_b_public_dev.py",
                submission,
                "--manifest",
                manifest,
                "--reference",
                reference,
                "--view-menu",
                view_menu,
                "--cost-menu",
                cost_menu,
                "--metrics-out",
                str(public_dev_out / f"{name}.metrics.csv"),
                "--summary-out",
                str(public_dev_out / f"{name}.summary.json"),
                "--leaderboard-out",
                str(public_dev_out / f"{name}.leaderboard.csv"),
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"Submission kit check passed. Generated files under {out_dir}")


if __name__ == "__main__":
    main()
