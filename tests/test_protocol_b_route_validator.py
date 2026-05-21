from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_protocol_b_route.py"


def run_validator(
    submission: str, view_menu: str, cost_menu: str, manifest: str | None = None
) -> subprocess.CompletedProcess[str]:
    cmd = [
            sys.executable,
            str(VALIDATOR),
            str(ROOT / submission),
            "--view-menu",
            str(ROOT / view_menu),
            "--cost-menu",
            str(ROOT / cost_menu),
    ]
    if manifest:
        cmd.extend(["--manifest", str(ROOT / manifest)])
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_protocol_b_valid_fixtures_pass() -> None:
    cases = [
        (
            "submission_template/protocol_b_valid_dense.jsonl",
            "menus/dense_semantic.view_menu.json",
            "menus/dense_semantic.cost_menu.json",
            "manifests/dense_semantic.dev_manifest.jsonl",
        ),
        (
            "submission_template/protocol_b_valid_2wiki_structured.jsonl",
            "menus/2wiki_structured.view_menu.json",
            "menus/2wiki_structured.cost_menu.json",
            "manifests/2wiki_structured.dev_manifest.jsonl",
        ),
        (
            "submission_template/protocol_b_valid_2wiki_hyperlink.jsonl",
            "menus/2wiki_hyperlink.view_menu.json",
            "menus/2wiki_hyperlink.cost_menu.json",
            "manifests/2wiki_hyperlink.dev_manifest.jsonl",
        ),
    ]
    for submission, view_menu, cost_menu, manifest in cases:
        result = run_validator(submission, view_menu, cost_menu, manifest)
        assert result.returncode == 0, result.stderr
        assert "VALID:" in result.stdout


def test_protocol_b_invalid_leak_fixture_fails() -> None:
    result = run_validator(
        "submission_template/protocol_b_invalid_leak.jsonl",
        "menus/dense_semantic.view_menu.json",
        "menus/dense_semantic.cost_menu.json",
    )
    assert result.returncode == 1
    assert "ce_score_margin" in result.stderr
