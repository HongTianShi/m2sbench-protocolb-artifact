#!/usr/bin/env python
"""Check lightweight reproduction dependencies and required artifact materials."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LIGHTWEIGHT_PACKAGES = ["numpy"]
LIGHTWEIGHT_FILES = [
    "requirements.txt",
    "schema/route_schema.json",
    "schema/view_menu_schema.json",
    "schema/cost_menu_schema.json",
    "menus/dense_semantic.view_menu.json",
    "menus/dense_semantic.cost_menu.json",
    "menus/2wiki_structured.view_menu.json",
    "menus/2wiki_structured.cost_menu.json",
    "menus/2wiki_hyperlink.view_menu.json",
    "menus/2wiki_hyperlink.cost_menu.json",
    "public_dev/dense_semantic.manifest.jsonl",
    "public_dev/evaluator_only/dense_semantic.reference.jsonl",
    "public_dev/2wiki_structured.manifest.jsonl",
    "public_dev/evaluator_only/2wiki_structured.reference.jsonl",
    "public_dev/2wiki_hyperlink.manifest.jsonl",
    "public_dev/evaluator_only/2wiki_hyperlink.reference.jsonl",
    "submission_template/protocol_b_public_dev_dense_semantic_router.jsonl",
    "submission_template/protocol_b_public_dev_2wiki_structured_router.jsonl",
    "submission_template/protocol_b_public_dev_2wiki_hyperlink_router.jsonl",
    "reports/unified_systems_profile_audit.json",
    "docs/cost_profile_derivation.json",
]

OPTIONAL_SYSTEM_PACKAGES = ["torch", "faiss", "sentence_transformers", "ir_datasets"]


def _missing_packages(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _install_requirements(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(path)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-missing", action="store_true", help="Install missing lightweight packages from requirements.txt.")
    parser.add_argument("--check-system", action="store_true", help="Also report optional GPU/systems-profile packages.")
    args = parser.parse_args()

    missing_files = [item for item in LIGHTWEIGHT_FILES if not (ROOT / item).exists()]
    if missing_files:
        raise SystemExit("Missing required lightweight reproduction files: " + ", ".join(missing_files))

    missing_light = _missing_packages(LIGHTWEIGHT_PACKAGES)
    if missing_light and args.install_missing:
        _install_requirements(ROOT / "requirements.txt")
        missing_light = _missing_packages(LIGHTWEIGHT_PACKAGES)
    if missing_light:
        raise SystemExit(
            "Missing lightweight package(s): "
            + ", ".join(missing_light)
            + ". Run: python scripts/check_reproduction_prereqs.py --install-missing"
        )

    result = {
        "status": "REPRODUCTION PREFLIGHT PASSED",
        "python": sys.version.split()[0],
        "lightweight_packages": {name: "available" for name in LIGHTWEIGHT_PACKAGES},
        "required_files": len(LIGHTWEIGHT_FILES),
    }
    if args.check_system:
        missing_system = _missing_packages(OPTIONAL_SYSTEM_PACKAGES)
        result["optional_system_packages"] = {
            name: ("missing" if name in missing_system else "available") for name in OPTIONAL_SYSTEM_PACKAGES
        }
        result["system_note"] = "Optional packages are needed only for heavy profiler/rebuild paths, not smoke reproduction."
    print(json.dumps(result, indent=2, sort_keys=True))
    print("REPRODUCTION PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
