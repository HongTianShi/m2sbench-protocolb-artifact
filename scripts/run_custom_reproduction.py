#!/usr/bin/env python
"""Interactively choose lightweight reproduction components."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ask(label: str) -> bool:
    answer = input(f"Run {label}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()) / "m2sbench_smoke_custom")
    args = parser.parse_args()

    components: list[str] = []
    if _ask("dense public-dev smoke"):
        components.append("dense")
    if _ask("2Wiki structured/hyperlink public-dev smoke"):
        components.append("2wiki")
    if _ask("cost-profile smoke"):
        components.append("cost")
    if not components:
        print("No components selected.")
        return
    subprocess.run(
        [
            sys.executable,
            "scripts/run_smoke_reproduction.py",
            "--out-dir",
            str(args.out_dir),
            "--components",
            *components,
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
