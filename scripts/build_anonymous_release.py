#!/usr/bin/env python
"""Build a clean anonymous artifact archive from a working checkout.

The checked-out repository may contain `.git`, caches, local outputs, or test
artifacts while authors are editing. This script creates a review-upload archive
that excludes those development-only files.
"""

from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dist" / "m2sbench_anonymous_release"

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "constructor_cells",
    "dist",
    "outputs",
    "polish",
}

EXCLUDE_SUFFIXES = {
    ".log",
    ".tmp",
    ".pyc",
    ".pyo",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
}


def should_exclude(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if any(part.endswith("_cache") or part.endswith("_caches") for part in path.parts):
        return True
    if any(part.startswith("overnight_logs_") for part in path.parts):
        return True
    if path.name.startswith(("2wiki_hyper_seed_", "2wiki_hyper_1hop_", "2wiki_hyper_2hop_", "2wiki_title_links_")):
        return True
    if len(path.parts) >= 2 and path.parts[0] == "m2sbench" and path.parts[1] == "scripts":
        return True
    if len(path.parts) >= 2 and path.parts[0] == "m2sbench" and path.parts[1] == "research":
        return True
    if len(path.parts) >= 2 and path.parts[0] == "tests" and path.name.startswith("test_research_"):
        return True
    if path.name in {"blocker_system_profile_microbenchmark.json", "overnight_mainline_summary.json"}:
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def copy_clean(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        if should_exclude(rel):
            continue
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--zip", action="store_true", help="Also create <output>.zip")
    args = parser.parse_args()

    out = args.output.resolve()
    root = ROOT.resolve()
    if out == root:
        raise SystemExit("refusing to use repository root as output directory")
    if root not in out.parents:
        raise SystemExit("refusing to write release output outside the repository dist/ directory")
    rel_parts = out.relative_to(root).parts
    if not rel_parts or rel_parts[0] != "dist":
        raise SystemExit("release output must be under the excluded dist/ directory")
    subprocess.run([sys.executable, "scripts/regenerate_release_metadata.py"], cwd=ROOT, check=True)
    copy_clean(root, out)
    print(f"Wrote clean release directory: {out}")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", out)
        print(f"Wrote clean release archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
