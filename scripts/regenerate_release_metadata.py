#!/usr/bin/env python
"""Regenerate artifact inventory and SHA manifest for the clean release surface."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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

SELF_REFERENTIAL = {
    "ARTIFACT_INVENTORY.md",
    "reports/REPORT_MANIFEST_SHA256.md",
}


def should_exclude(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if any(part.endswith("_cache") or part.endswith("_caches") for part in rel.parts):
        return True
    if any(part.startswith("overnight_logs_") for part in rel.parts):
        return True
    if rel.name.startswith(("2wiki_hyper_seed_", "2wiki_hyper_1hop_", "2wiki_hyper_2hop_", "2wiki_title_links_")):
        return True
    if len(rel.parts) >= 2 and rel.parts[0] == "m2sbench" and rel.parts[1] in {"scripts", "research"}:
        return True
    if len(rel.parts) >= 2 and rel.parts[0] == "tests" and rel.name.startswith("test_research_"):
        return True
    if rel.name in {"blocker_system_profile_microbenchmark.json", "overnight_mainline_summary.json"}:
        return True
    return rel.suffix.lower() in EXCLUDE_SUFFIXES


def role_for(rel: str) -> str:
    if rel.startswith("reports/"):
        return "Frozen report or report index"
    if rel.startswith("docs/"):
        return "Documentation / contract / data card"
    if rel.startswith("schema/"):
        return "Machine-readable schema"
    if rel.startswith("menus/"):
        return "View or cost menu declaration"
    if rel.startswith("manifests/"):
        return "Benchmark or slice manifest"
    if rel.startswith("public_dev/"):
        return "Public-dev manifest/reference fixture"
    if rel.startswith("scripts/"):
        return "Runnable script"
    if rel.startswith("solvers/") or rel.startswith("submission_template/"):
        return "Solver or submission template"
    if rel.startswith("figure_prompts/"):
        return "AI-assisted figure prompt provenance"
    if rel.startswith("optional_adapters/"):
        return "Optional adapter, not core reproduction"
    if "/" not in rel:
        return "Top-level reviewer entry point"
    return "Artifact file"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if should_exclude(rel):
            continue
        if path.is_file():
            files.append(rel)
    return sorted(files, key=lambda p: p.as_posix().lower())


def write_inventory(files: list[Path]) -> None:
    lines = [
        "# Artifact Inventory",
        "",
        "This inventory describes the clean paper-facing artifact surface. Development-only files, `.git`, `dist/`, generated `outputs/`, caches, bytecode, large local caches, and legacy manuscript/polish research scaffolding are excluded from the anonymous release archive.",
        "",
        "| Path | Bytes | Role |",
        "| --- | ---: | --- |",
    ]
    for rel in files:
        rel_s = rel.as_posix()
        size = (ROOT / rel).stat().st_size
        lines.append(f"| `{rel_s}` | {size} | {role_for(rel_s)} |")
    lines.append("")
    (ROOT / "ARTIFACT_INVENTORY.md").write_text("\n".join(lines), encoding="utf-8")


def write_hash_manifest(files: list[Path]) -> None:
    lines = [
        "# Report and Contract SHA256 Manifest",
        "",
        "Generated after final artifact-governance edits. Paths are relative to the artifact root. The self-referential inventory and hash-manifest files are intentionally excluded from this checksum table.",
        "",
        "| SHA256 | Path |",
        "| --- | --- |",
    ]
    for rel in files:
        rel_s = rel.as_posix()
        if rel_s in SELF_REFERENTIAL:
            continue
        lines.append(f"| {sha256(ROOT / rel)} | `{rel_s}` |")
    lines.append("")
    out = ROOT / "reports" / "REPORT_MANIFEST_SHA256.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    files = included_files()
    write_inventory(files)
    files = included_files()
    write_hash_manifest(files)
    for generated in ("ARTIFACT_INVENTORY.md", "reports/REPORT_MANIFEST_SHA256.md"):
        text = (ROOT / generated).read_text(encoding="utf-8")
        if "$rel" in text:
            raise SystemExit(f"metadata generation failed: literal $rel found in {generated}")
    print(f"Regenerated inventory for {len(files)} files.")


if __name__ == "__main__":
    main()
