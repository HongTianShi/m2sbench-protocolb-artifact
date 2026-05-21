#!/usr/bin/env python
"""Validate a public-dev manifest/reference pack.

This lightweight checker complements JSON Schema. It checks the concrete files
used by the reviewer-facing public-dev scorer: method-visible manifests must not
contain hidden/evaluator-only aliases, reference rows must contain view scores,
and manifest/reference keys must match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_protocol_b_route import find_forbidden_keys, load_jsonl  # noqa: E402


FORBIDDEN_PUBLIC_DEV_KEYS = {
    "answer",
    "answers",
    "ce_score",
    "ce_scores",
    "ce_score_margin",
    "evidence_triple",
    "evidence_triples",
    "evaluator_only",
    "full_score",
    "full_scores",
    "hidden_reference",
    "hidden_target",
    "label",
    "labels",
    "oracle_utility",
    "oracle_utilities",
    "oracle_view",
    "private_seed",
    "qrel",
    "qrels",
    "source_label",
    "source_labels",
    "support_fact",
    "support_facts",
    "support_title",
    "support_titles",
    "target",
    "targets",
    "view_score",
    "view_scores",
    "view_utilities",
}


def _key(row: dict[str, Any]) -> tuple[str, str]:
    query_id = row.get("query_id")
    cell_id = row.get("cell_id")
    if not isinstance(query_id, str) or not isinstance(cell_id, str):
        raise ValueError("rows require string query_id and cell_id")
    return query_id, cell_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.manifest)
    reference_rows = load_jsonl(args.reference)
    errors: list[str] = []

    manifest_keys: set[tuple[str, str]] = set()
    for row in manifest_rows:
        line_no = row.get("_line_no", "?")
        try:
            key = _key(row)
        except ValueError as exc:
            errors.append(f"{args.manifest}:{line_no}: {exc}")
            continue
        if key in manifest_keys:
            errors.append(f"{args.manifest}:{line_no}: duplicate manifest key {key}")
        manifest_keys.add(key)
        visible = row.get("visible_features", {})
        hits = find_forbidden_keys(visible, FORBIDDEN_PUBLIC_DEV_KEYS, "$.visible_features")
        if hits:
            errors.append(f"{args.manifest}:{line_no}: forbidden evaluator-only feature key(s): {', '.join(hits)}")
        declared_views = row.get("declared_views")
        if not isinstance(declared_views, list) or not declared_views:
            errors.append(f"{args.manifest}:{line_no}: declared_views must be a non-empty list")

    reference_keys: set[tuple[str, str]] = set()
    for row in reference_rows:
        line_no = row.get("_line_no", "?")
        try:
            key = _key(row)
        except ValueError as exc:
            errors.append(f"{args.reference}:{line_no}: {exc}")
            continue
        if key in reference_keys:
            errors.append(f"{args.reference}:{line_no}: duplicate reference key {key}")
        reference_keys.add(key)
        view_scores = row.get("view_scores")
        if not isinstance(view_scores, dict) or not view_scores:
            errors.append(f"{args.reference}:{line_no}: view_scores must be a non-empty object")
        elif not all(isinstance(value, (int, float)) for value in view_scores.values()):
            errors.append(f"{args.reference}:{line_no}: all view_scores values must be numeric")

    missing_refs = sorted(manifest_keys - reference_keys)
    extra_refs = sorted(reference_keys - manifest_keys)
    if missing_refs:
        errors.append(f"missing reference rows for {len(missing_refs)} manifest key(s)")
    if extra_refs:
        errors.append(f"reference contains {len(extra_refs)} key(s) absent from manifest")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(f"PUBLIC DEV PACK INVALID: {len(errors)} error(s)")

    print(
        json.dumps(
            {
                "status": "PUBLIC DEV PACK VALID",
                "manifest": str(args.manifest),
                "reference": str(args.reference),
                "rows": len(manifest_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
