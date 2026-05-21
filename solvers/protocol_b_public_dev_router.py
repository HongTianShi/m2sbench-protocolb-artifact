#!/usr/bin/env python
"""Tiny legal Protocol B public-dev router.

This is an executable example for participants. It reads method-visible
`public_dev/*.manifest.jsonl` rows and emits route JSONL. It never reads
reference rows, qrels, support facts, answers, CE scores, or oracle utilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _choose(row: dict[str, Any]) -> str:
    views = set(row.get("declared_views", []))
    features = row.get("visible_features", {})
    slice_id = row.get("slice_id", "")

    if slice_id == "dense_semantic_access":
        margin = float(features.get("cheap_margin", 0.0))
        entropy = float(features.get("cheap_entropy", 0.0))
        agreement = float(features.get("agreement_sketch", 0.0))
        if margin > 0.82 and entropy < 0.35 and "summary" in views:
            return "summary"
        if agreement > 0.65 and "int8" in views:
            return "int8"
        if entropy > 0.55 and "hnsw64" in views:
            return "hnsw64"
        return "full" if "full" in views else sorted(views)[0]

    if slice_id == "2wiki_structured_evidence":
        title = float(features.get("title_overlap", 0.0))
        graph = float(features.get("graph_agreement", 0.0))
        path = float(features.get("path_sketch_signal", 0.0))
        if title > 0.72 and "summary" in views:
            return "summary"
        if path > 0.65 and "two_hop" in views:
            return "two_hop"
        if graph > 0.45 and "one_hop" in views:
            return "one_hop"
        return "full_context" if "full_context" in views else sorted(views)[0]

    if slice_id == "2wiki_hyperlink_stress":
        title = float(features.get("title_overlap", 0.0))
        graph = float(features.get("graph_agreement", 0.0))
        path = float(features.get("path_sketch_signal", 0.0))
        if title > 0.65 and "summary" in views:
            return "summary"
        if path > 0.70 and "hyperlink_2hop" in views:
            return "hyperlink_2hop"
        if graph > 0.45 and "hyperlink_1hop" in views:
            return "hyperlink_1hop"
        return "provided_context" if "provided_context" in views else sorted(views)[0]

    return sorted(views)[0] if views else "summary"


def route_row(row: dict[str, Any]) -> dict[str, Any]:
    route = _choose(row)
    ranked = [route] + [view for view in row.get("declared_views", []) if view != route]
    return {
        "query_id": row["query_id"],
        "cell_id": row["cell_id"],
        "tier": row["tier"],
        "cost_menu": row["cost_menu"],
        "route": route,
        "ranked_views": ranked,
        "method": "protocol_b_public_dev_router",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(json.dumps(route_row(json.loads(line)), sort_keys=True) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
