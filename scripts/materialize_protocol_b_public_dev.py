#!/usr/bin/env python
"""Materialize lightweight public-dev Protocol B scoring packs.

The heavy paper audits are frozen under reports/.  This script creates a
reviewer-facing public-dev loop with method-visible manifests, evaluator-held
reference utilities, and example submissions for the official dense and 2Wiki
slices.  The packs are deliberately small enough to ship in the artifact while
exercising the same route schema, view menus, cost menus, and scorer used by
future submissions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _costs(menu_path: Path) -> dict[str, float]:
    menu = _load_json(menu_path)
    return {key: float(value["declared_cost"]) for key, value in menu["view_costs"].items()}


def _utility(raw: dict[str, float], costs: dict[str, float], lam: float) -> dict[str, float]:
    return {view: raw[view] - lam * costs.get(view, 0.0) for view in raw}


def _best_view(raw: dict[str, float], costs: dict[str, float], lam: float) -> str:
    utilities = _utility(raw, costs, lam)
    return max(utilities, key=lambda view: utilities[view])


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def build_dense(n: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    views = ["summary", "binary", "pq", "int8", "hnsw16", "hnsw64", "full", "ce"]
    costs = _costs(ROOT / "menus" / "dense_semantic.cost_menu.json")
    lam = 0.08
    manifest: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    router: list[dict[str, Any]] = []
    for i in range(n):
        query_id = f"dense_dev_q{i:04d}"
        cell_id = f"dense_dev_cell_{i:04d}"
        kind = i % 5
        margin = rng.random()
        entropy = rng.random()
        ce_signal = rng.random()
        raw = {
            "summary": _bounded(0.03 + 0.04 * rng.random()),
            "binary": _bounded(0.23 + 0.08 * rng.random()),
            "pq": _bounded(0.22 + 0.10 * rng.random()),
            "int8": _bounded(0.28 + 0.10 * rng.random()),
            "hnsw16": _bounded(0.27 + 0.10 * rng.random()),
            "hnsw64": _bounded(0.29 + 0.09 * rng.random()),
            "full": _bounded(0.30 + 0.12 * rng.random()),
            "ce": _bounded(0.31 + 0.15 * rng.random()),
        }
        if kind == 0:
            raw["summary"] = _bounded(raw["summary"] + 0.22)
        elif kind == 1:
            raw["hnsw16"] = _bounded(raw["hnsw16"] + 0.12)
        elif kind == 2:
            raw["int8"] = _bounded(raw["int8"] + 0.10)
        elif kind == 3:
            raw["ce"] = _bounded(raw["ce"] + 0.18)
        else:
            raw["pq"] = _bounded(raw["pq"] + 0.09)
        best = _best_view(raw, costs, lam)
        visible_route = "hnsw16"
        if margin > 0.72:
            visible_route = "summary"
        elif ce_signal > 0.78 and entropy > 0.45:
            visible_route = "ce"
        elif entropy > 0.66:
            visible_route = "int8"
        elif margin < 0.18:
            visible_route = "hnsw64"
        manifest.append(
            {
                "query_id": query_id,
                "cell_id": cell_id,
                "slice_id": "dense_semantic_access",
                "split": "public_dev",
                "tier": "B1_score",
                "cost_menu": "dense-semantic-v1-op",
                "declared_views": views,
                "visible_features": {
                    "candidate_count": 50 + (i % 37),
                    "cheap_margin": round(margin, 4),
                    "cheap_entropy": round(entropy, 4),
                    "agreement_sketch": round(ce_signal, 4),
                },
            }
        )
        reference.append(
            {
                "query_id": query_id,
                "cell_id": cell_id,
                "scorer_id": "public_dev_dense_ndcg10_v1",
                "view_scores": raw,
                "private_note": "Public-dev reference. These fields are evaluator-held during submission.",
            }
        )
        fixed.append({"query_id": query_id, "cell_id": cell_id, "tier": "B1_score", "cost_menu": "dense-semantic-v1-op", "route": "hnsw64", "method": "fixed_hnsw64"})
        router.append({"query_id": query_id, "cell_id": cell_id, "tier": "B1_score", "cost_menu": "dense-semantic-v1-op", "route": visible_route, "method": "cheap_visible_router"})
        assert best in views
    return manifest, reference, fixed, router


def build_2wiki(n: int, seed: int, *, hyperlink: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    if hyperlink:
        views = ["summary", "provided_context", "hyperlink_1hop", "hyperlink_2hop", "full_local_pool"]
        cost_menu = "2wiki-hyper-v1-op"
        slice_id = "2wiki_hyperlink_stress"
        scorer_id = "public_dev_2wiki_hyperlink_support_title_v1"
        costs = _costs(ROOT / "menus" / "2wiki_hyperlink.cost_menu.json")
    else:
        views = ["summary", "one_hop", "two_hop", "full_context", "ce"]
        cost_menu = "2wiki-structured-v1-op"
        slice_id = "2wiki_structured_evidence"
        scorer_id = "public_dev_2wiki_support_title_v1"
        costs = _costs(ROOT / "menus" / "2wiki_structured.cost_menu.json")
    lam = 0.08
    manifest: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    fixed: list[dict[str, Any]] = []
    router: list[dict[str, Any]] = []
    types = ["comparison", "bridge", "bridge_comparison", "inference"]
    for i in range(n):
        row_type = types[i % len(types)]
        query_id = f"{'hyper' if hyperlink else 'ctx'}_2wiki_q{i:04d}"
        cell_id = f"{'hyper' if hyperlink else 'ctx'}_2wiki_cell_{i:04d}"
        ambiguity = rng.random()
        graph_agreement = rng.random()
        path_signal = rng.random()
        if hyperlink:
            raw = {
                "summary": _bounded(0.64 + 0.16 * rng.random()),
                "provided_context": _bounded(0.72 + 0.13 * rng.random()),
                "hyperlink_1hop": _bounded(0.75 + 0.14 * rng.random()),
                "hyperlink_2hop": _bounded(0.72 + 0.16 * rng.random()),
                "full_local_pool": _bounded(0.70 + 0.18 * rng.random()),
            }
            if row_type == "comparison":
                raw["summary"] = _bounded(raw["summary"] + 0.15)
            elif row_type == "bridge":
                raw["hyperlink_1hop"] = _bounded(raw["hyperlink_1hop"] + 0.12)
            elif row_type == "bridge_comparison":
                raw["hyperlink_2hop"] = _bounded(raw["hyperlink_2hop"] + 0.12)
            else:
                raw["full_local_pool"] = _bounded(raw["full_local_pool"] + 0.10)
            fixed_view = "hyperlink_1hop"
            visible_route = "summary" if ambiguity < 0.22 else ("hyperlink_2hop" if path_signal > 0.72 else "hyperlink_1hop")
        else:
            raw = {
                "summary": _bounded(0.72 + 0.12 * rng.random()),
                "one_hop": _bounded(0.86 + 0.08 * rng.random()),
                "two_hop": _bounded(0.82 + 0.10 * rng.random()),
                "full_context": _bounded(0.84 + 0.09 * rng.random()),
                "ce": _bounded(0.82 + 0.13 * rng.random()),
            }
            if row_type == "comparison":
                raw["summary"] = _bounded(raw["summary"] + 0.13)
            elif row_type == "bridge":
                raw["one_hop"] = _bounded(raw["one_hop"] + 0.08)
            elif row_type == "bridge_comparison":
                raw["two_hop"] = _bounded(raw["two_hop"] + 0.10)
            else:
                raw["full_context"] = _bounded(raw["full_context"] + 0.08)
            fixed_view = "one_hop"
            visible_route = "summary" if ambiguity < 0.18 else ("two_hop" if path_signal > 0.65 else "one_hop")
        manifest.append(
            {
                "query_id": query_id,
                "cell_id": cell_id,
                "slice_id": slice_id,
                "split": "public_dev",
                "tier": "B1_hyperlink" if hyperlink else "B1_path",
                "cost_menu": cost_menu,
                "declared_views": views,
                "visible_features": {
                    "question_type": row_type,
                    "title_overlap": round(1.0 - ambiguity, 4),
                    "graph_agreement": round(graph_agreement, 4),
                    "path_sketch_signal": round(path_signal, 4),
                },
            }
        )
        reference.append(
            {
                "query_id": query_id,
                "cell_id": cell_id,
                "scorer_id": scorer_id,
                "view_scores": raw,
                "private_note": "Public-dev reference. Support facts, answers, triples, and view scores are evaluator-held during submission.",
            }
        )
        fixed.append({"query_id": query_id, "cell_id": cell_id, "tier": "B1_hyperlink" if hyperlink else "B1_path", "cost_menu": cost_menu, "route": fixed_view, "method": f"fixed_{fixed_view}"})
        router.append({"query_id": query_id, "cell_id": cell_id, "tier": "B1_hyperlink" if hyperlink else "B1_path", "cost_menu": cost_menu, "route": visible_route, "method": "cheap_visible_router"})
    return manifest, reference, fixed, router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-rows", type=int, default=80)
    parser.add_argument("--structured-rows", type=int, default=120)
    parser.add_argument("--hyperlink-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    public = ROOT / "public_dev"
    evaluator_public = public / "evaluator_only"
    submission = ROOT / "submission_template"
    builders = [
        ("dense_semantic", build_dense(args.dense_rows, args.seed)),
        ("2wiki_structured", build_2wiki(args.structured_rows, args.seed + 1, hyperlink=False)),
        ("2wiki_hyperlink", build_2wiki(args.hyperlink_rows, args.seed + 2, hyperlink=True)),
    ]
    for prefix, (manifest, reference, fixed, router) in builders:
        _write_jsonl(public / f"{prefix}.manifest.jsonl", manifest)
        _write_jsonl(evaluator_public / f"{prefix}.reference.jsonl", reference)
        _write_jsonl(submission / f"protocol_b_public_dev_{prefix}_fixed.jsonl", fixed)
        _write_jsonl(submission / f"protocol_b_public_dev_{prefix}_router.jsonl", router)
    print(
        "Wrote method-visible public-dev manifests under "
        f"{public.relative_to(ROOT)}, evaluator-held references under "
        f"{evaluator_public.relative_to(ROOT)}, and example submissions under submission_template/"
    )


if __name__ == "__main__":
    main()
