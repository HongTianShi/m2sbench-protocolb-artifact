#!/usr/bin/env python
"""Validate paper-facing Protocol B route JSONL submissions.

This validator is intentionally lightweight: it uses only the Python standard
library and checks the contract properties that plain JSON Schema cannot check
well, including menu membership, one-primary-action semantics, cost-menu
consistency, and recursive evaluator-only leakage keys inside diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ALLOWED_TOP_LEVEL_KEYS = {
    "query_id",
    "cell_id",
    "tier",
    "cost_menu",
    "route",
    "ranked_views",
    "view_set",
    "abstain",
    "method",
    "runtime_s",
    "diagnostics",
}

FORBIDDEN_KEYS = {
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
    "target_points",
    "targets",
    "view_utilities",
    "view_score",
    "view_scores",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")


def collect_forbidden_keys(view_menu: dict[str, Any]) -> set[str]:
    """Collect static and menu-declared evaluator-only field names."""
    keys = set(FORBIDDEN_KEYS)
    for field in ("hidden_evaluator_fields", "evaluator_only_fields"):
        values = view_menu.get(field, [])
        if isinstance(values, list):
            keys.update(normalized_key(str(value)) for value in values if str(value).strip())
    views = view_menu.get("views", [])
    if isinstance(views, list):
        for view in views:
            if not isinstance(view, dict):
                continue
            for value in view.get("forbidden_before_purchase", []):
                if str(value).strip():
                    keys.add(normalized_key(str(value)))
    return keys


def find_forbidden_keys(obj: Any, forbidden_keys: set[str], prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "_line_no":
                continue
            norm = normalized_key(str(key))
            if norm in forbidden_keys:
                hits.append(f"{prefix}.{key}")
            hits.extend(find_forbidden_keys(value, forbidden_keys, f"{prefix}.{key}"))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(find_forbidden_keys(value, forbidden_keys, f"{prefix}[{idx}]"))
    return hits


def get_cost_menu_id(cost_menu: dict[str, Any]) -> str:
    for key in ("cost_menu", "cost_menu_id", "menu_id", "profile_id"):
        value = cost_menu.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("cost menu JSON must contain cost_menu, cost_menu_id, menu_id, or profile_id")


def get_declared_views(view_menu: dict[str, Any]) -> set[str]:
    views = view_menu.get("views")
    if not isinstance(views, list):
        raise ValueError("view menu must contain a views list")
    declared: set[str] = set()
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("view_id"), str):
            raise ValueError("each view menu entry must contain a string view_id")
        declared.add(view["view_id"])
    if not declared:
        raise ValueError("view menu must declare at least one view")
    return declared


def validate_menu_pair(view_menu: dict[str, Any], cost_menu: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cost_menu_id = get_cost_menu_id(cost_menu)
    default_cost_menu = view_menu.get("default_cost_menu")
    accepted_cost_menus: set[str] = set()
    if isinstance(default_cost_menu, str) and default_cost_menu:
        accepted_cost_menus.add(default_cost_menu)
    for key in ("accepted_cost_menus", "allowed_cost_menus"):
        values = view_menu.get(key, [])
        if isinstance(values, list):
            accepted_cost_menus.update(str(value) for value in values if str(value).strip())
    if accepted_cost_menus and cost_menu_id not in accepted_cost_menus:
        errors.append(
            f"supplied cost menu {cost_menu_id!r} is not accepted by the view menu "
            f"(accepted: {', '.join(sorted(accepted_cost_menus))})"
        )

    view_costs = cost_menu.get("view_costs")
    if not isinstance(view_costs, dict):
        errors.append("cost menu must contain a view_costs object")
        return errors

    views = view_menu.get("views")
    if not isinstance(views, list):
        errors.append("view menu must contain a views list")
        return errors

    seen_view_ids: set[str] = set()
    missing_cost_keys: list[str] = []
    for view in views:
        if not isinstance(view, dict):
            errors.append("each view menu entry must be an object")
            continue
        view_id = view.get("view_id")
        cost_key = view.get("cost_key")
        if not isinstance(view_id, str) or not view_id:
            errors.append("each view menu entry must contain a non-empty view_id")
            continue
        if view_id in seen_view_ids:
            errors.append(f"duplicate view_id in view menu: {view_id!r}")
        seen_view_ids.add(view_id)
        if not isinstance(cost_key, str) or not cost_key:
            errors.append(f"view {view_id!r} must contain a non-empty cost_key")
        elif cost_key not in view_costs:
            missing_cost_keys.append(f"{view_id}->{cost_key}")

    if missing_cost_keys:
        errors.append(
            "view cost_key(s) missing from cost_menu.view_costs: "
            + ", ".join(sorted(missing_cost_keys))
        )

    for key, value in view_costs.items():
        if not isinstance(value, dict):
            errors.append(f"cost entry {key!r} must be an object")
        elif not isinstance(value.get("declared_cost"), (int, float)) or value["declared_cost"] < 0:
            errors.append(f"cost entry {key!r} requires non-negative declared_cost")

    return errors


def action_rules(view_menu: dict[str, Any], cost_menu: dict[str, Any]) -> tuple[bool, bool]:
    """Return whether set-valued or abstention actions are enabled.

    Current paper-facing official slices leaderboard-score only a single bought
    route.  `view_set` and `abstain` remain schema extensions, but the menu must
    explicitly declare their scoring rule before the validator accepts them as
    primary actions.
    """
    allow_view_set = bool(view_menu.get("set_rule") or cost_menu.get("set_rule"))
    allow_abstain = bool(
        view_menu.get("abstain_rule")
        or cost_menu.get("abstain_rule")
        or "abstain_cost" in cost_menu
    )
    return allow_view_set, allow_abstain


def load_manifest(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    manifest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        query_id = row.get("query_id")
        cell_id = row.get("cell_id")
        if not isinstance(query_id, str) or not isinstance(cell_id, str):
            line_no = row.get("_line_no", "?")
            raise ValueError(f"{path}:{line_no}: manifest rows require query_id and cell_id")
        manifest[(query_id, cell_id)] = row
    return manifest


def list_value(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{key} must not contain duplicate views")
    return value


def primary_actions(row: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if isinstance(row.get("route"), str) and row["route"]:
        actions.append("route")
    if row.get("route") is None and row.get("ranked_views"):
        actions.append("ranked_views")
    if row.get("view_set"):
        actions.append("view_set")
    if row.get("abstain") is True:
        actions.append("abstain")
    return actions


def validate_row(
    row: dict[str, Any],
    *,
    declared_views: set[str],
    cost_menu_id: str,
    allowed_tiers: set[str],
    manifest: dict[tuple[str, str], dict[str, Any]],
    forbidden_keys: set[str],
    strict_extra_keys: bool,
    allow_view_set: bool = False,
    allow_abstain: bool = False,
) -> list[str]:
    line_no = row.get("_line_no", "?")
    label = f"line {line_no}"
    errors: list[str] = []

    if strict_extra_keys:
        extra = sorted(set(row) - ALLOWED_TOP_LEVEL_KEYS - {"_line_no"})
        if extra:
            errors.append(f"{label}: unknown top-level keys: {', '.join(extra)}")

    for key in ("query_id", "cell_id", "tier", "cost_menu"):
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"{label}: missing or invalid string field {key}")

    if isinstance(row.get("tier"), str) and allowed_tiers and row["tier"] not in allowed_tiers:
        errors.append(f"{label}: tier {row['tier']!r} is not allowed by the view menu")

    if isinstance(row.get("cost_menu"), str) and row["cost_menu"] != cost_menu_id:
        errors.append(
            f"{label}: cost_menu {row['cost_menu']!r} does not match declared menu {cost_menu_id!r}"
        )

    if manifest and isinstance(row.get("query_id"), str) and isinstance(row.get("cell_id"), str):
        key = (row["query_id"], row["cell_id"])
        if key not in manifest:
            errors.append(f"{label}: query_id/cell_id pair is not present in the released manifest")
        else:
            manifest_row = manifest[key]
            for field in ("tier", "cost_menu"):
                if field in manifest_row and row.get(field) != manifest_row[field]:
                    errors.append(
                        f"{label}: {field} {row.get(field)!r} differs from manifest value {manifest_row[field]!r}"
                    )
            manifest_declared = manifest_row.get("declared_views")
            if manifest_declared is not None:
                if (
                    not isinstance(manifest_declared, list)
                    or not manifest_declared
                    or not all(isinstance(item, str) and item for item in manifest_declared)
                ):
                    errors.append(f"{label}: manifest declared_views must be a non-empty list of strings")
                else:
                    declared_views = declared_views.intersection(set(manifest_declared))
                    if not declared_views:
                        errors.append(f"{label}: manifest declared_views has no overlap with the view menu")

    actions = primary_actions(row)
    if len(actions) != 1:
        errors.append(f"{label}: exactly one primary action is required; found {actions or 'none'}")
    if "view_set" in actions and not allow_view_set:
        errors.append(
            f"{label}: view_set is diagnostic for this menu; no set_rule is declared for leaderboard scoring"
        )
    if "abstain" in actions and not allow_abstain:
        errors.append(
            f"{label}: abstain is diagnostic for this menu; no abstain_rule or abstain_cost is declared"
        )

    if "route" in row and row.get("route") is not None:
        if not isinstance(row.get("route"), str) or not row["route"]:
            errors.append(f"{label}: route must be a non-empty string")
        elif row["route"] not in declared_views:
            errors.append(f"{label}: route {row['route']!r} is not declared in the view menu")

    for key in ("ranked_views", "view_set"):
        try:
            values = list_value(row, key)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        missing = sorted(set(values) - declared_views)
        if missing:
            errors.append(f"{label}: {key} contains undeclared views: {', '.join(missing)}")

    if "abstain" in row and not isinstance(row["abstain"], bool):
        errors.append(f"{label}: abstain must be boolean when present")

    if "runtime_s" in row:
        runtime = row["runtime_s"]
        if not isinstance(runtime, (int, float)) or runtime < 0:
            errors.append(f"{label}: runtime_s must be a non-negative number")

    leakage = find_forbidden_keys(row, forbidden_keys)
    if leakage:
        errors.append(f"{label}: evaluator-only or forbidden fields present: {', '.join(leakage)}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Protocol B route JSONL file")
    parser.add_argument("--view-menu", type=Path, required=True, help="Declared view menu JSON")
    parser.add_argument("--cost-menu", type=Path, required=True, help="Declared cost menu JSON")
    parser.add_argument("--manifest", type=Path, help="Optional released query/cell manifest JSONL")
    parser.add_argument(
        "--allow-extra-keys",
        action="store_true",
        help="Allow unknown top-level keys. Forbidden evaluator-only keys are still rejected recursively.",
    )
    args = parser.parse_args(argv)

    try:
        view_menu = load_json(args.view_menu)
        cost_menu = load_json(args.cost_menu)
        declared_views = get_declared_views(view_menu)
        forbidden_keys = collect_forbidden_keys(view_menu)
        cost_menu_id = get_cost_menu_id(cost_menu)
        menu_errors = validate_menu_pair(view_menu, cost_menu)
        if menu_errors:
            raise ValueError("; ".join(menu_errors))
        allow_view_set, allow_abstain = action_rules(view_menu, cost_menu)
        allowed_tiers = set(view_menu.get("allowed_tiers", []))
        manifest = load_manifest(args.manifest)
        rows = load_jsonl(args.submission)
    except Exception as exc:  # noqa: BLE001 - command-line validator should print concise failures
        print(f"validator setup failed: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        query_id = row.get("query_id")
        cell_id = row.get("cell_id")
        if isinstance(query_id, str) and isinstance(cell_id, str):
            key = (query_id, cell_id)
            if key in seen_keys:
                errors.append(f"line {row.get('_line_no', '?')}: duplicate submitted route row for {key}")
            seen_keys.add(key)
        errors.extend(
            validate_row(
                row,
                declared_views=declared_views,
                cost_menu_id=cost_menu_id,
                allowed_tiers=allowed_tiers,
                manifest=manifest,
                forbidden_keys=forbidden_keys,
                strict_extra_keys=not args.allow_extra_keys,
                allow_view_set=allow_view_set,
                allow_abstain=allow_abstain,
            )
        )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"INVALID: {len(errors)} error(s) across {len(rows)} row(s)", file=sys.stderr)
        return 1

    print(
        f"VALID: {len(rows)} row(s), {len(declared_views)} declared view(s), cost menu {cost_menu_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
