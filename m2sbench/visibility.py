from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PUBLIC_TOP_LEVEL_FIELDS = {
    "cell_id",
    "protocol",
    "n_points",
    "summary",
    "granted_views",
    "cost_menu",
    "output_schema",
}
HIDDEN_TOP_LEVEL_FIELDS = {"evaluator_only", "target_points", "family", "budget_id", "seed", "phase_name", "oracle_view", "view_utilities"}


def load_visibility_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_visible_cell(cell: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    visible = {key: value for key, value in cell.items() if key in PUBLIC_TOP_LEVEL_FIELDS}
    if manifest is not None:
        visible["visibility_contract"] = {
            "protocol": manifest.get("protocol", "Protocol B"),
            "leakage_rule": manifest.get("visibility_contract", {}).get("leakage_rule", ""),
        }
    return visible


def assert_no_hidden_fields(cell: dict[str, Any]) -> None:
    overlap = sorted(HIDDEN_TOP_LEVEL_FIELDS.intersection(cell))
    if overlap:
        raise AssertionError(f"Method-visible cell contains evaluator-only fields: {overlap}")


def filter_demo_cells(cells: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    visible_cells = [method_visible_cell(cell, manifest) for cell in cells]
    for cell in visible_cells:
        assert_no_hidden_fields(cell)
    return visible_cells
