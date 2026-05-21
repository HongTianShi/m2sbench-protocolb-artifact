from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_CELL_SCHEMA = ROOT / "challenge_cell_schema.json"
SUBMISSION_SCHEMA = ROOT / "submission_schema.json"
PROTOCOL_B_ROUTE_SCHEMA = ROOT / "schema" / "route_schema.json"
PROTOCOL_B_VIEW_MENU_SCHEMA = ROOT / "schema" / "view_menu_schema.json"
PROTOCOL_B_COST_MENU_SCHEMA = ROOT / "schema" / "cost_menu_schema.json"


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schema_paths() -> dict[str, str]:
    return {
        "challenge_cell_schema": str(CHALLENGE_CELL_SCHEMA),
        "submission_schema": str(SUBMISSION_SCHEMA),
        "protocol_b_route_schema": str(PROTOCOL_B_ROUTE_SCHEMA),
        "protocol_b_view_menu_schema": str(PROTOCOL_B_VIEW_MENU_SCHEMA),
        "protocol_b_cost_menu_schema": str(PROTOCOL_B_COST_MENU_SCHEMA),
        "note": "challenge_cell_schema/submission_schema are legacy controlled-cell quickstart schemas; protocol_b_* entries are paper-facing dense/2Wiki route schemas.",
    }
