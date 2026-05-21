from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_demo_cells(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append({key: value for key, value in row.items() if key != "evaluator_only"})
        reference = {"cell_id": row["cell_id"]}
        reference.update(row.get("evaluator_only", {}))
        reference_rows.append(reference)
    return public_rows, reference_rows
