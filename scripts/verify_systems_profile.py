#!/usr/bin/env python
"""Verify frozen systems-profile rows and derived cost profiles.

This is a lightweight consistency check over retained JSON rows. It does not
rerun GPU/CPU profilers or claim deployment-cost truth.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _close(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    report_path = ROOT / "reports" / "unified_systems_profile_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    max_materialized = max(float(row["bytes_touched"] + row["index_size_bytes"]) for row in rows)
    max_p95 = max(float(row["p95_us"]) for row in rows)
    errors: list[str] = []
    for row in rows:
        view = row["view"]
        materialized = float(row["bytes_touched"] + row["index_size_bytes"])
        if int(row.get("materialized_bytes", -1)) != int(materialized):
            errors.append(f"{view}: materialized_bytes mismatch")
        expected_mem = materialized / max_materialized
        if not _close(float(row["C_mem"]), expected_mem):
            errors.append(f"{view}: C_mem mismatch {row['C_mem']} != {expected_mem}")
        expected_lat = float(row["p95_us"]) / max_p95
        if not _close(float(row["C_lat"]), expected_lat):
            errors.append(f"{view}: C_lat mismatch {row['C_lat']} != {expected_lat}")
        expected_qps = 1e6 / float(row["p50_us"])
        if not _close(float(row["qps_from_p50"]), expected_qps, tol=1e-6):
            errors.append(f"{view}: qps_from_p50 mismatch {row['qps_from_p50']} != {expected_qps}")
        if not _close(float(row["C_op"]), float(row["cumulative_cost"])):
            errors.append(f"{view}: C_op mismatch {row['C_op']} != {row['cumulative_cost']}")

    for winner in report.get("profile_winners", []):
        profile = winner["profile"]
        utilities = winner["utilities"]
        expected_winner = max(utilities, key=utilities.get)
        if winner["winner"] != expected_winner:
            errors.append(f"{profile}: winner mismatch {winner['winner']} != {expected_winner}")
        if not _close(float(winner["winner_utility"]), float(utilities[expected_winner])):
            errors.append(f"{profile}: winner_utility mismatch")

    if errors:
        raise SystemExit("SYSTEMS PROFILE VERIFICATION FAILED\n" + "\n".join(errors))
    print(json.dumps({"status": "SYSTEMS PROFILE VERIFICATION PASSED", "rows": len(rows)}, indent=2))
    print("SYSTEMS PROFILE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
