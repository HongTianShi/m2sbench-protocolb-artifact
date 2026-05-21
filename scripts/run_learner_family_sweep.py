#!/usr/bin/env python
"""Build a learner-family diagnostic summary from released sweep reports.

This utility is a diagnostic audit, not a model-selection leaderboard.  It is
intended for Protocol B slices where an evaluator-only oracle has headroom but
legal learned gains are weak: the CSV answers whether changing learner family
rescues the slice, which families are stable, and which complex learners
underperform under the same visibility/cost contract.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value.strip("\"'")


def load_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small config shape used by configs/learner_sweep_dense.yaml.

    This keeps the diagnostic command runnable with only the standard library.
    It supports top-level scalars, folded text blocks, top-level lists of
    scalars or mappings, and one-level nested mappings.
    """
    raw_lines = [line.rstrip("\n") for line in text.splitlines()]
    config: dict[str, Any] = {}
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith(" "):
            raise ValueError(f"unexpected indentation at line {i + 1}: {line}")
        if ":" not in line:
            raise ValueError(f"expected key/value at line {i + 1}: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == ">":
            i += 1
            block: list[str] = []
            while i < len(raw_lines) and (raw_lines[i].startswith(" ") or not raw_lines[i].strip()):
                if raw_lines[i].strip():
                    block.append(raw_lines[i].strip())
                i += 1
            config[key] = " ".join(block)
            continue
        if value:
            config[key] = parse_scalar(value)
            i += 1
            continue

        i += 1
        child_lines: list[str] = []
        while i < len(raw_lines) and (raw_lines[i].startswith(" ") or not raw_lines[i].strip()):
            if raw_lines[i].strip():
                child_lines.append(raw_lines[i])
            i += 1
        if not child_lines:
            config[key] = {}
            continue
        if child_lines[0].lstrip().startswith("- "):
            items: list[Any] = []
            current: dict[str, Any] | None = None
            for child in child_lines:
                stripped = child.strip()
                if stripped.startswith("- "):
                    payload = stripped[2:].strip()
                    if ":" in payload:
                        current = {}
                        k, v = payload.split(":", 1)
                        current[k.strip()] = parse_scalar(v)
                        items.append(current)
                    else:
                        current = None
                        items.append(parse_scalar(payload))
                elif current is not None and ":" in stripped:
                    k, v = stripped.split(":", 1)
                    current[k.strip()] = parse_scalar(v)
                else:
                    raise ValueError(f"unsupported list item under {key}: {child}")
            config[key] = items
        else:
            mapping: dict[str, Any] = {}
            for child in child_lines:
                stripped = child.strip()
                if ":" not in stripped:
                    raise ValueError(f"unsupported mapping item under {key}: {child}")
                k, v = stripped.split(":", 1)
                mapping[k.strip()] = parse_scalar(v)
            config[key] = mapping
    return config


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        try:
            data = load_minimal_yaml(text)
        except Exception:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"config must contain a mapping: {path}")
    return data


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def ci_pair(row: dict[str, Any]) -> tuple[Any, Any]:
    ci = row.get("delta_vs_best_fixed_ci") or row.get("delta_vs_fixed_full_ci")
    if isinstance(ci, dict):
        return ci.get("lo", ""), ci.get("hi", "")
    if isinstance(ci, list) and len(ci) >= 2:
        return ci[0], ci[1]
    return "", ""


def delta_mean(row: dict[str, Any]) -> Any:
    return row.get("delta_vs_best_fixed_mean", row.get("delta_vs_fixed_full_mean", ""))


def infer_family(model: str) -> str:
    name = model.lower()
    if name == "oracle":
        return "evaluator-only oracle"
    if name.startswith("fixed_"):
        return "fixed reference"
    if "rule" in name or "qpp" in name or "cascade" in name:
        return "QPP/rule"
    if "rank" in name or "lambdamart" in name:
        return "rank-learning"
    if "mlp" in name or "tabnet" in name:
        return "neural"
    if any(token in name for token in ["rf", "tree", "gbdt", "lightgbm", "xgb", "catboost", "adaboost", "extra_trees"]):
        return "tree/boosted tree"
    if any(token in name for token in ["ridge", "lasso", "elasticnet", "logistic"]):
        return "linear/calibrated"
    if any(token in name for token in ["svr", "kernel", "knn"]):
        return "kernel/nonparametric"
    return "other"


def choice_share(row: dict[str, Any], view_order: list[str] | None = None) -> str:
    choices = row.get("choice_share") or row.get("choices") or {}
    if not isinstance(choices, dict):
        return ""
    parts = []
    keys = view_order or list(choices.keys())
    for key in keys:
        if key in choices:
            parts.append(f"{key}:{float(choices[key]):.3f}")
    return "; ".join(parts)


def note_for(row: dict[str, Any], configured_note: str) -> str:
    status = str(row.get("status", "ok"))
    if status and status not in {"ok", "None"}:
        if status.startswith(("failed", "skipped", "unavailable")):
            return status if not configured_note else f"{configured_note}; {status}"
        if status.startswith("rule_"):
            return configured_note if configured_note else f"selected rule: {status}"
    return configured_note or status


def collect_rows(config: dict[str, Any], root: Path) -> list[dict[str, str]]:
    family_map = dict(config.get("family_map") or {})
    notes = dict(config.get("notes") or {})
    exclude = set(config.get("exclude_models") or [])
    view_order = list(config.get("view_order") or [])
    dedupe = bool(config.get("deduplicate_models", True))
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    for spec in config.get("input_reports", []):
        if not isinstance(spec, dict):
            raise SystemExit("each input_reports item must be a mapping")
        source = str(spec.get("source") or spec.get("path"))
        report_path = root / str(spec["path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report.get("summaries", []):
            model = str(row.get("model", ""))
            if not model or model in exclude:
                continue
            if dedupe and model in seen:
                continue
            seen.add(model)
            lo, hi = ci_pair(row)
            delta = delta_mean(row)
            family = family_map.get(model, infer_family(model))
            note = note_for(row, notes.get(model, ""))
            out.append(
                {
                    "slice": str(config.get("slice", "")),
                    "source report": source,
                    "model": model,
                    "learner family": str(family),
                    "utility": fmt_float(row.get("utility_mean")),
                    "delta vs best fixed": fmt_float(delta),
                    "95% CI": f"[{fmt_float(lo)}, {fmt_float(hi)}]" if lo != "" and hi != "" else "",
                    "positive split share": fmt_float(row.get("positive_split_share"), 3),
                    "oracle gap closed": fmt_float(row.get("gap_closed_mean", row.get("oracle_gap_closed_mean")), 3),
                    "notes / failure mode": note,
                    "n reps": str(row.get("n_reps", "")),
                    "regret": fmt_float(row.get("regret_mean")),
                    "cost": fmt_float(row.get("cost_mean")),
                    "choice share": choice_share(row, view_order),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "slice",
        "source report",
        "model",
        "learner family",
        "utility",
        "delta vs best fixed",
        "95% CI",
        "positive split share",
        "oracle gap closed",
        "notes / failure mode",
        "n reps",
        "regret",
        "cost",
        "choice share",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True, help="Protocol B slice id, e.g. dense_semantic")
    parser.add_argument("--config", type=Path, required=True, help="Learner-sweep diagnostic YAML/JSON config")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    config = load_config((ROOT / args.config).resolve() if not args.config.is_absolute() else args.config)
    configured_slice = str(config.get("slice", ""))
    if configured_slice and configured_slice != args.slice:
        raise SystemExit(f"--slice {args.slice!r} does not match config slice {configured_slice!r}")

    rows = collect_rows(config, ROOT)
    write_csv((ROOT / args.out).resolve() if not args.out.is_absolute() else args.out, rows)
    print(f"wrote {args.out}")
    print("diagnostic audit only; not a model-selection leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
