from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import sys
import traceback
import zipfile
from typing import Any


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _date_yyyymmdd(value: str) -> str:
    if not value:
        return dt.date.today().strftime("%Y%m%d")
    return dt.date.fromisoformat(value).strftime("%Y%m%d")


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _open_session(blpapi: Any, host: str, port: int) -> Any:
    options = blpapi.SessionOptions()
    options.setServerHost(host)
    options.setServerPort(int(port))
    session = blpapi.Session(options)
    if not session.start():
        raise RuntimeError("Could not start Bloomberg API session. Is Bloomberg Terminal running and logged in?")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Could not open //blp/refdata service.")
    return session


def _field_value(field_data: Any, field: str) -> str:
    if not field_data.hasElement(field):
        return ""
    element = field_data.getElement(field)
    if element.isNull():
        return ""
    return _value_to_text(element.getValue())


def _collect_bdh(spec: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    import blpapi  # type: ignore

    session = _open_session(blpapi, spec.get("host", "localhost"), int(spec.get("port", 8194)))
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    try:
        service = session.getService("//blp/refdata")
        request = service.createRequest("HistoricalDataRequest")
        for security in spec["securities"]:
            request.append("securities", security)
        for field in spec["fields"]:
            request.append("fields", field)
        request.set("startDate", _date_yyyymmdd(spec.get("start_date", "")))
        request.set("endDate", _date_yyyymmdd(spec.get("end_date", "")))
        request.set("periodicitySelection", spec.get("periodicity", "DAILY"))
        session.sendRequest(request)
        while True:
            event = session.nextEvent(5000)
            for message in event:
                if not message.hasElement("securityData"):
                    continue
                security_data = message.getElement("securityData")
                security = security_data.getElementAsString("security")
                if security_data.hasElement("securityError"):
                    errors.append({"security": security, "error": str(security_data.getElement("securityError"))})
                    continue
                if security_data.hasElement("fieldExceptions"):
                    exceptions = security_data.getElement("fieldExceptions")
                    for idx in range(exceptions.numValues()):
                        errors.append({"security": security, "field_exception": str(exceptions.getValueAsElement(idx))})
                field_data_array = security_data.getElement("fieldData")
                for idx in range(field_data_array.numValues()):
                    field_data = field_data_array.getValueAsElement(idx)
                    date_value = field_data.getElement("date").getValue()
                    for field in spec["fields"]:
                        rows.append(
                            {
                                "security": security,
                                "date": _value_to_text(date_value),
                                "field": field,
                                "value": _field_value(field_data, field),
                            }
                        )
            if event.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()
    return rows, errors


def _write_long_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["security", "date", "field", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_wide_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = sorted({f"{row['security']}|{row['field']}" for row in rows})
    by_date: dict[str, dict[str, str]] = {}
    for row in rows:
        by_date.setdefault(row["date"], {})[f"{row['security']}|{row['field']}"] = row["value"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", *columns])
        writer.writeheader()
        for date in sorted(by_date):
            writer.writerow({"date": date, **by_date[date]})


def _zip_dir(src: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent).as_posix())


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp_local": dt.datetime.now().isoformat(timespec="seconds"),
    }


def run(spec_path: Path, out_dir: Path, dry_run: bool = False) -> Path:
    spec = _read_json(spec_path)
    run_id = f"bloomberg_optional_adapter_{_now_stamp()}"
    export_dir = out_dir / run_id
    raw_dir = export_dir / "raw"
    logs_dir = export_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _write_json(export_dir / "request_spec_used.json", spec)
    audit: dict[str, Any] = {
        "run_id": run_id,
        "status": "started",
        "environment": _environment(),
        "dry_run": dry_run,
    }
    rows: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    try:
        if dry_run:
            audit["status"] = "dry_run"
        else:
            rows, errors = _collect_bdh(spec)
            _write_long_csv(raw_dir / "bdh_daily_long.csv", rows)
            _write_wide_csv(raw_dir / "bdh_daily_wide.csv", rows)
            audit["status"] = "ok" if rows else "no_rows"
    except ModuleNotFoundError as exc:
        audit["status"] = "missing_blpapi"
        audit["error"] = str(exc)
        audit["action"] = "Install or expose Bloomberg Python blpapi on this Terminal machine, then rerun."
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = str(exc)
        audit["traceback"] = traceback.format_exc()
    audit["n_long_rows"] = len(rows)
    audit["n_errors"] = len(errors)
    audit["errors"] = errors[:50]
    _write_json(export_dir / "adapter_audit.json", audit)
    file_hashes = {}
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            file_hashes[path.relative_to(export_dir).as_posix()] = _sha256(path)
    manifest = {
        "run_id": run_id,
        "adapter_name": spec.get("adapter_name", "m2sbench_optional_bloomberg_protocol_b_adapter"),
        "status": audit["status"],
        "licensing_note": "Returned Bloomberg data remains subject to the license and entitlements of the account that ran this adapter.",
        "request": {
            "start_date": spec.get("start_date"),
            "end_date": spec.get("end_date") or dt.date.today().isoformat(),
            "periodicity": spec.get("periodicity", "DAILY"),
            "n_securities": len(spec.get("securities", [])),
            "fields": spec.get("fields", []),
        },
        "files": file_hashes,
    }
    _write_json(export_dir / "adapter_manifest.json", manifest)
    (export_dir / "RETURN_THIS_EXPORT.txt").write_text(
        "Copy this entire zip back to the M2S-Bench machine and run:\n"
        "python scripts/import_bloomberg_optional_export.py <this_zip>\n",
        encoding="utf-8",
    )
    zip_path = out_dir / f"{run_id}.zip"
    _zip_dir(export_dir, zip_path)
    print(f"Wrote export: {zip_path}")
    print(f"Status: {audit['status']}; rows: {len(rows)}; errors: {len(errors)}")
    if audit["status"] not in {"ok", "dry_run"}:
        print("See adapter_audit.json inside the export zip for details.")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect optional Bloomberg Terminal data for M2S-Bench.")
    parser.add_argument("--spec", type=Path, default=Path("request_spec.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("exports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.spec, args.out_dir, args.dry_run)


if __name__ == "__main__":
    main()
