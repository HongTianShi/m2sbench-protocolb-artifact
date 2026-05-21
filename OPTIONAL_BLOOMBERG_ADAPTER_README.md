# Optional Bloomberg Adapter Demo

This optional adapter lets a Bloomberg Terminal machine create a licensed local
export, then lets the M2S-Bench workspace import that export into the same
Protocol B pipeline used by the public demo.

The adapter is intentionally optional because Bloomberg access is not universal,
returned data may be license-restricted, and the paper's core benchmark should not
depend on a proprietary terminal.

## Package Location

The Bloomberg-machine adapter lives in:

```text
optional_adapters/bloomberg_protocol_b_adapter/
```

For the audited engineering package, the adapter is kept as an expanded folder
rather than as a nested zip so every shipped file remains visible in the package
inventory.

## Bloomberg Machine

1. Copy the `optional_adapters/bloomberg_protocol_b_adapter/` folder to the Bloomberg Terminal machine.
2. Double-click `SETUP_PYTHON_ENV.bat` if Python/`blpapi` is not ready.
3. Start Bloomberg Terminal and log in.
4. Double-click `RUN_BLOOMBERG_ADAPTER.bat`.
5. Copy the newest `exports/bloomberg_optional_adapter_*.zip` back to this machine.

The setup script creates a package-local `.venv`, installs `blpapi`, and makes
the adapter launcher prefer that environment. If Python is missing and `winget`
is available, it installs Python 3.11 first. The `blpapi` package is installed
from the bundled offline wheel before falling back to Bloomberg's Python package
index and other sources.
If the Bloomberg package index is blocked, download the Windows/Python SDK from
Bloomberg Terminal via `BDEV <GO>` and run `INSTALL_LOCAL_BLPAPI_SDK.bat` in the
adapter folder.

The no-Python Excel fallback uses the smaller default `request_spec.json`
(`PX_LAST` only, from 2024 onward). If Excel shows five-digit values in the BDH
output, those are normally Excel date serials; the returned CSV normalizes them
to `yyyy-mm-dd`. Sheet-level BDH errors are summarized in
`logs/bdh_sheet_status.csv` inside the returned zip.

The Excel fallback avoids Bloomberg Ribbon refresh macros by default. If the
workbook stays at `#N/A Requesting Data`, press `Esc` in Excel, click a blank
cell, wait for BDH output to populate or refresh manually, then return to the
console and press Enter so the script can export the loaded rows.

## M2S-Bench Machine

```powershell
cd C:\path\to\m2sbench\clean_bundle_work
python scripts\import_bloomberg_optional_export.py C:\path\to\bloomberg_optional_adapter_YYYYMMDD_HHMMSS.zip
python scripts\run_demo.py --cells data\bloomberg_optional_demo_cells.jsonl --visibility data\bloomberg_optional_visibility_manifest.json --solver solvers\template_solver.py --out outputs\bloomberg_optional_submission.jsonl
python scripts\evaluate_submission.py outputs\bloomberg_optional_submission.jsonl --challenge data\bloomberg_optional_demo_cells.jsonl
```

This produces a local optional leaderboard row without changing the paper's
frozen main results.

## Verified Local Run

One licensed local smoke run has completed the full path: Bloomberg Desktop API
export, return zip import, optional Protocol-B cell construction, solver
execution, public evaluation, and leaderboard-row generation. The run exported
4726 `PX_LAST` rows across 8 securities, imported 50 optional cells, passed the
leakage guard and submission validation, and produced aggregate leaderboard
metrics. See `OPTIONAL_BLOOMBERG_VERIFIED_RUN.md` for the non-data audit summary.
The returned export and generated optional Bloomberg cells remain local/private
unless the Bloomberg license permits redistribution.
