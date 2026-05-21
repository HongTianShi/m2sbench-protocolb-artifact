# Optional Bloomberg Adapter Demo

This package is for a Bloomberg Terminal machine only. It collects a small,
licensed local export and returns a zip file that can be imported back into the
M2S-Bench workspace.

This is an optional adapter demo, not a required artifact dependency. Reviewers
without Bloomberg access can ignore it; the main Protocol B demo uses the frozen
public cells in the paper artifact.

## One-Click Run

1. Copy this folder or the packaged zip to the Bloomberg Terminal machine.
2. Extract it to a local folder.
3. If Python is not configured yet, double-click `SETUP_PYTHON_ENV.bat`.
4. If `blpapi` is still missing, use Bloomberg Terminal `BDEV <GO>` to download
   the Bloomberg API SDK for Windows/Python, then run
   `INSTALL_LOCAL_BLPAPI_SDK.bat`.
5. Start Bloomberg Terminal and log in.
6. Double-click `RUN_BLOOMBERG_ADAPTER.bat`.
7. Copy the generated `exports/bloomberg_optional_adapter_*.zip` back to the
   M2S-Bench machine.

PowerShell alternative:

```powershell
powershell -ExecutionPolicy Bypass -File SETUP_PYTHON_ENV.ps1
powershell -ExecutionPolicy Bypass -File INSTALL_LOCAL_BLPAPI_SDK.ps1 -SdkPath C:\path\to\downloaded_bloomberg_sdk.zip
powershell -ExecutionPolicy Bypass -File RUN_BLOOMBERG_ADAPTER.ps1
```

## What It Exports

- `raw/bdh_daily_long.csv`: tidy BDH-style daily panel.
- `raw/bdh_daily_wide.csv`: wide panel for quick inspection.
- `adapter_manifest.json`: request parameters, entitlement-neutral metadata, hashes.
- `adapter_audit.json`: status, row counts, errors, and environment diagnostics.
- `request_spec_used.json`: exact request spec used for this run.

The export contains only data accessible from the Bloomberg account running the
script. Do not share the returned export unless your data license permits it.

The default `request_spec.json` is intentionally small: several liquid market
series, `PX_LAST` only, from 2024 onward. This is enough for the optional
Protocol-B adapter demo and avoids a very large Excel workbook. A broader stress
request is kept in `request_spec_full.json`, but it is expected to produce more
field/security errors because some Bloomberg fields are not meaningful for every
asset class.

## Requirements

- Bloomberg Terminal running on the same machine.
- Bloomberg Desktop API listening on `localhost:8194`.
- Python 3.
- Python package `blpapi` importable by that Python installation.

`SETUP_PYTHON_ENV.bat` creates a package-local `.venv` and installs `blpapi`.
If Python is missing and Windows Package Manager is available, it installs
64-bit Python 3.11 with `winget` first. The adapter launcher prefers this local
`.venv` over any global Python. The setup script uses Bloomberg's Python
package index, the default package index, and local Bloomberg SDK locations such
as `C:\blp\API\APIv3\Python`. The public anonymous repository does not
redistribute a Bloomberg SDK wheel.

If Python or `blpapi` is still missing, the launcher automatically falls back to
an Excel/BDH path that uses the Bloomberg Excel Add-In. The launcher also writes
`logs/adapter_console_*.log`; if no export zip is created, inspect the log
locally or share only a redacted excerpt.

The Excel fallback now runs in passive mode by default: it writes BDH formulas
but does not invoke Bloomberg Ribbon refresh macros. This avoids machines where
the Bloomberg Ribbon reports "Unable to launch this component" or thinks Excel
is in edit mode. If rows are still not available after the passive wait, the
script leaves Excel open and prompts you to press Enter after you have manually
left edit mode and let the BDH formulas populate.

## Quick Troubleshooting

Open PowerShell in this folder and run:

```powershell
.\SETUP_PYTHON_ENV.bat
.\.venv\Scripts\python.exe -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import blpapi; print('blpapi OK')"
python --version
python -c "import sys; print(sys.executable)"
python -c "import blpapi; print('blpapi OK')"
```

If `.venv\Scripts\python.exe` is missing, setup did not finish. If the `.venv`
`blpapi` import fails, copy back the newest `logs/python_setup_*.log`. If only
the global `python -c "import blpapi"` fails, that is fine; the launcher uses the
package-local `.venv` first.

If direct access to Bloomberg's Python package index times out, download the SDK
from Bloomberg Terminal via `BDEV <GO>` and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_LOCAL_BLPAPI_SDK.ps1 -SdkPath C:\path\to\downloaded_sdk.zip
```

The installer searches the SDK archive, Downloads, Desktop, and common
`C:\blp\API\APIv3\Python` locations for a `blpapi` wheel or setup directory.

In Excel fallback mode, five-digit values such as `42373` are usually Excel date
serials, not failed prices. The script formats the date column as `yyyy-mm-dd`
and exports normalized dates in `raw/bdh_daily_long.csv`. For sheet-level
diagnostics, check `logs/bdh_sheet_status.csv` inside the returned zip.

If every sheet stays at `#N/A Requesting Data`, first press `Esc` in Excel and
click a blank cell or the README sheet so Excel is definitely not editing a
formula. Then wait for Bloomberg to populate the BDH output. Only use Bloomberg's
manual refresh controls after leaving edit mode.

## Local Import Back Into M2S-Bench

On the M2S-Bench machine:

```powershell
cd C:\path\to\m2sbench\clean_bundle_work
python scripts\import_bloomberg_optional_export.py C:\path\to\bloomberg_optional_adapter_YYYYMMDD_HHMMSS.zip
python scripts\run_demo.py --cells data\bloomberg_optional_demo_cells.jsonl --visibility data\bloomberg_optional_visibility_manifest.json --solver solvers\template_solver.py --out outputs\bloomberg_optional_submission.jsonl
python scripts\evaluate_submission.py outputs\bloomberg_optional_submission.jsonl --challenge data\bloomberg_optional_demo_cells.jsonl
```
