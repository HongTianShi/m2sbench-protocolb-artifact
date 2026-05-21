# Returning Data To M2S-Bench

After the Bloomberg machine creates `exports/bloomberg_optional_adapter_*.zip`,
copy that zip back to the M2S-Bench machine and run:

```powershell
cd C:\path\to\m2sbench\clean_bundle_work
python scripts\import_bloomberg_optional_export.py C:\path\to\bloomberg_optional_adapter_YYYYMMDD_HHMMSS.zip
python scripts\run_demo.py --cells data\bloomberg_optional_demo_cells.jsonl --visibility data\bloomberg_optional_visibility_manifest.json --solver solvers\template_solver.py --out outputs\bloomberg_optional_submission.jsonl
python scripts\evaluate_submission.py outputs\bloomberg_optional_submission.jsonl --challenge data\bloomberg_optional_demo_cells.jsonl
```

The optional Bloomberg export is for local adapter validation only. It is not
needed to reproduce the paper's main benchmark tables.
