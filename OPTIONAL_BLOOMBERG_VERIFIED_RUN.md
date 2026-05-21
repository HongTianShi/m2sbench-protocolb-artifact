# Optional Bloomberg Adapter Verified Run

This private/local smoke run verifies that the optional Bloomberg adapter can
complete the full Protocol-B loop on a licensed Bloomberg Terminal machine. The
returned Bloomberg data are not part of the public reproduction path and should
not be redistributed unless the local license permits it.

## Verified Path

```text
Bloomberg Terminal + Desktop API
        |
Python blpapi adapter
        |
licensed local export zip
        |
scripts/import_bloomberg_optional_export.py
        |
data/bloomberg_optional_demo_cells.jsonl
        |
scripts/run_demo.py
        |
scripts/evaluate_submission.py
        |
outputs/bloomberg_optional_leaderboard.md
```

## Aggregate Audit

- Adapter status: `ok`
- API errors recorded by adapter: `0`
- Rows exported: `4726`
- Securities with rows: `8`
- Field: `PX_LAST`
- Date span: `2024-01-02` to `2026-05-07`
- Imported Protocol-B cells: `50`
- Leakage guard: passed; evaluator-only fields were removed before solver execution
- Submission validation: passed
- Evaluator output: valid rate `1.000`, feasible rate `1.000`
- Returned export checksum: recorded locally but not redistributed with the public artifact

The optional run demonstrates operational adapter readiness. It is intentionally
separate from the frozen public benchmark cells, because Bloomberg access and
returned data are license-controlled.
