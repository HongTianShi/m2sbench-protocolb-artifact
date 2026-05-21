# Reproduction Guide

This artifact separates static inspection from reproduction. The heavy paper
audits are already frozen under `reports/`; reviewers can run the lightweight
checks below without downloading external corpora or using a GPU.

## 1. Quick Check / Dependency And Material Preflight

Checks that lightweight Python dependencies and all smoke-reproduction materials
are present. With `--install-missing`, the script installs missing lightweight
packages from `requirements.txt` only.

```bash
python scripts/check_reproduction_prereqs.py --install-missing
```

Expected final line:

```text
REPRODUCTION PREFLIGHT PASSED
```

Optional systems/profile dependencies are listed separately in
`requirements-systems.txt` and `environment-systems.yml`; they are not required
for smoke reproduction.

## 2. Lightweight Reproduction

Runs the paper-facing public-dev dense, 2Wiki structured, 2Wiki hyperlink, and
cost-profile smoke checks. These commands validate route JSONL files, score
against evaluator-held public-dev references, and write compact smoke reports.
They do not recompute the heavy BEIR/2Wiki/CE experiments.

```bash
python scripts/run_smoke_reproduction.py
```

Expected final lines:

```text
SMOKE REPRODUCTION PASSED
Generated:
  <printed-output-dir>/dense_smoke.json
  <printed-output-dir>/2wiki_smoke.json
  <printed-output-dir>/cost_smoke.json
```

The script also writes per-component scorer details under
`<printed-output-dir>/details/` and records those paths in
`<printed-output-dir>/smoke_reproduction_summary.json`. By default, this output
directory is under the system temp directory rather than the release tree. To
choose a specific reviewer-run directory, pass:

```bash
python scripts/run_smoke_reproduction.py --out-dir ../m2sbench_smoke_tmp
```

## 3. Full Reproduction

Full reproduction reruns expensive audits such as dense semantic access, 2Wiki
7GB hyperlink stress, CE purchase, compression ladders, and systems profiling.
It may require GPU packages, public dataset caches, and long local runtime.
Use the frozen reports for review unless you intentionally want to rebuild.

Representative commands are listed in `README.md` under "Paper Audit Records".
The full systems environment is described in `environment-systems.yml` and
`requirements-systems.txt`.

## 4. Custom Partial Reproduction

For a guided subset, run:

```bash
python scripts/run_custom_reproduction.py
```

The script asks whether to run dense, 2Wiki, and cost-profile smoke components.
It only calls the same lightweight smoke components described above.

## Cleanup

Smoke outputs are disposable. Remove only the directory you passed as
`--out-dir`, or the printed default temp directory. For example:

```bash
rm -rf ../m2sbench_smoke_tmp
```

Do not remove `reports/` itself, `dist/m2sbench_anonymous_release`, or
`dist/m2sbench_anonymous_release.zip`; those are the paper-facing release
materials.
