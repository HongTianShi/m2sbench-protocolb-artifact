# Anonymous Artifact Release Checklist

This checklist is intended for maintainers before uploading or refreshing an
anonymous review artifact.

## Contract Checks

- `schema/route_schema.json`, `schema/view_menu_schema.json`, and
  `schema/cost_menu_schema.json` parse as JSON.
- Every official slice has `menus/<slice>.view_menu.json` and
  `menus/<slice>.cost_menu.json`.
- Official view-menu rows declare `visible_fields`, `paid_fields`,
  `forbidden_before_purchase`, `parents`, and a frozen `scorer_id`.
- `hidden_evaluator_fields` and `evaluator_only_fields` are aligned in official
  view menus.
- `scripts/validate_protocol_b_route.py` accepts the valid fixtures and rejects
  the invalid leakage fixture.
- `scripts/validate_public_dev_pack.py` accepts all public-dev
  manifest/reference packs.
- The paper-facing contract points to `schema/` and `menus/`; root-level
  controlled-cell demo files are clearly marked as legacy quickstart material.
- `docs/official_benchmark_contract.md` lists official, stress, diagnostic, and
  portability status for each paper-facing slice.
- `manifests/official_slices.json` is synchronized with the contract card and
  uses the 10k 2Wiki hyperlink stress report as canonical.

## Visibility and Leakage Checks

- Public-dev evaluator files under `evaluator_only/` are marked as forbidden for
  solvers and are not described as method-visible inputs.
- Hidden-test policy states aggregate-only feedback, submission caps, and
  qrel/support/CE-score withholding.
- The release contains no local absolute paths such as personal Windows paths or
  machine-specific mount points.
- Frozen reports distinguish validity anchors, scale stress audits, mechanism
  diagnostics, and portability checks.

## Data and Provenance Checks

- External datasets are referenced by public dataset names or user-provided
  paths, not redistributed raw private data.
- Large optional corpora are documented as local inputs; report snapshots include
  only the records needed for review navigation.
- Licensed adapters do not include returned vendor data.
- `docs/provenance_and_genai.md` is present and matches the paper's GenAI
  disclosure and public-data memorization boundary.

## Packaging Checks

- `.git`, caches, notebook checkpoints, build artifacts, and local logs are not
  included in export archives.
- A development checkout or GitHub upload clone may contain `.git`; the
  anonymous web view and any downloadable archive should expose only tracked
  files, never `.git` history, remotes, caches, logs, or notebook checkpoints.
- `python scripts/build_anonymous_release.py --zip` creates a clean export under
  `dist/` with `.git`, `*_cache` directories, binary caches, bytecode, logs, and
  notebook checkpoints omitted.
- `ARTIFACT_INVENTORY.md` lists new `menus/`, schema, docs, validator, and
  report pointer files.
- `reports/PAPER_EVIDENCE_INDEX.md` maps paper claims to frozen records.
- `python scripts/regenerate_release_metadata.py` has been run, and neither
  `ARTIFACT_INVENTORY.md` nor `reports/REPORT_MANIFEST_SHA256.md` contains a
  literal `$rel` placeholder.
- `python scripts/check_reproduction_prereqs.py --install-missing` prints
  `REPRODUCTION PREFLIGHT PASSED`.
- `python scripts/run_smoke_reproduction.py` prints `SMOKE REPRODUCTION PASSED`
  and generates dense, 2Wiki, and cost smoke JSON files under the printed
  system-temp output directory unless `--out-dir` is supplied.
- `python scripts/verify_systems_profile.py` prints
  `SYSTEMS PROFILE VERIFICATION PASSED`.
