# Static-Overfit Audit

This folder records the anti-overfitting checks for the public Protocol B synthetic leaderboard. The released 320 cells are treated as a public development/review slice. The same generator and evaluator can instantiate private-seed cells by changing evaluator-held seeds while preserving the public method interface.

Files in this folder:

- `public_320_leaderboard.csv`: the public 320-cell blind baseline leaderboard.
- `public_320_bootstrap_rank_stability.csv`: bootstrap rank stability over the public 320 cells.
- `private_seed_320_runs.csv`: regenerated 320-cell private-seed audit runs under the same Protocol B contract.
- `private_seed_320_leaderboard.csv`: leaderboard summary for the regenerated private-seed 320-cell slice.
- `public_private_seed_method_delta.csv`: method-level public/private mean Norm. CD deltas.
- `reference_32_leaderboard.csv`: small same-generator reference-slice check.
- `constructor_240_leaderboard.csv`: independent constructor-slice leaderboard check.
- `static_overfit_audit_summary.csv` and `.json`: compact paper-facing audit summaries.
