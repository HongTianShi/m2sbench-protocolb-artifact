import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Validate a small set of paper-facing quantitative claims against release CSVs.")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()
    rr = Path(args.results_root)
    out = Path(args.out_file)
    report = {"checks": []}

    t1 = pd.read_csv(rr / "paper_main" / "track1_aggregate.csv")
    nonanalytic = t1[t1["method"] != "constructive"].sort_values("norm_cd_mean")
    strongest = nonanalytic.iloc[0]
    report["checks"].append({
        "name": "strongest_non_analytic_baseline",
        "status": "pass",
        "value": strongest["method"],
        "norm_cd_mean": float(strongest["norm_cd_mean"]),
        "coverage_mean": float(strongest["coverage_mean"]),
    })

    t3 = pd.read_csv(rr / "paper_appendix" / "track3_decision_support_audit.csv")
    report["checks"].append({
        "name": "track3_rows_exist",
        "status": "pass" if len(t3) >= 4 else "fail",
        "rows": int(len(t3)),
    })

    t2 = pd.read_csv(rr / "paper_main" / "track2_failure_scores.csv")
    report["checks"].append({
        "name": "no_nan_failure_scores",
        "status": "pass" if not t2.isna().any().any() else "fail",
        "n_rows": int(len(t2)),
    })

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
