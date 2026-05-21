import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot hardness modeling outputs.")
    ap.add_argument("--feature_importance", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    imp = pd.read_csv(args.feature_importance).head(8)
    pred = pd.read_csv(args.predictions)

    plt.figure(figsize=(7,4.4))
    plt.bar(range(len(imp)), imp["importance_mean"])
    plt.xticks(range(len(imp)), imp["feature"], rotation=35, ha="right")
    plt.ylabel("Permutation importance")
    plt.title("Exploratory hardness-factor importance")
    plt.tight_layout()
    plt.savefig(out / "fig_hardness_importance.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5.2,5.0))
    plt.scatter(pred["ambiguity_severity"], pred["predicted_severity"])
    mn = min(pred["ambiguity_severity"].min(), pred["predicted_severity"].min())
    mx = max(pred["ambiguity_severity"].max(), pred["predicted_severity"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Observed severity")
    plt.ylabel("Predicted severity (CV)")
    plt.title("Observed vs predicted ambiguity severity")
    plt.tight_layout()
    plt.savefig(out / "fig_hardness_pred_vs_obs.png", dpi=220)
    plt.close()

if __name__ == "__main__":
    main()
