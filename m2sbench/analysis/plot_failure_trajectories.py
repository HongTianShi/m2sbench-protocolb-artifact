import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def representative_cells(track1_best_iterative):
    best = pd.read_csv(track1_best_iterative)
    fin = best[best["target_family"] == "financial"].sort_values("ambiguity_severity", ascending=False).iloc[0]
    non = best[best["target_family"] != "financial"].sort_values("ambiguity_severity", ascending=False).iloc[0]
    return [(fin["reference_family"], fin["target_name"], "Hardest financial cell"),
            (non["reference_family"], non["target_name"], "Hardest non-financial cell")]

def main():
    ap = argparse.ArgumentParser(description="Plot representative failure trajectories.")
    ap.add_argument("--trajectory_runs", required=True)
    ap.add_argument("--track1_best_iterative", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(args.trajectory_runs)
    reps = representative_cells(args.track1_best_iterative)
    methods = ["direct-optimization","distance-field","sinkhorn-guided","hard-projection"]
    metrics = [("norm_chamfer","Norm. CD"),("coverage","Coverage"),("topology_raw","Topology proxy raw")]
    fig, axes = plt.subplots(len(reps), len(metrics), figsize=(11.5, 7.0), sharex=False)
    for i, (budget, family, title) in enumerate(reps):
        sub = traj[(traj["budget_name"] == budget) & (traj["family"] == family)]
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            for m in methods:
                d = sub[sub["method"] == m].groupby("iteration")[metric].median().reset_index()
                if len(d):
                    ax.plot(d["iteration"], d[metric], label=m)
            ax.set_title(f"{title} — {label}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            if i == 0 and j == len(metrics)-1:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_failure_trajectories.png", dpi=220)
    plt.close()

if __name__ == "__main__":
    main()
