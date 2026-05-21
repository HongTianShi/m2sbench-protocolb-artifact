import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fig_track1(results_main, results_appendix, out_dir):
    df = pd.read_csv(results_main / "track1_ambiguity_severity.csv")
    families = sorted(df["target_name"].unique())
    budgets = ["dinosaur-budget","isotropic","corr-ellipse","anisotropic"]
    arr = np.zeros((len(families), len(budgets)))
    for i, f in enumerate(families):
        for j, b in enumerate(budgets):
            sub = df[(df["target_name"] == f) & (df["reference_family"] == b)]
            if len(sub):
                arr[i, j] = sub.iloc[0]["ambiguity_severity"]
    plt.figure(figsize=(7.2, 4.8))
    plt.imshow(arr, aspect="auto")
    plt.xticks(range(len(budgets)), budgets, rotation=20)
    plt.yticks(range(len(families)), families)
    plt.colorbar(label="Ambiguity severity")
    plt.tight_layout()
    out = out_dir / "figure_track1_heatmap.png"
    plt.savefig(out, dpi=200); plt.close()
    return out, [str(results_main / "track1_ambiguity_severity.csv")]

def fig_track2(results_main, results_appendix, out_dir):
    df = pd.read_csv(results_main / "track2_failure_scores.csv")
    methods = ["direct-optimization","distance-field","sinkhorn-guided","hard-projection"]
    budgets = ["dinosaur-budget","isotropic","corr-ellipse","anisotropic"]
    code = {"locking":0,"collapse":1,"topology":2}
    arr = np.zeros((len(methods), len(budgets)))
    labels = [["" for _ in budgets] for _ in methods]
    for i, m in enumerate(methods):
        for j, b in enumerate(budgets):
            sub = df[(df["method"] == m) & (df["budget_name"] == b)]
            if len(sub):
                dom = str(sub.iloc[0]["dominant_failure"])
                arr[i, j] = code.get(dom, np.nan)
                labels[i][j] = dom[0].upper()
    plt.figure(figsize=(7.2, 4.4))
    plt.imshow(arr, aspect="auto")
    plt.xticks(range(len(budgets)), budgets, rotation=20)
    plt.yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(len(budgets)):
            if labels[i][j]:
                plt.text(j, i, labels[i][j], ha="center", va="center")
    plt.tight_layout()
    out = out_dir / "figure_track2_phase.png"
    plt.savefig(out, dpi=200); plt.close()
    return out, [str(results_main / "track2_failure_scores.csv")]

def fig_track3(results_main, results_appendix, out_dir):
    src = results_main / "track3_confused_pairs.csv"
    if not src.exists():
        src = results_appendix / "appendix_point_confused_pairs.csv"
    df = pd.read_csv(src).sort_values("sym_confusion", ascending=False).head(10)
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(range(len(df)), df["sym_confusion"])
    plt.xticks(range(len(df)), df["pair"], rotation=35, ha="right")
    plt.tight_layout()
    out = out_dir / "figure_track3_confusion.png"
    plt.savefig(out, dpi=200); plt.close()
    return out, [str(src)]

def fig_track4(results_main, results_appendix, out_dir):
    df = pd.read_csv(results_main / "runtime_clarified.csv")
    plt.figure(figsize=(7.2, 4.4))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("N")
        plt.plot(sub["N"], sub["runtime_s"], marker="o", label=method)
    plt.xlabel("N"); plt.ylabel("Runtime (s)")
    plt.legend()
    plt.tight_layout()
    out = out_dir / "figure_track4_runtime.png"
    plt.savefig(out, dpi=200); plt.close()
    return out, [str(results_main / "runtime_clarified.csv")]

def main():
    ap = argparse.ArgumentParser(description="Reproduce main or appendix figures from paper-facing CSV outputs.")
    ap.add_argument("--paper-figures", action="store_true")
    ap.add_argument("--appendix-figures", action="store_true")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoke_figure", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_main = Path(args.results_root) / "paper_main"
    results_appendix = Path(args.results_root) / "paper_appendix"
    manifest = {}
    if args.paper_figures or args.smoke_figure:
        figure_specs = [("Figure2", fig_track1), ("Figure4", fig_track2), ("Figure7", fig_track3), ("Figure10", fig_track4)]
        if args.smoke_figure:
            figure_specs = [("Figure2", fig_track1)]
        for key, fn in figure_specs:
            out, sources = fn(results_main, results_appendix, out_dir)
            manifest[key] = {"output": str(out), "sources": sources, "from_raw_run_regeneration": True}
    if args.appendix_figures:
        # include the two new analysis figures if present
        for name in ["fig_hardness_importance.png", "fig_hardness_pred_vs_obs.png", "fig_failure_trajectories.png"]:
            src = Path("figures") / name
            if src.exists():
                manifest[f"Appendix:{name}"] = {"output": str(src), "sources": [], "from_raw_run_regeneration": True}
    (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
