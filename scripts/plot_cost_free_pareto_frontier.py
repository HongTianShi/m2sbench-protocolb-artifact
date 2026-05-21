"""Render the pre-cost utility-dominance figure used in the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "summaries" / "cost_model_audit" / "cost_agnostic_pareto_frontier.csv"
OUT_PATH = ROOT / "figures" / "cost_free_pareto_frontier.pdf"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    plot = df[
        (df["domain"] == "synthetic_balanced_accuracy")
        & (df["slice"] == "synthetic_heldout")
    ].copy()
    order = ["moments_only", "occupancy_grid", "raster", "point_cloud"]
    plot["view"] = pd.Categorical(plot["view"], categories=order, ordered=True)
    plot = plot.sort_values("view")

    labels = {"moments_only": "M", "occupancy_grid": "O", "raster": "R", "point_cloud": "P"}
    colors = {
        "moments_only": "#4E79A7",
        "occupancy_grid": "#59A14F",
        "raster": "#9C755F",
        "point_cloud": "#F28E2B",
    }
    markers = {"moments_only": "o", "occupancy_grid": "s", "raster": "^", "point_cloud": "D"}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.25))

    occ = plot[plot["view"] == "occupancy_grid"].iloc[0]
    ax.add_patch(
        Rectangle(
            (occ["cost"], 0.50),
            0.62 - occ["cost"],
            occ["benefit"] - 0.50,
            facecolor="#F6D6D3",
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.text(0.435, 0.565, "dominated\nby O", ha="center", va="center", color="#9B3A34", fontsize=7)

    frontier = plot[plot["pareto_status"] == "frontier"]
    ax.plot(frontier["cost"], frontier["benefit"], color="#2F5D8C", lw=1.2, zorder=2)
    for _, row in plot.iterrows():
        view = row["view"]
        status = row["pareto_status"]
        ax.scatter(
            row["cost"],
            row["benefit"],
            s=44 if status == "frontier" else 38,
            marker=markers[view],
            facecolor=colors[view],
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        dy = 0.018 if view in ["moments_only", "occupancy_grid"] else -0.026
        ax.text(row["cost"], row["benefit"] + dy, labels[view], ha="center", va="center", fontsize=7.5, weight="bold")

    for view in ["raster", "point_cloud"]:
        row = plot[plot["view"] == view].iloc[0]
        ax.annotate(
            "",
            xy=(row["cost"], row["benefit"]),
            xytext=(occ["cost"], occ["benefit"]),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#9B3A34", linestyle="--"),
            zorder=1,
        )

    ax.set_xlim(-0.03, 0.63)
    ax.set_ylim(0.50, 0.79)
    ax.set_xlabel("View granularity / relative operation units")
    ax.set_ylabel("Task utility A (before cost)")
    ax.set_xticks([0, 0.28, 0.42, 0.58])
    ax.set_xticklabels(["M\n0", "O\n28", "R\n42", "P\n58"])
    ax.grid(axis="y", color="#DDDDDD", lw=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("Utility dominance before cost adjustment", pad=4)
    ax.text(0.0, 0.505, "Synthetic held-out balanced utility", ha="left", va="bottom", fontsize=6.8, color="#555555")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
