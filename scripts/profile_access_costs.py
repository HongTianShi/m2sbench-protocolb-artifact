"""Profile local access-view latency without changing benchmark scoring.

The paper reports fixed, cross-platform relative access units for the public
leaderboard. This optional script lets a practitioner map those units to a
local machine by timing small synthetic operations that mimic summary reads,
coarse occupancy aggregation, dense raster decoding, and full point-view
materialization. The resulting profile is advisory; it is not used by the
public evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_UNITS = {
    "moments_only": 0.0,
    "occupancy_grid": 28.0,
    "raster": 42.0,
    "point_cloud": 58.0,
}


def _median_seconds(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _profile(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(args.seed)
    points = rng.normal(size=(args.n_points, args.dim)).astype(np.float32)
    raster = rng.normal(size=(args.grid_size, args.grid_size)).astype(np.float32)
    bins = np.linspace(-3.0, 3.0, args.grid_size + 1, dtype=np.float32)

    summary = {
        "mean": points.mean(axis=0),
        "cov": np.cov(points, rowvar=False),
        "count": args.n_points,
    }

    def moments_only() -> None:
        _ = summary["mean"][0] + summary["count"]

    def occupancy_grid() -> None:
        x = np.clip(points[:, 0], -3.0, 3.0)
        y = np.clip(points[:, 1 if args.dim > 1 else 0], -3.0, 3.0)
        ix = np.searchsorted(bins, x, side="right") - 1
        iy = np.searchsorted(bins, y, side="right") - 1
        occ = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
        np.add.at(occ, (np.clip(ix, 0, args.grid_size - 1), np.clip(iy, 0, args.grid_size - 1)), 1.0)

    def dense_raster() -> None:
        decoded = np.tanh(raster)
        _ = decoded.mean(axis=0).sum()

    def point_cloud() -> None:
        centered = points - points.mean(axis=0, keepdims=True)
        if args.pairwise:
            sample = centered[: min(args.n_points, 256)]
            diff = sample[:, None, :] - sample[None, :, :]
            _ = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0)).mean()
        else:
            _ = np.sqrt(np.maximum((centered * centered).sum(axis=1), 0.0)).mean()

    operations = {
        "moments_only": moments_only,
        "occupancy_grid": occupancy_grid,
        "raster": dense_raster,
        "point_cloud": point_cloud,
    }
    medians = {name: _median_seconds(fn, args.reps) for name, fn in operations.items()}
    positive_baseline = max(medians["occupancy_grid"], 1e-12)
    rows = []
    for name in DEFAULT_UNITS:
        rows.append(
            {
                "view": name,
                "median_seconds": medians[name],
                "local_ratio_to_occupancy": medians[name] / positive_baseline,
                "benchmark_relative_units": DEFAULT_UNITS[name],
                "benchmark_normalized_cost": DEFAULT_UNITS[name] / 100.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--pairwise", action="store_true", help="include a capped pairwise point operation")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _profile(args)
    csv_path = args.out_dir / "access_cost_profile.csv"
    json_path = args.out_dir / "access_cost_profile.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"parameters": vars(args) | {"out_dir": str(args.out_dir)}, "rows": rows}, handle, indent=2)
    print(f"Wrote {csv_path} and {json_path}")
    for row in rows:
        print(
            f"{row['view']}: {row['median_seconds']:.6g}s, "
            f"local ratio {row['local_ratio_to_occupancy']:.3f}, "
            f"benchmark cost {row['benchmark_normalized_cost']:.3f}"
        )


if __name__ == "__main__":
    main()
