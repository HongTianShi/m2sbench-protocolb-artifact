"""Write a compact dimensionality inventory for public stress slices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import signal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "summaries" / "dimension_stress_audit"


def urban_frame_count(target_sr: int, max_seconds: float, n_fft: int, hop_length: int) -> int:
    n = int(target_sr * max_seconds)
    _, _, z = signal.stft(
        np.zeros(n, dtype=np.float32),
        fs=target_sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=True,
    )
    return int(z.shape[1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fashion_manifest = (
        ROOT
        / "budgeted_runs"
        / "fashion_100h"
        / "shards"
        / "shard_00000_seed_20260501"
        / "reports"
        / "fashion_mnist_visual_manifest.json"
    )
    urban_manifest = (
        ROOT
        / "budgeted_runs"
        / "urban_audio_20h"
        / "shards"
        / "shard_00000_seed_20260601"
        / "reports"
        / "urban_sound_timefreq_manifest.json"
    )
    fashion = json.loads(fashion_manifest.read_text(encoding="utf-8"))["config"]
    urban = json.loads(urban_manifest.read_text(encoding="utf-8"))["config"]
    frames = urban_frame_count(
        int(urban["target_sr"]),
        float(urban["max_seconds"]),
        int(urban["n_fft"]),
        int(urban["hop_length"]),
    )
    rows = [
        {
            "slice": "Synthetic main",
            "matched_support": "point cloud",
            "support_dim": 2,
            "raw_or_rich_view_dim": "64x64 raster / point set",
            "summary_dim": 5,
            "notes": "visual controlled cells; extended by high-dimensional audit",
        },
        {
            "slice": "Fashion-MNIST contours",
            "matched_support": "edge-contour point cloud",
            "support_dim": 2,
            "raw_or_rich_view_dim": "28x28 raw image = 784; occupancy 8x8=64; raster 16x16=256",
            "summary_dim": 7,
            "notes": f"{fashion['n_points']} contour points; 10 public clothing classes",
        },
        {
            "slice": "UrbanSound8K time-frequency",
            "matched_support": "mel-time point cloud",
            "support_dim": 2,
            "raw_or_rich_view_dim": f"{urban['n_mels']}x{frames} log-mel grid = {int(urban['n_mels']) * frames}; coarse 8x8=64; full 16x16=256",
            "summary_dim": 8,
            "notes": f"{urban['n_points']} time-frequency points; 10 public audio classes",
        },
        {
            "slice": "High-dimensional matched-summary audit",
            "matched_support": "ambient embedding point cloud",
            "support_dim": "2,10,50,128",
            "raw_or_rich_view_dim": "same ambient point cloud; coarse and higher-order sketches",
            "summary_dim": "d + d(d+1)/2 + 1",
            "notes": "auxiliary stress audit; all cells whitened to the same mean/covariance summary",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cross_domain_dimension_inventory.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
