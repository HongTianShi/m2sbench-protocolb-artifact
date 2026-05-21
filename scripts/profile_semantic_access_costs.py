"""Profile semantic summary/PQ/full access costs for the IVF-PQ audit.

The benchmark scorer uses versioned operation-unit costs so that leaderboard
scores are platform independent. This script provides a local calibration
anchor: it times operations with the candidate-list sizes observed in the
large IVF-PQ audit and reports approximate bytes touched by each view.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "summaries" / "ivf_pq_rerank_large_adapter" / "large_ivf_pq_rerank_per_query_sample.csv"


def _median_us(fn, reps: int) -> float:
    values: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1_000_000.0)
    return float(np.median(values))


def _candidate_counts(path: Path, max_rows: int) -> np.ndarray:
    counts: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts.append(int(float(row["candidate_count"])))
            if len(counts) >= max_rows:
                break
    if not counts:
        raise ValueError(f"No candidate_count rows found in {path}")
    return np.asarray(counts, dtype=np.int32)


def profile(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    counts = _candidate_counts(args.sample_csv, args.max_rows)
    sampled_counts = rng.choice(counts, size=args.reps, replace=True)
    max_n = int(sampled_counts.max())
    dim = args.dim
    subvectors = args.subvectors
    codewords = args.codewords
    subdim = dim // subvectors
    if dim % subvectors != 0:
        raise ValueError("dim must be divisible by subvectors")

    query = rng.normal(size=dim).astype(np.float32)
    centroid = rng.normal(size=dim).astype(np.float32)
    full = rng.normal(size=(max_n, dim)).astype(np.float32)
    codes = rng.integers(0, codewords, size=(max_n, subvectors), dtype=np.uint8)
    codebooks = rng.normal(size=(subvectors, codewords, subdim)).astype(np.float32)
    q_sub = query.reshape(subvectors, subdim)
    lut = np.einsum("sd,scd->sc", q_sub, codebooks, optimize=True).astype(np.float32)

    index = {"i": 0}

    def _next_n() -> int:
        i = index["i"]
        index["i"] = (i + 1) % len(sampled_counts)
        return int(sampled_counts[i])

    def summary_score() -> None:
        _ = float(np.dot(query, centroid))

    def pq_score() -> None:
        n = _next_n()
        local_codes = codes[:n]
        scores = np.zeros(n, dtype=np.float32)
        for s in range(subvectors):
            scores += lut[s, local_codes[:, s]]
        _ = float(scores[: args.topk].sum())

    def full_score() -> None:
        n = _next_n()
        scores = full[:n] @ query
        _ = float(scores[: args.topk].sum())

    summary_us = _median_us(summary_score, args.reps)
    pq_us = _median_us(pq_score, args.reps)
    full_us = _median_us(full_score, args.reps)
    median_n = float(np.median(counts))
    p90_n = float(np.percentile(counts, 90))
    summary_bytes = dim * 4
    pq_bytes_median = int(median_n * subvectors + subvectors * codewords * subdim * 4)
    full_bytes_median = int(median_n * dim * 4)
    rows = [
        {
            "view": "summary_centroid",
            "median_us": summary_us,
            "ratio_to_pq": summary_us / max(pq_us, 1e-12),
            "approx_bytes_at_median_n": summary_bytes,
            "benchmark_cost": 0.0,
        },
        {
            "view": "pq_codes",
            "median_us": pq_us,
            "ratio_to_pq": 1.0,
            "approx_bytes_at_median_n": pq_bytes_median,
            "benchmark_cost": 0.20,
        },
        {
            "view": "full_vectors",
            "median_us": full_us,
            "ratio_to_pq": full_us / max(pq_us, 1e-12),
            "approx_bytes_at_median_n": full_bytes_median,
            "benchmark_cost": 0.58,
        },
    ]
    return {
        "parameters": {
            "sample_csv": str(args.sample_csv),
            "max_rows": args.max_rows,
            "reps": args.reps,
            "dim": dim,
            "subvectors": subvectors,
            "codewords": codewords,
            "topk": args.topk,
            "seed": args.seed,
        },
        "candidate_count": {
            "median": median_n,
            "p90": p90_n,
            "max_sampled_for_profile": max_n,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--max-rows", type=int, default=25_000)
    parser.add_argument("--reps", type=int, default=2_000)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--subvectors", type=int, default=12)
    parser.add_argument("--codewords", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = profile(args)
    json_path = args.out_dir / "semantic_access_cost_profile.json"
    md_path = args.out_dir / "semantic_access_cost_profile.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    rows = result["rows"]
    lines = [
        "# Semantic Access Cost Profile",
        "",
        "Local calibration for summary/PQ/full evidence operations using candidate-list sizes from the large IVF-PQ audit.",
        "",
        "| view | median us | ratio to PQ | approx bytes at median n | benchmark cost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {view} | {median_us:.3f} | {ratio_to_pq:.3f} | {approx_bytes_at_median_n} | {benchmark_cost:.2f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(
        "Candidate counts: median {median:.0f}, p90 {p90:.0f}.".format(**result["candidate_count"])
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for row in rows:
        print(
            f"{row['view']}: {row['median_us']:.3f} us, "
            f"bytes {row['approx_bytes_at_median_n']}, cost {row['benchmark_cost']:.2f}"
        )


if __name__ == "__main__":
    main()
