from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REPORTS = ROOT / "reports"
SUMMARIES = ROOT / "summaries"
OUT_DIR = SUMMARIES / "ivf_pq_rerank_adapter"


@dataclass(slots=True)
class IvfPqConfig:
    svd_dim: int = 128
    n_clusters: int = 80
    pq_subvectors: int = 8
    pq_codewords: int = 16
    topk: int = 10
    lambda_cost: float = 0.08
    cost_summary: float = 0.0
    cost_pq: float = 0.20
    cost_full: float = 0.58
    random_state: int = 20260521
    max_iter: int = 80
    batch_size: int = 512


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return str(value)
            return f"{float(value):.3f}".rstrip("0").rstrip(".")
        text = str(value)
        return text.replace("|", "\\|")

    headers = [str(col).replace("|", "\\|") for col in frame.columns]
    rows = [[fmt(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _manifest_path(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            return path.name


def _find_dataset_file(filename: str, must_contain: str) -> Path:
    candidates = [
        p
        for p in (WORKSPACE / "文本数据集").rglob(filename)
        if must_contain.lower() in str(p).lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} containing {must_contain}")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def _load_content_dataset(name: str) -> tuple[list[str], sparse.csr_matrix, np.ndarray, Path]:
    if name == "cora":
        path = _find_dataset_file("cora.content", "cora")
    elif name == "citeseer":
        path = _find_dataset_file("citeseer.content", "citeseer")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    doc_ids = frame.iloc[:, 0].astype(str).tolist()
    labels = frame.iloc[:, -1].astype(str).to_numpy()
    features = frame.iloc[:, 1:-1].astype(np.float32).to_numpy(copy=False)
    return doc_ids, sparse.csr_matrix(features, dtype=np.float32), labels, path


def _dense_vectors(x: sparse.csr_matrix, config: IvfPqConfig) -> np.ndarray:
    n_components = min(config.svd_dim, x.shape[0] - 2, x.shape[1] - 2)
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state, n_iter=7)
    dense = svd.fit_transform(x).astype(np.float32)
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
    return dense


def _train_ivf(vectors: np.ndarray, config: IvfPqConfig) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = min(config.n_clusters, max(8, len(vectors) // 30))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        n_init=3,
        reassignment_ratio=0.01,
    )
    assignments = model.fit_predict(vectors)
    centroids = normalize(model.cluster_centers_.astype(np.float32), norm="l2", axis=1)
    return assignments.astype(np.int32), centroids.astype(np.float32)


def _train_pq_reconstruction(vectors: np.ndarray, config: IvfPqConfig) -> np.ndarray:
    n, dim = vectors.shape
    m = min(config.pq_subvectors, dim)
    while dim % m != 0 and m > 1:
        m -= 1
    subdim = dim // m
    reconstructed = np.zeros_like(vectors, dtype=np.float32)
    for part in range(m):
        start = part * subdim
        end = (part + 1) * subdim if part < m - 1 else dim
        block = vectors[:, start:end]
        k = min(config.pq_codewords, max(2, n // 32))
        codebook = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.random_state + part,
            batch_size=config.batch_size,
            max_iter=config.max_iter,
            n_init=2,
            reassignment_ratio=0.01,
        )
        codes = codebook.fit_predict(block)
        reconstructed[:, start:end] = codebook.cluster_centers_.astype(np.float32)[codes]
    return normalize(reconstructed, norm="l2", axis=1).astype(np.float32)


def _ndcg_at_k(labels: np.ndarray, query_label: str, ranking: np.ndarray, k: int) -> float:
    if ranking.size == 0:
        return 0.0
    rel = (labels[ranking[:k]] == query_label).astype(float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))
    total_relevant = int(np.sum(labels[ranking] == query_label))
    ideal_len = min(k, total_relevant)
    if ideal_len <= 0:
        return 0.0
    idcg = float(np.sum(discounts[:ideal_len]))
    return dcg / idcg if idcg > 0 else 0.0


def _cluster_stats(vectors: np.ndarray, labels: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id in range(len(centroids)):
        idx = np.where(assignments == cluster_id)[0]
        if idx.size == 0:
            continue
        sims = vectors[idx] @ centroids[cluster_id]
        counts = pd.Series(labels[idx]).value_counts(normalize=True)
        entropy = float(-(counts * np.log2(counts + 1e-12)).sum())
        rows.append(
            {
                "cluster_id": cluster_id,
                "n": int(idx.size),
                "mean_centroid_similarity": float(np.mean(sims)),
                "radius": float(np.mean(1.0 - sims)),
                "radius_std": float(np.std(1.0 - sims)),
                "label_entropy": entropy,
                "majority_share": float(counts.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _query_records(
    dataset: str,
    doc_ids: list[str],
    vectors: np.ndarray,
    pq_vectors: np.ndarray,
    labels: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    cluster_frame: pd.DataFrame,
    config: IvfPqConfig,
) -> pd.DataFrame:
    cluster_lookup = {int(row.cluster_id): row for row in cluster_frame.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for q_idx, query in enumerate(vectors):
        cluster_id = int(np.argmax(centroids @ query))
        candidates = np.where(assignments == cluster_id)[0]
        candidates = candidates[candidates != q_idx]
        if candidates.size < 2:
            continue
        centroid = centroids[cluster_id]
        summary_scores = vectors[candidates] @ centroid
        pq_scores = pq_vectors[candidates] @ query
        full_scores = vectors[candidates] @ query
        summary_rank = candidates[np.argsort(-summary_scores)]
        pq_rank = candidates[np.argsort(-pq_scores)]
        full_rank = candidates[np.argsort(-full_scores)]
        ndcg_summary = _ndcg_at_k(labels, labels[q_idx], summary_rank, config.topk)
        ndcg_pq = _ndcg_at_k(labels, labels[q_idx], pq_rank, config.topk)
        ndcg_full = _ndcg_at_k(labels, labels[q_idx], full_rank, config.topk)
        utilities = {
            "summary": ndcg_summary - config.lambda_cost * config.cost_summary,
            "pq": ndcg_pq - config.lambda_cost * config.cost_pq,
            "full": ndcg_full - config.lambda_cost * config.cost_full,
        }
        best_view = max(utilities, key=utilities.get)
        stats = cluster_lookup[cluster_id]
        query_centroid_distance = float(1.0 - np.dot(query, centroid))
        rows.append(
            {
                "dataset": dataset,
                "doc_id": doc_ids[q_idx],
                "query_index": int(q_idx),
                "label": labels[q_idx],
                "cluster_id": cluster_id,
                "candidate_count": int(candidates.size),
                "cluster_n": int(stats.n),
                "cluster_radius": float(stats.radius),
                "cluster_radius_std": float(stats.radius_std),
                "cluster_label_entropy": float(stats.label_entropy),
                "cluster_majority_share": float(stats.majority_share),
                "query_centroid_distance": query_centroid_distance,
                "summary_ndcg": ndcg_summary,
                "pq_ndcg": ndcg_pq,
                "full_ndcg": ndcg_full,
                "summary_utility": utilities["summary"],
                "pq_utility": utilities["pq"],
                "full_utility": utilities["full"],
                "full_gain_over_summary": ndcg_full - ndcg_summary,
                "pq_gain_over_summary": ndcg_pq - ndcg_summary,
                "full_minus_pq_ndcg": ndcg_full - ndcg_pq,
                "best_view": best_view,
                "best_utility": utilities[best_view],
            }
        )
    return pd.DataFrame(rows)


def _fixed_policy_summary(frame: pd.DataFrame, view: str) -> dict[str, Any]:
    return {
        "policy": f"fixed_{view}",
        "utility": float(frame[f"{view}_utility"].mean()),
        "ndcg": float(frame[f"{view}_ndcg"].mean()),
        "avg_cost": {"summary": 0.0, "pq": 0.20, "full": 0.58}[view],
        "best_view_acc": float((frame["best_view"] == view).mean()),
    }


def _proxy_router(frame: pd.DataFrame, config: IvfPqConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "candidate_count",
        "cluster_n",
        "cluster_radius",
        "cluster_radius_std",
        "cluster_majority_share",
        "query_centroid_distance",
    ]
    targets = ["summary_utility", "pq_utility", "full_utility"]
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    groups = frame["dataset"].astype(str) + ":" + frame["cluster_id"].astype(str)
    splitter = GroupKFold(n_splits=5)
    x = frame[features]
    y = frame[targets]
    costs = {"summary": config.cost_summary, "pq": config.cost_pq, "full": config.cost_full}
    views = np.array(["summary", "pq", "full"])
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, groups=groups), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=240,
                    max_depth=8,
                    min_samples_leaf=8,
                    random_state=config.random_state + fold,
                    n_jobs=1,
                )
            ),
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(x.iloc[test_idx]), dtype=float)
        choices = views[np.argmax(pred, axis=1)]
        test = frame.iloc[test_idx].reset_index(drop=True)
        chosen_util = np.array([test.loc[i, f"{choice}_utility"] for i, choice in enumerate(choices)], dtype=float)
        chosen_ndcg = np.array([test.loc[i, f"{choice}_ndcg"] for i, choice in enumerate(choices)], dtype=float)
        chosen_cost = np.array([costs[choice] for choice in choices], dtype=float)
        best = test["best_utility"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(test)),
                "utility": float(np.mean(chosen_util)),
                "ndcg": float(np.mean(chosen_ndcg)),
                "avg_cost": float(np.mean(chosen_cost)),
                "regret": float(np.mean(best - chosen_util)),
                "best_view_acc": float(np.mean(test["best_view"].to_numpy() == choices)),
                "non_full_share": float(np.mean(choices != "full")),
            }
        )
    folds = pd.DataFrame(fold_rows)
    rows.append(
        {
            "policy": "proxy_router_rf",
            "utility": float(folds["utility"].mean()),
            "utility_sd": float(folds["utility"].std(ddof=1)),
            "ndcg": float(folds["ndcg"].mean()),
            "avg_cost": float(folds["avg_cost"].mean()),
            "regret": float(folds["regret"].mean()),
            "best_view_acc": float(folds["best_view_acc"].mean()),
            "non_full_share": float(folds["non_full_share"].mean()),
        }
    )
    return pd.DataFrame(rows), folds


def _summaries(query_frame: pd.DataFrame, config: IvfPqConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for name, group in [("all", query_frame), *query_frame.groupby("dataset")]:
        row: dict[str, Any] = {
            "scope": name,
            "queries": int(len(group)),
            "clusters": int(group[["dataset", "cluster_id"]].drop_duplicates().shape[0]),
            "summary_ndcg": float(group["summary_ndcg"].mean()),
            "pq_ndcg": float(group["pq_ndcg"].mean()),
            "full_ndcg": float(group["full_ndcg"].mean()),
            "summary_utility": float(group["summary_utility"].mean()),
            "pq_utility": float(group["pq_utility"].mean()),
            "full_utility": float(group["full_utility"].mean()),
            "full_gain": float(group["full_gain_over_summary"].mean()),
            "pq_gain": float(group["pq_gain_over_summary"].mean()),
            "full_dominated_by_pq_share": float((group["full_minus_pq_ndcg"] < 0).mean()),
            "best_not_full_share": float((group["best_view"] != "full").mean()),
            "best_summary_share": float((group["best_view"] == "summary").mean()),
            "best_pq_share": float((group["best_view"] == "pq").mean()),
            "best_full_share": float((group["best_view"] == "full").mean()),
        }
        rows.append(row)
    aggregate = pd.DataFrame(rows)

    q30 = query_frame["cluster_radius"].quantile(0.30)
    q70 = query_frame["cluster_radius"].quantile(0.70)
    slices = [
        ("compact_30pct", query_frame[query_frame["cluster_radius"] <= q30]),
        ("diffuse_30pct", query_frame[query_frame["cluster_radius"] >= q70]),
    ]
    slice_rows: list[dict[str, Any]] = []
    for name, group in slices:
        slice_rows.append(
            {
                "slice": name,
                "queries": int(len(group)),
                "cluster_radius_mean": float(group["cluster_radius"].mean()),
                "summary_ndcg": float(group["summary_ndcg"].mean()),
                "pq_ndcg": float(group["pq_ndcg"].mean()),
                "full_ndcg": float(group["full_ndcg"].mean()),
                "full_gain": float(group["full_gain_over_summary"].mean()),
                "best_not_full_share": float((group["best_view"] != "full").mean()),
                "near_zero_full_gain_share": float((group["full_gain_over_summary"] <= 0.01).mean()),
            }
        )
    slices_frame = pd.DataFrame(slice_rows)
    return aggregate, slices_frame


def run(config: IvfPqConfig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    query_frames: list[pd.DataFrame] = []
    dataset_manifest: list[dict[str, Any]] = []
    for dataset in ["cora", "citeseer"]:
        doc_ids, x, labels, path = _load_content_dataset(dataset)
        vectors = _dense_vectors(x, config)
        assignments, centroids = _train_ivf(vectors, config)
        pq_vectors = _train_pq_reconstruction(vectors, config)
        cluster_frame = _cluster_stats(vectors, labels, assignments, centroids)
        cluster_frame.insert(0, "dataset", dataset)
        cluster_frame.to_csv(OUT_DIR / f"{dataset}_ivf_cluster_stats.csv", index=False)
        query_frame = _query_records(
            dataset, doc_ids, vectors, pq_vectors, labels, assignments, centroids, cluster_frame, config
        )
        query_frames.append(query_frame)
        dataset_manifest.append(
            {
                "dataset": dataset,
                "source": _manifest_path(path, WORKSPACE),
                "documents": int(len(doc_ids)),
                "input_features": int(x.shape[1]),
                "embedding_dim": int(vectors.shape[1]),
                "ivf_cells": int(cluster_frame["cluster_id"].nunique()),
                "mean_cell_size": float(cluster_frame["n"].mean()),
            }
        )
    all_queries = pd.concat(query_frames, ignore_index=True)
    all_queries.to_csv(OUT_DIR / "ivf_pq_rerank_per_query.csv", index=False)
    aggregate, slices = _summaries(all_queries, config)
    aggregate.to_csv(OUT_DIR / "ivf_pq_rerank_summary.csv", index=False)
    slices.to_csv(OUT_DIR / "ivf_pq_rerank_sufficiency_slices.csv", index=False)
    proxy_summary, proxy_folds = _proxy_router(all_queries, config)
    fixed_rows = pd.DataFrame([_fixed_policy_summary(all_queries, view) for view in ["summary", "pq", "full"]])
    policy = pd.concat([fixed_rows, proxy_summary], ignore_index=True, sort=False)
    policy.to_csv(OUT_DIR / "ivf_pq_rerank_policy.csv", index=False)
    proxy_folds.to_csv(OUT_DIR / "ivf_pq_rerank_proxy_folds.csv", index=False)
    manifest = {
        "config": asdict(config),
        "datasets": dataset_manifest,
        "mapping": {
            "cell": "IVF coarse cluster",
            "summary": "cluster centroid, within-cell dispersion, and cell size",
            "pq_view": "product-quantized subvector reconstruction used for approximate reranking",
            "full_view": "full dense vector reranking within the selected IVF cell",
            "utility": f"NDCG@{config.topk} - lambda * access_cost, with lambda={config.lambda_cost}",
        },
        "outputs": {
            "summary": _manifest_path(OUT_DIR / "ivf_pq_rerank_summary.csv"),
            "slices": _manifest_path(OUT_DIR / "ivf_pq_rerank_sufficiency_slices.csv"),
            "policy": _manifest_path(OUT_DIR / "ivf_pq_rerank_policy.csv"),
            "per_query": _manifest_path(OUT_DIR / "ivf_pq_rerank_per_query.csv"),
        },
    }
    (REPORTS / "ivf_pq_rerank_adapter_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (REPORTS / "ivf_pq_rerank_adapter_audit.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# IVF-PQ Rerank Adapter Audit\n\n")
        f.write("This adapter maps M2S-Bench cells to IVF coarse clusters over real citation-text vectors. ")
        f.write("PQ reconstruction is the compressed structural view; full dense-vector reranking is the richest view.\n\n")
        f.write("## Aggregate\n\n")
        f.write(_markdown_table(aggregate))
        f.write("\n\n## Sufficiency slices\n\n")
        f.write(_markdown_table(slices))
        f.write("\n\n## Routing policies\n\n")
        f.write(_markdown_table(policy))
        f.write("\n")
    print(aggregate.to_string(index=False))
    print()
    print(slices.to_string(index=False))
    print()
    print(policy.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight IVF-PQ rerank adapter audit on real citation-text vectors.")
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=80)
    parser.add_argument("--pq-subvectors", type=int, default=8)
    parser.add_argument("--pq-codewords", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    args = parser.parse_args()
    config = IvfPqConfig(
        svd_dim=args.svd_dim,
        n_clusters=args.n_clusters,
        pq_subvectors=args.pq_subvectors,
        pq_codewords=args.pq_codewords,
        topk=args.topk,
        lambda_cost=args.lambda_cost,
    )
    run(config)


if __name__ == "__main__":
    main()
