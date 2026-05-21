from __future__ import annotations

import argparse
import json
import math
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TEXT_ROOT = WORKSPACE / "\u6587\u672c\u6570\u636e\u96c6"
REPORTS = ROOT / "reports"
SUMMARIES = ROOT / "summaries"
OUT_DIR = SUMMARIES / "ivf_pq_rerank_large_adapter"


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
        return str(value).replace("|", "\\|")

    headers = [str(col).replace("|", "\\|") for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


@dataclass(slots=True)
class LargeIvfPqConfig:
    svd_dim: int = 192
    hash_features: int = 2**18
    target_docs: int = 180_000
    max_docred: int = 80_000
    max_douban: int = 55_000
    max_jd: int = 30_000
    max_taptap: int = 15_000
    docs_per_cell: int = 250
    min_clusters: int = 32
    max_clusters: int = 512
    pq_subvectors: int = 12
    pq_codewords: int = 32
    topk: int = 10
    lambda_cost: float = 0.08
    cost_summary: float = 0.0
    cost_pq: float = 0.20
    cost_full: float = 0.58
    random_state: int = 20260524
    kmeans_iter: int = 80
    batch_size: int = 2048
    per_query_sample: int = 25_000


def _manifest_path(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError:
            return path.name


def _find_one(pattern: str, must_contain: str | None = None) -> Path:
    candidates = [p for p in TEXT_ROOT.rglob(pattern) if p.is_file()]
    if must_contain:
        candidates = [p for p in candidates if must_contain.lower() in str(p).lower()]
    if not candidates:
        raise FileNotFoundError(f"Could not find {pattern} under {TEXT_ROOT}")
    return sorted(candidates, key=lambda p: (-p.stat().st_size, len(str(p))))[0]


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _load_docred(limit: int) -> tuple[pd.DataFrame, Path]:
    path = _find_one("train_distant.json", "DocRED")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict[str, str]] = []
    for i, item in enumerate(data[:limit]):
        sents = item.get("sents") or []
        text = " ".join(" ".join(sent) for sent in sents)
        labels = item.get("labels") or []
        if labels:
            rels = sorted({str(label.get("r", "UNK")) for label in labels})
            label = rels[0] if len(rels) == 1 else "+".join(rels[:2])
        else:
            label = "no_relation"
        rows.append({"dataset": "docred", "doc_id": f"docred:{i}", "text": text, "label": f"docred:{label}"})
    return pd.DataFrame(rows), path


def _load_douban(limit: int) -> tuple[pd.DataFrame, Path]:
    path = [p for p in TEXT_ROOT.rglob("comments.csv") if p.stat().st_size > 100_000_000][0]
    rows: list[pd.DataFrame] = []
    need = limit
    for chunk in pd.read_csv(
        path,
        encoding="gb18030",
        usecols=["COMMENT_ID", "CONTENT", "RATING"],
        chunksize=20_000,
        on_bad_lines="skip",
    ):
        chunk = chunk.dropna(subset=["CONTENT", "RATING"])
        chunk = chunk[chunk["CONTENT"].astype(str).str.len() >= 8]
        if chunk.empty:
            continue
        chunk = chunk.head(need)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "douban",
                    "doc_id": "douban:" + chunk["COMMENT_ID"].astype(str),
                    "text": _clean_text(chunk["CONTENT"]),
                    "label": "douban:rating_" + chunk["RATING"].astype(float).round().astype(int).astype(str),
                }
            )
        )
        need -= len(chunk)
        if need <= 0:
            break
    return pd.concat(rows, ignore_index=True), path


def _load_jd(limit: int) -> tuple[pd.DataFrame, Path]:
    path = _find_one("\u5546\u54c1\u8bc4\u8bba\u60c5\u611f\u9884\u6d4b.gz")
    with tarfile.open(path, "r:*") as tf:
        handle = tf.extractfile("\u8bad\u7ec3\u96c6.csv")
        if handle is None:
            raise FileNotFoundError("Missing training CSV inside JD review tarball")
        frame = pd.read_csv(handle, usecols=["\u6570\u636eID", "\u8bc4\u8bba\u6807\u9898", "\u8bc4\u8bba\u5185\u5bb9", "\u8bc4\u5206"])
    frame = frame.dropna(subset=["\u8bc4\u8bba\u5185\u5bb9", "\u8bc4\u5206"]).head(limit)
    text = _clean_text(frame["\u8bc4\u8bba\u6807\u9898"].fillna("") + " " + frame["\u8bc4\u8bba\u5185\u5bb9"].fillna(""))
    return (
        pd.DataFrame(
            {
                "dataset": "jd_reviews",
                "doc_id": "jd:" + frame["\u6570\u636eID"].astype(str),
                "text": text,
                "label": "jd:rating_" + frame["\u8bc4\u5206"].astype(float).round().astype(int).astype(str),
            }
        ),
        path,
    )


def _load_taptap(limit: int) -> tuple[pd.DataFrame, Path]:
    path = _find_one("taptap_review_ready.csv")
    frame = pd.read_csv(path, usecols=["review", "sentiment"]).dropna(subset=["review", "sentiment"]).head(limit)
    return (
        pd.DataFrame(
            {
                "dataset": "taptap",
                "doc_id": [f"taptap:{i}" for i in range(len(frame))],
                "text": _clean_text(frame["review"]),
                "label": "taptap:sent_" + frame["sentiment"].astype(int).astype(str),
            }
        ),
        path,
    )


def load_corpus(config: LargeIvfPqConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    loaders = [
        ("docred", _load_docred, config.max_docred),
        ("douban", _load_douban, config.max_douban),
        ("jd_reviews", _load_jd, config.max_jd),
        ("taptap", _load_taptap, config.max_taptap),
    ]
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for name, loader, limit in loaders:
        if limit <= 0:
            continue
        frame, source = loader(limit)
        frame = frame[frame["text"].astype(str).str.len() > 0].reset_index(drop=True)
        frames.append(frame)
        manifest.append(
            {
                "dataset": name,
                "source": _manifest_path(source, WORKSPACE),
                "documents": int(len(frame)),
                "labels": int(frame["label"].nunique()),
            }
        )
    corpus = pd.concat(frames, ignore_index=True)
    if len(corpus) > config.target_docs:
        corpus = (
            corpus.sample(n=config.target_docs, random_state=config.random_state)
            .sort_values(["dataset", "doc_id"])
            .reset_index(drop=True)
        )
    return corpus, manifest


def _build_vectors(corpus: pd.DataFrame, config: LargeIvfPqConfig) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        n_features=config.hash_features,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        dtype=np.float32,
    )
    x_counts = vectorizer.transform(corpus["text"].tolist())
    x_tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(x_counts)
    n_components = min(config.svd_dim, x_tfidf.shape[0] - 2, x_tfidf.shape[1] - 2)
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state, n_iter=5)
    dense = svd.fit_transform(x_tfidf).astype(np.float32)
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
    meta = {
        "hash_features": int(config.hash_features),
        "svd_dim": int(dense.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return dense, meta


def _cluster_count(n: int, config: LargeIvfPqConfig) -> int:
    return int(min(config.max_clusters, max(config.min_clusters, round(n / config.docs_per_cell))))


def _train_ivf(vectors: np.ndarray, config: LargeIvfPqConfig, seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
    k = _cluster_count(len(vectors), config)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=config.random_state + seed_offset,
        batch_size=config.batch_size,
        max_iter=config.kmeans_iter,
        n_init=3,
        reassignment_ratio=0.01,
    )
    assignments = model.fit_predict(vectors)
    centroids = normalize(model.cluster_centers_.astype(np.float32), norm="l2", axis=1)
    return assignments.astype(np.int32), centroids.astype(np.float32)


def _train_pq(vectors: np.ndarray, config: LargeIvfPqConfig, seed_offset: int) -> np.ndarray:
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
        k = min(config.pq_codewords, max(2, n // 256))
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.random_state + seed_offset + 31 * part,
            batch_size=config.batch_size,
            max_iter=config.kmeans_iter,
            n_init=2,
            reassignment_ratio=0.01,
        )
        codes = model.fit_predict(block)
        reconstructed[:, start:end] = model.cluster_centers_.astype(np.float32)[codes]
    return normalize(reconstructed, norm="l2", axis=1).astype(np.float32)


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
                "cluster_id": int(cluster_id),
                "n": int(idx.size),
                "radius": float(np.mean(1.0 - sims)),
                "radius_std": float(np.std(1.0 - sims)),
                "label_entropy": entropy,
                "majority_share": float(counts.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _top_order(scores: np.ndarray, k: int) -> np.ndarray:
    valid = np.isfinite(scores)
    if not np.any(valid):
        return np.array([], dtype=np.int64)
    kk = min(k, int(np.sum(valid)))
    if kk <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, kk - 1)[:kk]
    return part[np.argsort(-scores[part])]


def _ndcg(labels: np.ndarray, query_label: str, order: np.ndarray, total_rel: int, k: int) -> float:
    if order.size == 0 or total_rel <= 0:
        return 0.0
    rel = (labels[order[:k]] == query_label).astype(float)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_len = min(k, total_rel)
    idcg = float(np.sum(discounts[:ideal_len]))
    return dcg / idcg if idcg > 0 else 0.0


def _query_records_for_dataset(
    dataset: str,
    docs: pd.DataFrame,
    vectors: np.ndarray,
    pq_vectors: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    cluster_frame: pd.DataFrame,
    config: LargeIvfPqConfig,
) -> pd.DataFrame:
    labels = docs["label"].to_numpy(dtype=object)
    cluster_lookup = {int(row.cluster_id): row for row in cluster_frame.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    costs = {"summary": config.cost_summary, "pq": config.cost_pq, "full": config.cost_full}
    for cluster_id in sorted(np.unique(assignments)):
        local_idx = np.where(assignments == cluster_id)[0]
        m = len(local_idx)
        if m < 3:
            continue
        v = vectors[local_idx]
        pq = pq_vectors[local_idx]
        local_labels = labels[local_idx]
        centroid = centroids[int(cluster_id)]
        summary_scores = v @ centroid
        summary_base = np.argsort(-summary_scores)
        full_scores = v @ v.T
        pq_scores = v @ pq.T
        np.fill_diagonal(full_scores, -np.inf)
        np.fill_diagonal(pq_scores, -np.inf)
        stats = cluster_lookup[int(cluster_id)]
        label_counts = pd.Series(local_labels).value_counts().to_dict()
        for j, global_idx in enumerate(local_idx):
            q_label = local_labels[j]
            total_rel = int(label_counts.get(q_label, 0)) - 1
            if total_rel <= 0:
                continue
            summary_order = summary_base[summary_base != j][: config.topk]
            pq_order = _top_order(pq_scores[j], config.topk)
            full_order = _top_order(full_scores[j], config.topk)
            ndcg_summary = _ndcg(local_labels, q_label, summary_order, total_rel, config.topk)
            ndcg_pq = _ndcg(local_labels, q_label, pq_order, total_rel, config.topk)
            ndcg_full = _ndcg(local_labels, q_label, full_order, total_rel, config.topk)
            utilities = {
                "summary": ndcg_summary - config.lambda_cost * costs["summary"],
                "pq": ndcg_pq - config.lambda_cost * costs["pq"],
                "full": ndcg_full - config.lambda_cost * costs["full"],
            }
            best_view = max(utilities, key=utilities.get)
            rows.append(
                {
                    "dataset": dataset,
                    "doc_id": docs.iloc[int(global_idx)]["doc_id"],
                    "query_index": int(global_idx),
                    "label": str(q_label),
                    "cluster_id": int(cluster_id),
                    "candidate_count": int(m - 1),
                    "cluster_n": int(stats.n),
                    "cluster_radius": float(stats.radius),
                    "cluster_radius_std": float(stats.radius_std),
                    "cluster_label_entropy": float(stats.label_entropy),
                    "cluster_majority_share": float(stats.majority_share),
                    "query_centroid_distance": float(1.0 - np.dot(v[j], centroid)),
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


def _fixed_policy(frame: pd.DataFrame, view: str, config: LargeIvfPqConfig) -> dict[str, Any]:
    return {
        "policy": f"fixed_{view}",
        "utility": float(frame[f"{view}_utility"].mean()),
        "ndcg": float(frame[f"{view}_ndcg"].mean()),
        "avg_cost": {"summary": config.cost_summary, "pq": config.cost_pq, "full": config.cost_full}[view],
        "best_view_acc": float((frame["best_view"] == view).mean()),
        "non_full_share": float(view != "full"),
    }


def _proxy_router(frame: pd.DataFrame, config: LargeIvfPqConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "candidate_count",
        "cluster_n",
        "cluster_radius",
        "cluster_radius_std",
        "cluster_majority_share",
        "query_centroid_distance",
    ]
    targets = ["summary_utility", "pq_utility", "full_utility"]
    groups = frame["dataset"].astype(str) + ":" + frame["cluster_id"].astype(str)
    n_splits = min(5, max(2, groups.nunique()))
    splitter = GroupKFold(n_splits=n_splits)
    x = frame[features]
    y = frame[targets]
    costs = {"summary": config.cost_summary, "pq": config.cost_pq, "full": config.cost_full}
    views = np.array(["summary", "pq", "full"])
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, groups=groups), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=10,
                    min_samples_leaf=12,
                    random_state=config.random_state + fold,
                    n_jobs=-1,
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
    ci = 1.96 * folds["utility"].std(ddof=1) / math.sqrt(len(folds)) if len(folds) > 1 else 0.0
    summary = pd.DataFrame(
        [
            {
                "policy": "proxy_router_rf",
                "utility": float(folds["utility"].mean()),
                "utility_sd": float(folds["utility"].std(ddof=1)) if len(folds) > 1 else 0.0,
                "utility_ci95": float(ci),
                "ndcg": float(folds["ndcg"].mean()),
                "avg_cost": float(folds["avg_cost"].mean()),
                "regret": float(folds["regret"].mean()),
                "best_view_acc": float(folds["best_view_acc"].mean()),
                "non_full_share": float(folds["non_full_share"].mean()),
            }
        ]
    )
    return summary, folds


def _aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    groups = [("all", frame), *frame.groupby("dataset")]
    for name, group in groups:
        rows.append(
            {
                "scope": str(name),
                "queries": int(len(group)),
                "cells": int(group[["dataset", "cluster_id"]].drop_duplicates().shape[0]),
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
        )
    aggregate = pd.DataFrame(rows)
    q30 = frame["cluster_radius"].quantile(0.30)
    q70 = frame["cluster_radius"].quantile(0.70)
    slices: list[dict[str, Any]] = []
    for name, group in [("compact_30pct", frame[frame["cluster_radius"] <= q30]), ("diffuse_30pct", frame[frame["cluster_radius"] >= q70])]:
        slices.append(
            {
                "slice": name,
                "queries": int(len(group)),
                "cluster_radius_mean": float(group["cluster_radius"].mean()),
                "summary_ndcg": float(group["summary_ndcg"].mean()),
                "pq_ndcg": float(group["pq_ndcg"].mean()),
                "full_ndcg": float(group["full_ndcg"].mean()),
                "full_gain": float(group["full_gain_over_summary"].mean()),
                "full_dominated_by_pq_share": float((group["full_minus_pq_ndcg"] < 0).mean()),
                "best_not_full_share": float((group["best_view"] != "full").mean()),
                "near_zero_full_gain_share": float((group["full_gain_over_summary"] <= 0.01).mean()),
            }
        )
    return aggregate, pd.DataFrame(slices)


def run(config: LargeIvfPqConfig) -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    corpus, sources = load_corpus(config)
    vectors, vector_meta = _build_vectors(corpus, config)
    query_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    dataset_manifest: list[dict[str, Any]] = []
    for offset, (dataset, docs) in enumerate(corpus.groupby("dataset", sort=True)):
        idx = docs.index.to_numpy()
        local_vectors = vectors[idx]
        local_docs = docs.reset_index(drop=True)
        assignments, centroids = _train_ivf(local_vectors, config, seed_offset=101 * (offset + 1))
        pq_vectors = _train_pq(local_vectors, config, seed_offset=1009 * (offset + 1))
        cluster_frame = _cluster_stats(local_vectors, local_docs["label"].to_numpy(dtype=object), assignments, centroids)
        cluster_frame.insert(0, "dataset", dataset)
        cluster_frames.append(cluster_frame)
        query_frame = _query_records_for_dataset(
            dataset, local_docs, local_vectors, pq_vectors, assignments, centroids, cluster_frame, config
        )
        query_frames.append(query_frame)
        dataset_manifest.append(
            {
                "dataset": dataset,
                "documents": int(len(local_docs)),
                "queries_scored": int(len(query_frame)),
                "ivf_cells": int(cluster_frame["cluster_id"].nunique()),
                "mean_cell_size": float(cluster_frame["n"].mean()),
                "labels": int(local_docs["label"].nunique()),
            }
        )
    all_queries = pd.concat(query_frames, ignore_index=True)
    clusters = pd.concat(cluster_frames, ignore_index=True)
    aggregate, slices = _aggregate(all_queries)
    proxy_summary, proxy_folds = _proxy_router(all_queries, config)
    fixed = pd.DataFrame([_fixed_policy(all_queries, view, config) for view in ["summary", "pq", "full"]])
    policy = pd.concat([fixed, proxy_summary], ignore_index=True, sort=False)
    aggregate.to_csv(OUT_DIR / "large_ivf_pq_rerank_summary.csv", index=False)
    slices.to_csv(OUT_DIR / "large_ivf_pq_rerank_sufficiency_slices.csv", index=False)
    policy.to_csv(OUT_DIR / "large_ivf_pq_rerank_policy.csv", index=False)
    proxy_folds.to_csv(OUT_DIR / "large_ivf_pq_rerank_proxy_folds.csv", index=False)
    clusters.to_csv(OUT_DIR / "large_ivf_cluster_stats.csv", index=False)
    sample_n = min(config.per_query_sample, len(all_queries))
    all_queries.sample(n=sample_n, random_state=config.random_state).to_csv(
        OUT_DIR / "large_ivf_pq_rerank_per_query_sample.csv", index=False
    )
    elapsed = time.perf_counter() - start
    manifest = {
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "sources": sources,
        "datasets": dataset_manifest,
        "vectorization": vector_meta,
        "mapping": {
            "cell": "IVF-style coarse text-vector cluster",
            "summary": "cluster centroid, dispersion, and list size",
            "pq_view": "product-quantized subvector reconstruction used for approximate reranking",
            "full_view": "full dense-vector reranking within the selected coarse cell",
            "utility": f"NDCG@{config.topk} - lambda * access_cost, lambda={config.lambda_cost}",
        },
        "outputs": {
            "summary": _manifest_path(OUT_DIR / "large_ivf_pq_rerank_summary.csv"),
            "slices": _manifest_path(OUT_DIR / "large_ivf_pq_rerank_sufficiency_slices.csv"),
            "policy": _manifest_path(OUT_DIR / "large_ivf_pq_rerank_policy.csv"),
            "folds": _manifest_path(OUT_DIR / "large_ivf_pq_rerank_proxy_folds.csv"),
            "cluster_stats": _manifest_path(OUT_DIR / "large_ivf_cluster_stats.csv"),
            "per_query_sample": _manifest_path(OUT_DIR / "large_ivf_pq_rerank_per_query_sample.csv"),
        },
    }
    (REPORTS / "large_ivf_pq_rerank_adapter_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (REPORTS / "large_ivf_pq_rerank_adapter_audit.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Large IVF-PQ Rerank Adapter Audit\n\n")
        f.write(
            f"Scored {len(all_queries):,} queries over "
            f"{clusters[['dataset', 'cluster_id']].drop_duplicates().shape[0]:,} IVF-style cells in {elapsed/60:.1f} minutes.\n\n"
        )
        f.write("## Aggregate\n\n")
        f.write(_markdown_table(aggregate))
        f.write("\n\n## Sufficiency slices\n\n")
        f.write(_markdown_table(slices))
        f.write("\n\n## Policies\n\n")
        f.write(_markdown_table(policy))
        f.write("\n")
    print(f"elapsed_min={elapsed/60:.2f}")
    print(aggregate.to_string(index=False))
    print()
    print(slices.to_string(index=False))
    print()
    print(policy.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a larger IVF-PQ-style rerank adapter audit over public text datasets.")
    parser.add_argument("--target-docs", type=int, default=180_000)
    parser.add_argument("--svd-dim", type=int, default=192)
    parser.add_argument("--hash-features", type=int, default=2**18)
    parser.add_argument("--docs-per-cell", type=int, default=250)
    parser.add_argument("--max-clusters", type=int, default=512)
    parser.add_argument("--pq-subvectors", type=int, default=12)
    parser.add_argument("--pq-codewords", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--max-docred", type=int, default=80_000)
    parser.add_argument("--max-douban", type=int, default=55_000)
    parser.add_argument("--max-jd", type=int, default=30_000)
    parser.add_argument("--max-taptap", type=int, default=15_000)
    args = parser.parse_args()
    config = LargeIvfPqConfig(
        target_docs=args.target_docs,
        svd_dim=args.svd_dim,
        hash_features=args.hash_features,
        docs_per_cell=args.docs_per_cell,
        max_clusters=args.max_clusters,
        pq_subvectors=args.pq_subvectors,
        pq_codewords=args.pq_codewords,
        topk=args.topk,
        lambda_cost=args.lambda_cost,
        max_docred=args.max_docred,
        max_douban=args.max_douban,
        max_jd=args.max_jd,
        max_taptap=args.max_taptap,
    )
    run(config)


if __name__ == "__main__":
    main()
