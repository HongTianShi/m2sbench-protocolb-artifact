from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_2wiki_structured_evidence_audit import (
    bm25_scores,
    entropy_from_scores,
    f1_at_k,
    margin,
    markdown_table,
    ndcg_at_k,
    norm_title,
    random_matched_scores,
    rank_from_scores,
    recall_at_k,
    safe_id,
    split_indices,
    token_set,
    tokens,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_MUSIQUE = Path(os.environ.get("M2SBENCH_MUSIQUE_DIR", "PATH_TO_MUSIQUE_HF_EXPORT"))

BASE_VIEW_ORDER = ["summary", "decomp_title", "paragraph", "support_set", "full_context"]
VIEW_COST = {
    "summary": 0.0,
    "decomp_title": 0.04,
    "paragraph": 0.16,
    "support_set": 0.30,
    "full_context": 0.55,
    "ce": 0.95,
}
CONTROL_ORDER_BASE = [
    "title_only",
    "decomposition_title",
    "real_paragraph",
    "real_support_set",
    "shuffled_paragraph",
    "same_title_shuffled_mapping",
    "random_paragraph",
    "same_count_random",
    "decomposition_shuffled",
    "full_context",
]
CONTROL_COST = {
    "title_only": VIEW_COST["summary"],
    "decomposition_title": VIEW_COST["decomp_title"],
    "real_paragraph": VIEW_COST["paragraph"],
    "real_support_set": VIEW_COST["support_set"],
    "shuffled_paragraph": VIEW_COST["paragraph"],
    "same_title_shuffled_mapping": VIEW_COST["paragraph"],
    "random_paragraph": VIEW_COST["paragraph"],
    "same_count_random": VIEW_COST["paragraph"],
    "decomposition_shuffled": VIEW_COST["support_set"],
    "full_context": VIEW_COST["full_context"],
    "ce": VIEW_COST["ce"],
}
FEATURE_TIERS = ["B0_title_decomp", "B1_paragraph_meta", "B1_decomp_meta", "B1_context_sketch"]


def view_order(include_ce: bool) -> list[str]:
    return BASE_VIEW_ORDER + (["ce"] if include_ce else [])


def control_order(include_ce: bool) -> list[str]:
    return CONTROL_ORDER_BASE + (["ce"] if include_ce else [])


def title_overlap_score(question: str, title: str) -> float:
    q = token_set(question)
    t = token_set(title)
    if not t:
        return 0.0
    score = len(q & t) / max(1, len(t))
    if norm_title(title) in norm_title(question):
        score += 1.0
    return float(score)


def row_hop_type(row_id: str, n_support: int) -> str:
    if "__" in row_id:
        return row_id.split("__", 1)[0]
    return f"{n_support}support"


def load_musique_rows(data_dir: Path, max_rows: int, split: str) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(data_dir))[split]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if len(rows) >= max_rows:
            break
        paragraphs = []
        for para in row["paragraphs"]:
            text = str(para.get("paragraph_text", "")).strip()
            title = str(para.get("title", "")).strip()
            if not text or not title:
                continue
            paragraphs.append(
                {
                    "idx": int(para.get("idx", len(paragraphs))),
                    "title": title,
                    "paragraph_text": text,
                    "is_supporting": bool(para.get("is_supporting", False)),
                }
            )
        support = [p for p in paragraphs if p["is_supporting"]]
        if not bool(row.get("answerable", False)) or not support:
            continue
        decomp_questions: list[str] = []
        decomp_answers: list[str] = []
        decomp_support_idx: list[int | None] = []
        for item in row.get("question_decomposition", []):
            q = str(item.get("question", "")).strip()
            if q:
                decomp_questions.append(q)
            decomp_answers.append(str(item.get("answer", "")))
            support_idx = item.get("paragraph_support_idx", None)
            decomp_support_idx.append(int(support_idx) if support_idx is not None else None)
        support_idx = {str(p["idx"]) for p in support}
        support_titles = {norm_title(p["title"]) for p in support}
        rows.append(
            {
                "id": str(row.get("id") or f"musique_{idx}"),
                "question": str(row["question"]),
                "answer": str(row.get("answer", "")),
                "answer_aliases": list(row.get("answer_aliases", [])),
                "answerable": bool(row.get("answerable", False)),
                "paragraphs": paragraphs,
                "decomp_questions": decomp_questions,
                "decomp_answers": decomp_answers,
                "decomp_support_idx": decomp_support_idx,
                "support_idx": support_idx,
                "support_titles": support_titles,
                "type": row_hop_type(str(row.get("id") or f"musique_{idx}"), len(support)),
            }
        )
    return rows


def support_title_recall(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    ranked = {norm_title(t) for t in ranked_titles[:k]}
    return float(len(ranked & gold_titles) / max(1, len(gold_titles)))


def paragraph_metrics(
    row: dict[str, Any],
    view: str,
    ranked_idx: list[str],
    ranked_titles: list[str],
    scores: np.ndarray,
    lambda_cost: float,
    cost: float,
    topk: int,
) -> dict[str, Any]:
    gold_idx = set(row["support_idx"])
    ndcg = ndcg_at_k(ranked_idx, gold_idx, topk)
    recall = recall_at_k(ranked_idx, gold_idx, topk)
    f1 = f1_at_k(ranked_idx, gold_idx, topk)
    return {
        "view": view,
        "ndcg": float(ndcg),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_title_recall": support_title_recall(ranked_titles, row["support_titles"], topk),
        "utility": float(ndcg - lambda_cost * cost),
        "cost": cost,
        "score_top": float(np.max(scores)) if len(scores) else 0.0,
        "score_margin": margin(scores),
        "score_entropy": entropy_from_scores(scores),
        "ranked_idx": ranked_idx[:topk],
        "ranked_titles": ranked_titles[:topk],
    }


def score_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    include_ce = not args.no_ce
    ce_scores_by_id: dict[str, np.ndarray] = {}
    if include_ce:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(args.ce_model, device=device)
        pairs: list[tuple[str, str]] = []
        row_slices: list[tuple[str, int, int]] = []
        cursor = 0
        for row in rows:
            start = cursor
            for para in row["paragraphs"]:
                text = para["paragraph_text"][: args.max_doc_chars]
                pairs.append((row["question"], f"{para['title']}. {text}"))
                cursor += 1
            row_slices.append((row["id"], start, cursor))
        scores = model.predict(pairs, batch_size=args.ce_batch_size, show_progress_bar=True)
        scores = np.asarray(scores, dtype=np.float32)
        for row_id, start, end in row_slices:
            ce_scores_by_id[row_id] = scores[start:end]

    rng = np.random.default_rng(args.seed)
    all_decomp_texts = [" ".join(r["decomp_questions"]) for r in rows if r["decomp_questions"]]
    records: list[dict[str, Any]] = []
    views = view_order(include_ce)
    controls = control_order(include_ce)
    for row in rows:
        paragraphs = row["paragraphs"]
        ids = [str(p["idx"]) for p in paragraphs]
        titles = [p["title"] for p in paragraphs]
        title_texts = titles
        paragraph_texts = [f"{p['title']}. {p['paragraph_text'][: args.max_doc_chars]}" for p in paragraphs]
        decomp_text = " ".join(row["decomp_questions"])
        decomp_query = f"{row['question']} {decomp_text}".strip()

        summary_scores = bm25_scores(row["question"], title_texts)
        overlap = np.asarray([title_overlap_score(row["question"], t) for t in titles], dtype=np.float32)
        summary_scores = summary_scores + 0.50 * overlap
        decomp_title_scores = bm25_scores(decomp_query, title_texts) + 0.25 * summary_scores
        paragraph_scores = bm25_scores(row["question"], paragraph_texts) + 0.15 * summary_scores
        decomp_parts = [bm25_scores(q, paragraph_texts) for q in row["decomp_questions"]]
        decomp_max = np.max(np.vstack(decomp_parts), axis=0) if decomp_parts else np.zeros(len(paragraphs), dtype=np.float32)
        support_set_scores = bm25_scores(decomp_query, paragraph_texts) + 0.30 * decomp_max + 0.15 * paragraph_scores
        full_context_scores = bm25_scores(decomp_query, paragraph_texts) + 0.20 * support_set_scores

        scores_by_view: dict[str, np.ndarray] = {
            "summary": summary_scores,
            "decomp_title": decomp_title_scores,
            "paragraph": paragraph_scores,
            "support_set": support_set_scores,
            "full_context": full_context_scores,
        }
        if include_ce:
            ce_scores = ce_scores_by_id[row["id"]]
            scores_by_view["ce"] = ce_scores + 0.10 * support_set_scores

        view_records: dict[str, dict[str, Any]] = {}
        for view in views:
            scores = scores_by_view[view]
            ranked_ids = rank_from_scores(ids, scores)
            id_to_title = {str(p["idx"]): p["title"] for p in paragraphs}
            ranked_titles = [id_to_title[idx] for idx in ranked_ids]
            view_records[view] = paragraph_metrics(
                row, view, ranked_ids, ranked_titles, scores, args.lambda_cost, VIEW_COST[view], args.topk
            )

        shuffled = np.asarray(paragraph_scores, dtype=np.float32).copy()
        rng.shuffle(shuffled)
        same_title = np.asarray(paragraph_scores, dtype=np.float32).copy()
        rng.shuffle(same_title)
        random_scores = random_matched_scores(paragraph_scores, rng)
        same_count_scores = rng.random(len(paragraphs), dtype=np.float32)
        if all_decomp_texts:
            other_decomp = all_decomp_texts[int(rng.integers(0, len(all_decomp_texts)))]
        else:
            other_decomp = ""
        decomposition_shuffled_scores = bm25_scores(f"{row['question']} {other_decomp}", paragraph_texts) + 0.15 * paragraph_scores
        control_scores: dict[str, np.ndarray] = {
            "title_only": summary_scores,
            "decomposition_title": decomp_title_scores,
            "real_paragraph": paragraph_scores,
            "real_support_set": support_set_scores,
            "shuffled_paragraph": shuffled,
            "same_title_shuffled_mapping": same_title,
            "random_paragraph": random_scores,
            "same_count_random": same_count_scores,
            "decomposition_shuffled": decomposition_shuffled_scores,
            "full_context": full_context_scores,
        }
        if include_ce:
            control_scores["ce"] = scores_by_view["ce"]

        control_records: dict[str, dict[str, Any]] = {}
        for control in controls:
            scores = control_scores[control]
            ranked_ids = rank_from_scores(ids, scores)
            id_to_title = {str(p["idx"]): p["title"] for p in paragraphs}
            ranked_titles = [id_to_title[idx] for idx in ranked_ids]
            control_records[control] = paragraph_metrics(
                row, control, ranked_ids, ranked_titles, scores, args.lambda_cost, CONTROL_COST[control], args.topk
            )

        oracle_view = max(views, key=lambda v: view_records[v]["utility"])
        doc_lens = [len(tokens(p["paragraph_text"])) for p in paragraphs]
        decomp_lens = [len(tokens(q)) for q in row["decomp_questions"]]
        records.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "answer": row["answer"],
                "decomp_questions": row["decomp_questions"],
                "decomp_answers": row["decomp_answers"],
                "decomp_support_idx": row["decomp_support_idx"],
                "support_idx": sorted(row["support_idx"]),
                "support_titles": sorted(row["support_titles"]),
                "n_context": len(paragraphs),
                "n_decomp": len(row["decomp_questions"]),
                "question_len": len(tokens(row["question"])),
                "context_stats": {
                    "mean_doc_len": float(np.mean(doc_lens)) if doc_lens else 0.0,
                    "std_doc_len": float(np.std(doc_lens)) if doc_lens else 0.0,
                    "max_doc_len": float(np.max(doc_lens)) if doc_lens else 0.0,
                    "mean_decomp_len": float(np.mean(decomp_lens)) if decomp_lens else 0.0,
                    "std_decomp_len": float(np.std(decomp_lens)) if decomp_lens else 0.0,
                    "max_decomp_len": float(np.max(decomp_lens)) if decomp_lens else 0.0,
                    "title_overlap_mean": float(np.mean(overlap)) if len(overlap) else 0.0,
                    "title_overlap_max": float(np.max(overlap)) if len(overlap) else 0.0,
                },
                "views": view_records,
                "controls": control_records,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def fixed_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str], views: list[str]) -> dict[str, Any]:
    utils, regrets, ndcgs, recalls, f1s, title_recalls, costs, chosen = [], [], [], [], [], [], [], []
    for pos, idx in enumerate(indices):
        rec = records[int(idx)]
        view = route[pos] if isinstance(route, list) else route
        vr = rec["views"][view]
        utils.append(vr["utility"])
        regrets.append(rec["oracle_utility"] - vr["utility"])
        ndcgs.append(vr["ndcg"])
        recalls.append(vr["support_recall"])
        f1s.append(vr["support_f1"])
        title_recalls.append(vr["support_title_recall"])
        costs.append(vr["cost"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utils)) if utils else 0.0,
        "raw_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "support_recall": float(np.mean(recalls)) if recalls else 0.0,
        "support_f1": float(np.mean(f1s)) if f1s else 0.0,
        "support_title_recall": float(np.mean(title_recalls)) if title_recalls else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
        "regret": float(np.mean(regrets)) if regrets else 0.0,
        "ce_buy": float(counts["ce"] / max(1, len(chosen))),
        "view_share": {v: round(counts[v] / max(1, len(chosen)), 4) for v in views if counts[v]},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray, tier: str, views: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    type_values = sorted({rec["type"] for rec in records})
    rows: list[list[float]] = []
    y: list[float] = []
    keys: list[tuple[int, str]] = []
    for idx in indices:
        rec = records[int(idx)]
        summary = rec["views"]["summary"]
        decomp = rec["views"]["decomp_title"]
        stats = rec["context_stats"]
        base = [
            rec["question_len"],
            rec["n_context"],
            rec["n_decomp"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
            decomp["score_top"],
            decomp["score_margin"],
            decomp["score_entropy"],
            stats["title_overlap_mean"],
            stats["title_overlap_max"],
        ]
        if tier in {"B1_paragraph_meta", "B1_decomp_meta", "B1_context_sketch"}:
            base += [stats["mean_doc_len"], stats["std_doc_len"], stats["max_doc_len"]]
        if tier in {"B1_decomp_meta", "B1_context_sketch"}:
            base += [stats["mean_decomp_len"], stats["std_decomp_len"], stats["max_decomp_len"]]
        if tier == "B1_context_sketch":
            base += [1.0 if rec["type"] == t else 0.0 for t in type_values]
        for view_i, view in enumerate(views):
            vr = rec["views"][view]
            view_profile = [
                vr["cost"],
                1.0 if view == "decomp_title" else 0.0,
                1.0 if view == "paragraph" else 0.0,
                1.0 if view == "support_set" else 0.0,
                1.0 if view == "full_context" else 0.0,
                1.0 if view == "ce" else 0.0,
            ]
            rows.append(base + [1.0 if i == view_i else 0.0 for i in range(len(views))] + view_profile)
            y.append(vr["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def route_from_predictions(keys: list[tuple[int, str]], preds: np.ndarray, views: list[str]) -> list[str]:
    by_idx: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (idx, view), pred in zip(keys, preds):
        by_idx[idx].append((view, float(pred)))
    return [max(by_idx[idx], key=lambda x: (x[1], -views.index(x[0])))[0] for idx in sorted(by_idx)]


def train_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int, views: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    tier_rows: list[dict[str, Any]] = []
    best_dev = -1e9
    best_model_name = ""
    best_tier = ""
    best_model: tuple[Any, str, list[tuple[int, str]]] | None = None
    for tier in FEATURE_TIERS:
        models = {
            f"rf_{tier}": RandomForestRegressor(n_estimators=180, min_samples_leaf=4, random_state=seed, n_jobs=-1),
            f"et_{tier}": ExtraTreesRegressor(n_estimators=240, min_samples_leaf=3, random_state=seed, n_jobs=-1),
            f"hgb_{tier}": HistGradientBoostingRegressor(max_iter=140, learning_rate=0.05, random_state=seed),
        }
        x_train, y_train, _ = make_features(records, train_idx, tier, views)
        x_dev, _, dev_keys = make_features(records, dev_idx, tier, views)
        x_test, _, test_keys = make_features(records, test_idx, tier, views)
        for name, model in models.items():
            model.fit(x_train, y_train)
            dev_route = route_from_predictions(dev_keys, model.predict(x_dev), views)
            dev_eval = fixed_policy(records, np.asarray(sorted(set(i for i, _ in dev_keys))), name, dev_route, views)
            test_route = route_from_predictions(test_keys, model.predict(x_test), views)
            test_eval = fixed_policy(records, np.asarray(sorted(set(i for i, _ in test_keys))), name, test_route, views)
            test_eval["feature_tier"] = tier
            test_eval["dev_utility"] = dev_eval["utility"]
            tier_rows.append(test_eval)
            if dev_eval["utility"] > best_dev:
                best_dev = dev_eval["utility"]
                best_model_name = name
                best_tier = tier
                best_model = (model, tier, test_keys)
    if best_model is None:
        return fixed_policy(records, test_idx, "best_legal_router(empty)", "summary", views), tier_rows
    model, tier, test_keys = best_model
    x_test, _, test_keys = make_features(records, test_idx, tier, views)
    test_route = route_from_predictions(test_keys, model.predict(x_test), views)
    out = fixed_policy(records, np.asarray(sorted(set(i for i, _ in test_keys))), f"best_legal_router({best_model_name})", test_route, views)
    out["feature_tier"] = best_tier
    out["dev_utility"] = float(best_dev)
    out["routes"] = test_route
    return out, tier_rows


def train_threshold_gate(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, views: list[str]) -> dict[str, Any]:
    rich_views = [v for v in views if v != "summary"]
    vals = np.asarray([records[int(i)]["views"]["summary"]["score_margin"] for i in train_idx], dtype=np.float64)
    thresholds = sorted(set(float(x) for x in np.quantile(vals, np.linspace(0.05, 0.95, 19)))) if len(vals) else [0.0]
    candidates: list[tuple[float, float, str, str]] = []
    for threshold in thresholds:
        for direction in ["high_summary", "low_summary"]:
            for rich in rich_views:
                route = []
                for idx in dev_idx:
                    val = records[int(idx)]["views"]["summary"]["score_margin"]
                    choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
                    route.append("summary" if choose_summary else rich)
                dev_eval = fixed_policy(records, dev_idx, "dev_gate", route, views)
                candidates.append((dev_eval["utility"], threshold, direction, rich))
    if not candidates:
        return fixed_policy(records, test_idx, "threshold_gate(summary)", "summary", views)
    dev_utility, threshold, direction, rich = max(candidates, key=lambda x: x[0])
    route = []
    for idx in test_idx:
        val = records[int(idx)]["views"]["summary"]["score_margin"]
        choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
        route.append("summary" if choose_summary else rich)
    out = fixed_policy(records, test_idx, f"threshold_gate(summary,{rich})", route, views)
    out["gate_threshold"] = float(threshold)
    out["gate_direction"] = direction
    out["rich_view"] = rich
    out["dev_utility"] = float(dev_utility)
    return out


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    include_ce = not args.no_ce
    views = view_order(include_ce)
    controls = control_order(include_ce)
    train_idx, dev_idx, test_idx = split_indices(len(records), args.seed)
    fixed_rows = [fixed_policy(records, test_idx, f"fixed_{view}", view, views) for view in views]
    best_fixed = max(fixed_rows, key=lambda row: row["utility"])
    gate = train_threshold_gate(records, train_idx, dev_idx, test_idx, views)
    learned, tier_rows = train_router(records, train_idx, dev_idx, test_idx, args.seed, views)
    best_adaptive = max([gate, learned], key=lambda row: row.get("dev_utility", -1e9))
    best_adaptive = dict(best_adaptive)
    best_adaptive["policy"] = "best_legal_adaptive"
    oracle_route = [records[int(i)]["oracle_view"] for i in test_idx]
    oracle = fixed_policy(records, test_idx, "oracle_route", oracle_route, views)
    denom = max(oracle["utility"] - best_fixed["utility"], 1e-12)
    for row in fixed_rows + [gate, learned, best_adaptive, oracle] + tier_rows:
        row["diff_vs_best_fixed"] = row["utility"] - best_fixed["utility"]
        row["gap_closed"] = (row["utility"] - best_fixed["utility"]) / denom

    repeat = []
    selected = []
    for r in range(args.repeats):
        tr, dv, te = split_indices(len(records), args.seed + 100 + r)
        fixed = [fixed_policy(records, te, f"fixed_{view}", view, views) for view in views]
        bf = max(fixed, key=lambda row: row["utility"])
        gate_r = train_threshold_gate(records, tr, dv, te, views)
        learned_r, _ = train_router(records, tr, dv, te, args.seed + r + 1, views)
        adaptive_r = max([gate_r, learned_r], key=lambda row: row.get("dev_utility", -1e9))
        repeat.append(adaptive_r["utility"] - bf["utility"])
        selected.append(adaptive_r["policy"])
    repeat_arr = np.asarray(repeat, dtype=np.float64)

    control_rows = []
    for control in controls:
        utils, ndcgs, recalls, f1s, title_recalls, costs, regrets = [], [], [], [], [], [], []
        for idx in test_idx:
            rec = records[int(idx)]
            row = rec["controls"][control]
            utils.append(row["utility"])
            ndcgs.append(row["ndcg"])
            recalls.append(row["support_recall"])
            f1s.append(row["support_f1"])
            title_recalls.append(row["support_title_recall"])
            costs.append(row["cost"])
            regrets.append(rec["oracle_utility"] - row["utility"])
        control_rows.append(
            {
                "control": control,
                "utility": float(np.mean(utils)),
                "raw_ndcg": float(np.mean(ndcgs)),
                "support_recall": float(np.mean(recalls)),
                "support_f1": float(np.mean(f1s)),
                "support_title_recall": float(np.mean(title_recalls)),
                "cost": float(np.mean(costs)),
                "regret": float(np.mean(regrets)),
            }
        )
    best_control = max(control_rows, key=lambda row: row["utility"])
    for row in control_rows:
        row["diff_vs_best_control"] = row["utility"] - best_control["utility"]

    type_rows = []
    for qtype in sorted({records[int(i)]["type"] for i in test_idx}):
        idxs = np.asarray([int(i) for i in test_idx if records[int(i)]["type"] == qtype])
        fixed = [fixed_policy(records, idxs, f"fixed_{view}", view, views) for view in views]
        bf = max(fixed, key=lambda row: row["utility"])
        oracle_type = fixed_policy(records, idxs, "oracle_route", [records[int(i)]["oracle_view"] for i in idxs], views)
        type_rows.append(
            {
                "type": qtype,
                "n": int(len(idxs)),
                "best_fixed": bf["policy"],
                "fixed_utility": bf["utility"],
                "oracle_utility": oracle_type["utility"],
                "oracle_gap": oracle_type["utility"] - bf["utility"],
            }
        )

    oracle_share = Counter(records[int(i)]["oracle_view"] for i in test_idx)
    return {
        "task": "musique_structured_evidence",
        "dataset": "MuSiQue",
        "status": "candidate_structured_replication_check",
        "scope_note": "Supporting structured-evidence replication over MuSiQue, not a separate official leaderboard core.",
        "n_rows": len(records),
        "split": args.split,
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "view_costs": {v: VIEW_COST[v] for v in views},
        "views": {
            "summary": "question plus paragraph-title sketch only",
            "decomp_title": "question plus decomposition-question text and paragraph titles",
            "paragraph": "paragraph text evidence scored by the original question",
            "support_set": "multi-paragraph evidence scored by question-decomposition queries",
            "full_context": "all local paragraphs for the example",
            "ce": "cross-encoder rerank over purchased paragraph evidence",
        },
        "hidden_fields": [
            "is_supporting",
            "answer",
            "answer_aliases",
            "question_decomposition.answer",
            "question_decomposition.paragraph_support_idx",
            "unpaid CE/full scores",
        ],
        "policy_rows": fixed_rows + [gate, learned, best_adaptive, oracle],
        "best_fixed": best_fixed["policy"],
        "feature_tier_rows": sorted(tier_rows, key=lambda row: row["utility"], reverse=True)[:8],
        "control_rows": control_rows,
        "type_rows": type_rows,
        "oracle_view_share": {k: float(v / max(1, len(test_idx))) for k, v in oracle_share.items()},
        "repeat_summary": {
            "learned_minus_best_fixed_mean": float(np.mean(repeat_arr)) if len(repeat_arr) else 0.0,
            "ci95": [float(np.quantile(repeat_arr, 0.025)), float(np.quantile(repeat_arr, 0.975))] if len(repeat_arr) else [0.0, 0.0],
            "positive_share": float(np.mean(repeat_arr > 0)) if len(repeat_arr) else 0.0,
            "selected_policies": dict(Counter(selected)),
            "repeats": int(len(repeat_arr)),
        },
    }


def build_trace(records: list[dict[str, Any]], include_ce: bool) -> dict[str, Any]:
    if not records:
        return {}
    views = view_order(include_ce)
    chosen = None
    for rec in records:
        if rec["views"]["support_set"]["utility"] > rec["views"]["summary"]["utility"] + 0.05:
            chosen = rec
            break
    if chosen is None:
        chosen = records[0]
    legal_route = max(views, key=lambda view: chosen["views"][view]["utility"])
    legal = {
        "query_id": chosen["id"],
        "cell_id": f"musique_{safe_id(chosen['id'])}",
        "tier": "B1_structured",
        "cost_menu": "musique-structured-v0-op",
        "ranked_views": list(dict.fromkeys([legal_route, "decomp_title", "paragraph", "support_set", "full_context"])),
        "route": legal_route,
    }
    illegal = {
        "query_id": chosen["id"],
        "cell_id": f"musique_{safe_id(chosen['id'])}",
        "tier": "B1_structured",
        "cost_menu": "musique-structured-v0-op",
        "route": "support_set",
        "is_supporting": chosen["support_idx"],
        "answer": chosen["answer"],
        "decomposition_answers": chosen["decomp_answers"],
    }
    return {
        "visible_fields": {
            "query_id": chosen["id"],
            "question": chosen["question"],
            "decomposition_questions": chosen["decomp_questions"],
            "declared_views": views,
            "visible_titles": chosen["views"]["summary"]["ranked_titles"],
            "released_state": "question, decomposition-question text, paragraph-title sketch, and coarse paragraph metadata; support labels, answers, and unpaid CE/full scores are hidden",
        },
        "legal_action": {
            "submitted_jsonl": legal,
            "charged_cost": VIEW_COST[legal_route],
            "scored_utility": chosen["views"][legal_route]["utility"],
        },
        "hidden_evaluator_fields": {
            "support_idx": chosen["support_idx"],
            "support_titles": chosen["support_titles"],
            "answer": chosen["answer"],
            "decomposition_answers": chosen["decomp_answers"],
            "decomposition_support_idx": chosen["decomp_support_idx"],
        },
        "illegal_variant": {
            "submitted_jsonl": illegal,
            "rejection_reason": "support labels, answers, and decomposition answers are evaluator-only fields.",
        },
        "scope_note": "This is structured evidence acquisition over MuSiQue paragraphs, not answer generation.",
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = "ce" if not args.no_ce else "noce"
    stem = f"musique_structured_evidence_{args.max_rows}_{suffix}"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    payload = dict(result)
    payload["protocol_trace"] = build_trace(records, include_ce=not args.no_ce)
    payload["examples"] = [
        {
            "id": rec["id"],
            "question": rec["question"],
            "type": rec["type"],
            "decomposition_questions": rec["decomp_questions"],
            "support_idx": rec["support_idx"],
            "support_titles": rec["support_titles"],
            "oracle_view": rec["oracle_view"],
            "view_utility": {v: round(rec["views"][v]["utility"], 4) for v in view_order(not args.no_ce)},
            "top_paragraphs": {v: rec["views"][v]["ranked_idx"] for v in view_order(not args.no_ce)},
        }
        for rec in records[: args.example_rows]
    ]
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)

    policy_cols = ["policy", "utility", "raw_ndcg", "support_recall", "support_f1", "support_title_recall", "cost", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "support_recall", "support_f1", "support_title_recall", "cost", "regret", "diff_vs_best_control"]
    tier_cols = ["policy", "feature_tier", "utility", "cost", "regret", "diff_vs_best_fixed", "gap_closed", "view_share"]
    type_cols = ["type", "n", "best_fixed", "fixed_utility", "oracle_utility", "oracle_gap"]
    md = [
        "# MuSiQue Structured Evidence Acquisition Audit",
        "",
        "Candidate Protocol B structured-evidence replication over MuSiQue paragraphs. A method sees the question, decomposition-question text, paragraph-title sketch, and coarse metadata, then may buy paragraph, multi-paragraph support-set, full-context, or optional CE evidence before evaluator-held support labels are scored. This is evidence acquisition, not answer generation.",
        "",
        f"- Status: {result['status']}",
        f"- Rows: {result['n_rows']} ({result['split_sizes']})",
        f"- Split: `{result['split']}`; top-k: {result['topk']}; lambda: {result['lambda_cost']}",
        f"- Oracle view share: {result['oracle_view_share']}",
        "",
        "## Fixed and adaptive policies",
        "",
        markdown_table(result["policy_rows"], policy_cols),
        "",
        "## Randomization and decomposition controls",
        "",
        markdown_table(result["control_rows"], control_cols),
        "",
        "## Feature-tier learner readout",
        "",
        markdown_table(result["feature_tier_rows"], tier_cols),
        "",
        "## Hop-type oracle headroom",
        "",
        markdown_table(result["type_rows"], type_cols),
        "",
        "## Repeated split stability",
        "",
        "```json",
        json.dumps(result["repeat_summary"], indent=2),
        "```",
        "",
        "## Protocol B legal/illegal trace",
        "",
        "```json",
        json.dumps(payload["protocol_trace"], indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_MUSIQUE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-ce", action="store_true")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--max-doc-chars", type=int, default=1800)
    parser.add_argument("--example-rows", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = load_musique_rows(args.data_dir, args.max_rows, args.split)
    records = score_records(rows, args)
    result = summarize(records, args)
    json_path, md_path = write_outputs(result, records, args)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "n_rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
