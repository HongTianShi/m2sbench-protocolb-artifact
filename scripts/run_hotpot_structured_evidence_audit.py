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
    deterministic_rng,
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
DEFAULT_HOTPOT = Path(os.environ.get("M2SBENCH_HOTPOT_DIR", "PATH_TO_HOTPOT_HF_DISTRACTOR_EXPORT"))

BASE_VIEW_ORDER = ["summary", "title_shortlist", "paragraph", "sentence", "full_context"]
VIEW_COST = {
    "summary": 0.0,
    "title_shortlist": 0.04,
    "paragraph": 0.16,
    "sentence": 0.28,
    "full_context": 0.55,
    "ce": 0.95,
}
CONTROL_ORDER_BASE = [
    "title_only",
    "title_shortlist",
    "real_paragraph",
    "real_sentence",
    "shuffled_title",
    "random_paragraph",
    "random_sentence",
    "full_context",
]
CONTROL_COST = {
    "title_only": VIEW_COST["summary"],
    "title_shortlist": VIEW_COST["title_shortlist"],
    "real_paragraph": VIEW_COST["paragraph"],
    "real_sentence": VIEW_COST["sentence"],
    "shuffled_title": VIEW_COST["summary"],
    "random_paragraph": VIEW_COST["paragraph"],
    "random_sentence": VIEW_COST["sentence"],
    "full_context": VIEW_COST["full_context"],
    "ce": VIEW_COST["ce"],
}
FEATURE_TIERS = ["B0_title", "B1_paragraph_meta", "B1_sentence_meta", "B1_context_sketch"]


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


def load_hotpot_rows(data_dir: Path, max_rows: int, split: str) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(data_dir))[split]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if len(rows) >= max_rows:
            break
        titles = [str(t) for t in row["context"]["title"]]
        sentences = [[str(s) for s in sent_list] for sent_list in row["context"]["sentences"]]
        contexts = [{"title": title, "sentences": sent_list} for title, sent_list in zip(titles, sentences)]
        support = row["supporting_facts"]
        support_titles = {norm_title(t) for t in support.get("title", [])}
        support_pairs = {
            (norm_title(t), int(sid))
            for t, sid in zip(support.get("title", []), support.get("sent_id", []))
            if str(t).strip()
        }
        if not support_titles:
            continue
        rows.append(
            {
                "id": str(row.get("id") or f"hotpot_{idx}"),
                "question": str(row["question"]),
                "answer": str(row.get("answer", "")),
                "type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
                "contexts": contexts,
                "support_titles": support_titles,
                "support_pairs": support_pairs,
            }
        )
    return rows


def support_sentence_f1(pred_pairs: list[tuple[str, int]], gold_pairs: set[tuple[str, int]], k: int) -> float:
    if not gold_pairs:
        return 0.0
    pred = {(norm_title(t), int(sid)) for t, sid in pred_pairs[:k]}
    hit = len(pred & gold_pairs)
    precision = hit / max(1, min(k, len(pred_pairs)))
    recall = hit / len(gold_pairs)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def rank_sentences(question: str, row: dict[str, Any]) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    sent_titles: list[str] = []
    sent_pairs: list[tuple[str, int]] = []
    sent_texts: list[str] = []
    for ctx in row["contexts"]:
        title = ctx["title"]
        for sid, sent in enumerate(ctx["sentences"]):
            sent_titles.append(title)
            sent_pairs.append((title, sid))
            sent_texts.append(f"{title}. {sent}")
    sent_scores = bm25_scores(question, sent_texts)
    order = sorted(range(len(sent_pairs)), key=lambda i: (-float(sent_scores[i]), norm_title(sent_titles[i]), sent_pairs[i][1]))
    ranked_pairs = [sent_pairs[i] for i in order]
    title_scores: dict[str, float] = defaultdict(float)
    for i, score in enumerate(sent_scores):
        title = sent_titles[i]
        title_scores[title] = max(title_scores[title], float(score))
    scores = np.asarray([title_scores[ctx["title"]] for ctx in row["contexts"]], dtype=np.float32)
    return ranked_pairs, sent_scores, scores


def title_level_metrics(
    row: dict[str, Any],
    view: str,
    ranked_titles: list[str],
    scores: np.ndarray,
    pred_pairs: list[tuple[str, int]],
    lambda_cost: float,
    cost: float,
    topk: int,
) -> dict[str, Any]:
    gold_titles = set(row["support_titles"])
    return {
        "view": view,
        "ndcg": float(ndcg_at_k(ranked_titles, gold_titles, topk)),
        "support_recall": float(recall_at_k(ranked_titles, gold_titles, topk)),
        "support_f1": float(f1_at_k(ranked_titles, gold_titles, topk)),
        "support_sentence_f1": float(support_sentence_f1(pred_pairs, row["support_pairs"], topk)),
        "utility": float(ndcg_at_k(ranked_titles, gold_titles, topk) - lambda_cost * cost),
        "cost": cost,
        "score_top": float(np.max(scores)) if len(scores) else 0.0,
        "score_margin": margin(scores),
        "score_entropy": entropy_from_scores(scores),
        "ranked_titles": ranked_titles[:topk],
        "predicted_support_pairs": pred_pairs[:topk],
    }


def score_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ce_scores_by_id: dict[str, np.ndarray] = {}
    if not args.no_ce:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(args.ce_model, device=device)
        pairs: list[tuple[str, str]] = []
        row_slices: list[tuple[str, int, int]] = []
        cursor = 0
        for row in rows:
            start = cursor
            for ctx in row["contexts"]:
                text = " ".join(ctx["sentences"])[: args.max_doc_chars]
                pairs.append((row["question"], f"{ctx['title']}. {text}"))
                cursor += 1
            row_slices.append((row["id"], start, cursor))
        scores = model.predict(pairs, batch_size=args.ce_batch_size, show_progress_bar=True)
        scores = np.asarray(scores, dtype=np.float32)
        for row_id, start, end in row_slices:
            ce_scores_by_id[row_id] = scores[start:end]

    records: list[dict[str, Any]] = []
    include_ce = not args.no_ce
    views = view_order(include_ce)
    controls = control_order(include_ce)
    for row in rows:
        titles = [ctx["title"] for ctx in row["contexts"]]
        title_texts = titles
        first_sent_texts = [f"{ctx['title']}. {(ctx['sentences'][0] if ctx['sentences'] else '')}" for ctx in row["contexts"]]
        para_texts = [f"{ctx['title']}. {' '.join(ctx['sentences'][: args.paragraph_sentences])}" for ctx in row["contexts"]]
        full_texts = [f"{ctx['title']}. {' '.join(ctx['sentences'])}" for ctx in row["contexts"]]

        summary_scores = bm25_scores(row["question"], title_texts)
        overlap = np.asarray([title_overlap_score(row["question"], t) for t in titles], dtype=np.float32)
        title_shortlist_scores = summary_scores + 0.60 * overlap
        paragraph_scores = bm25_scores(row["question"], para_texts) + 0.20 * title_shortlist_scores
        sent_pairs, _, sentence_title_scores = rank_sentences(row["question"], row)
        sentence_scores = sentence_title_scores + 0.15 * title_shortlist_scores
        full_scores = bm25_scores(row["question"], full_texts) + 0.10 * paragraph_scores
        ce_scores = ce_scores_by_id.get(row["id"], full_scores.copy())

        rng = deterministic_rng(row["id"], args.seed, "hotpot-controls")
        shuffled_title_scores = rng.permutation(title_shortlist_scores) if len(title_shortlist_scores) else title_shortlist_scores
        random_paragraph_scores = random_matched_scores(paragraph_scores, rng)
        random_sentence_scores = random_matched_scores(sentence_scores, rng)

        no_sentence_pairs = [(title, 0) for title in rank_from_scores(titles, summary_scores)]
        title_pair_rank = [(title, 0) for title in rank_from_scores(titles, title_shortlist_scores)]
        para_pair_rank = [(title, 0) for title in rank_from_scores(titles, paragraph_scores)]
        full_pair_rank = [(title, 0) for title in rank_from_scores(titles, full_scores)]
        ce_pair_rank = [(title, 0) for title in rank_from_scores(titles, ce_scores)]

        view_defs: dict[str, tuple[np.ndarray, list[tuple[str, int]]]] = {
            "summary": (summary_scores, no_sentence_pairs),
            "title_shortlist": (title_shortlist_scores, title_pair_rank),
            "paragraph": (paragraph_scores, para_pair_rank),
            "sentence": (sentence_scores, sent_pairs),
            "full_context": (full_scores, full_pair_rank),
        }
        if include_ce:
            view_defs["ce"] = (ce_scores, ce_pair_rank)
        control_defs: dict[str, tuple[np.ndarray, list[tuple[str, int]]]] = {
            "title_only": (summary_scores, no_sentence_pairs),
            "title_shortlist": (title_shortlist_scores, title_pair_rank),
            "real_paragraph": (paragraph_scores, para_pair_rank),
            "real_sentence": (sentence_scores, sent_pairs),
            "shuffled_title": (shuffled_title_scores, [(title, 0) for title in rank_from_scores(titles, shuffled_title_scores)]),
            "random_paragraph": (random_paragraph_scores, [(title, 0) for title in rank_from_scores(titles, random_paragraph_scores)]),
            "random_sentence": (random_sentence_scores, [(title, 0) for title in rank_from_scores(titles, random_sentence_scores)]),
            "full_context": (full_scores, full_pair_rank),
        }
        if include_ce:
            control_defs["ce"] = (ce_scores, ce_pair_rank)

        view_records = {}
        for view in views:
            scores, pred_pairs = view_defs[view]
            ranked = rank_from_scores(titles, scores)
            view_records[view] = title_level_metrics(
                row, view, ranked, scores, pred_pairs, args.lambda_cost, VIEW_COST[view], args.topk
            )

        control_records = {}
        for control in controls:
            scores, pred_pairs = control_defs[control]
            ranked = rank_from_scores(titles, scores)
            control_records[control] = title_level_metrics(
                row, control, ranked, scores, pred_pairs, args.lambda_cost, CONTROL_COST[control], args.topk
            )

        oracle_view = max(views, key=lambda v: view_records[v]["utility"])
        sent_lens = [len(tokens(sent)) for ctx in row["contexts"] for sent in ctx["sentences"]]
        doc_lens = [len(tokens(" ".join(ctx["sentences"]))) for ctx in row["contexts"]]
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "support_titles": sorted(row["support_titles"]),
                "support_pairs": sorted(list(row["support_pairs"])),
                "n_context": len(titles),
                "n_sentences": sum(len(ctx["sentences"]) for ctx in row["contexts"]),
                "question_len": len(tokens(row["question"])),
                "context_stats": {
                    "mean_doc_len": float(np.mean(doc_lens)) if doc_lens else 0.0,
                    "std_doc_len": float(np.std(doc_lens)) if doc_lens else 0.0,
                    "max_doc_len": float(np.max(doc_lens)) if doc_lens else 0.0,
                    "mean_sent_len": float(np.mean(sent_lens)) if sent_lens else 0.0,
                    "std_sent_len": float(np.std(sent_lens)) if sent_lens else 0.0,
                    "max_sent_len": float(np.max(sent_lens)) if sent_lens else 0.0,
                    "support_title_count": float(len(row["support_titles"])),
                },
                "views": view_records,
                "controls": control_records,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def fixed_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str], views: list[str]) -> dict[str, Any]:
    utils, regrets, ndcgs, recalls, supp_f1s, sent_f1s, costs, chosen = [], [], [], [], [], [], [], []
    for pos, idx in enumerate(indices):
        rec = records[int(idx)]
        view = route[pos] if isinstance(route, list) else route
        vr = rec["views"][view]
        utils.append(vr["utility"])
        regrets.append(rec["oracle_utility"] - vr["utility"])
        ndcgs.append(vr["ndcg"])
        recalls.append(vr["support_recall"])
        supp_f1s.append(vr["support_f1"])
        sent_f1s.append(vr["support_sentence_f1"])
        costs.append(vr["cost"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utils)) if utils else 0.0,
        "raw_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "support_recall": float(np.mean(recalls)) if recalls else 0.0,
        "support_f1": float(np.mean(supp_f1s)) if supp_f1s else 0.0,
        "support_sentence_f1": float(np.mean(sent_f1s)) if sent_f1s else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
        "regret": float(np.mean(regrets)) if regrets else 0.0,
        "ce_buy": float(counts["ce"] / max(1, len(chosen))),
        "view_share": {v: round(counts[v] / max(1, len(chosen)), 4) for v in views if counts[v]},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray, tier: str, views: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    rows, y, keys = [], [], []
    type_values = sorted({str(rec["type"]) for rec in records})
    level_values = sorted({str(rec["level"]) for rec in records})
    for idx in indices:
        rec = records[int(idx)]
        summary = rec["views"]["summary"]
        title_shortlist = rec["views"]["title_shortlist"]
        stats = rec["context_stats"]
        base = [
            rec["question_len"],
            rec["n_context"],
            rec["n_sentences"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
            title_shortlist["score_top"],
            title_shortlist["score_margin"],
            title_shortlist["score_entropy"],
        ]
        if tier in {"B1_paragraph_meta", "B1_sentence_meta", "B1_context_sketch"}:
            base += [stats["mean_doc_len"], stats["std_doc_len"], stats["max_doc_len"]]
        if tier in {"B1_sentence_meta", "B1_context_sketch"}:
            base += [stats["mean_sent_len"], stats["std_sent_len"], stats["max_sent_len"]]
        if tier == "B1_context_sketch":
            base += [1.0 if rec["type"] == t else 0.0 for t in type_values]
            base += [1.0 if rec["level"] == lvl else 0.0 for lvl in level_values]

        for view_i, view in enumerate(views):
            vr = rec["views"][view]
            view_profile = [
                vr["cost"],
                1.0 if view == "title_shortlist" else 0.0,
                1.0 if view == "paragraph" else 0.0,
                1.0 if view == "sentence" else 0.0,
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
    from sklearn.linear_model import Ridge

    tier_rows = []
    best_model_name = None
    best_tier = None
    best_dev = -1e9
    best_model = None
    for tier in FEATURE_TIERS:
        models = {
            f"ridge_{tier}": Ridge(alpha=1.0),
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
    assert best_model is not None and best_model_name is not None and best_tier is not None
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
    candidates = []
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
        utils, ndcgs, recalls, support_f1s, sentence_f1s, costs, regrets = [], [], [], [], [], [], []
        for idx in test_idx:
            rec = records[int(idx)]
            row = rec["controls"][control]
            utils.append(row["utility"])
            ndcgs.append(row["ndcg"])
            recalls.append(row["support_recall"])
            support_f1s.append(row["support_f1"])
            sentence_f1s.append(row["support_sentence_f1"])
            costs.append(row["cost"])
            regrets.append(rec["oracle_utility"] - row["utility"])
        control_rows.append(
            {
                "control": control,
                "utility": float(np.mean(utils)),
                "raw_ndcg": float(np.mean(ndcgs)),
                "support_recall": float(np.mean(recalls)),
                "support_f1": float(np.mean(support_f1s)),
                "support_sentence_f1": float(np.mean(sentence_f1s)),
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
        "task": "hotpot_structured_evidence",
        "dataset": "HotpotQA distractor",
        "status": "candidate_structured_evidence_slice",
        "n_rows": len(records),
        "split": args.split,
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "view_costs": {v: VIEW_COST[v] for v in views},
        "views": {
            "summary": "question plus context-title sketch only",
            "title_shortlist": "paid title shortlist/rerank over released titles",
            "paragraph": f"paragraph text truncated to the first {args.paragraph_sentences} sentence(s)",
            "sentence": "candidate sentence scoring inside context titles",
            "full_context": "all context paragraphs",
            "ce": "cross-encoder rerank over purchased paragraph evidence",
        },
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
        if rec["views"]["sentence"]["utility"] > rec["views"]["summary"]["utility"] + 0.05:
            chosen = rec
            break
    if chosen is None:
        chosen = records[0]
    legal_route = max(views, key=lambda view: chosen["views"][view]["utility"])
    legal = {
        "query_id": chosen["id"],
        "cell_id": f"hotpot_{safe_id(chosen['id'])}",
        "tier": "B1_structured",
        "cost_menu": "hotpot-structured-v0-op",
        "ranked_views": list(dict.fromkeys([legal_route, "title_shortlist", "paragraph", "sentence", "full_context"])),
        "route": legal_route,
    }
    illegal = {
        "query_id": chosen["id"],
        "cell_id": f"hotpot_{safe_id(chosen['id'])}",
        "tier": "B1_structured",
        "cost_menu": "hotpot-structured-v0-op",
        "route": "sentence",
        "supporting_facts": chosen["support_pairs"],
        "answer": chosen["answer"],
    }
    return {
        "visible_fields": {
            "query_id": chosen["id"],
            "question": chosen["question"],
            "declared_views": views,
            "visible_titles": chosen["views"]["summary"]["ranked_titles"],
            "released_state": "question plus context-title sketch; support facts, answers, and unpaid sentence/CE scores are hidden",
        },
        "legal_action": {
            "submitted_jsonl": legal,
            "charged_cost": VIEW_COST[legal_route],
            "scored_utility": chosen["views"][legal_route]["utility"],
        },
        "hidden_evaluator_fields": {
            "support_titles": chosen["support_titles"],
            "supporting_facts": chosen["support_pairs"],
            "answer": chosen["answer"],
        },
        "illegal_variant": {
            "submitted_jsonl": illegal,
            "rejection_reason": "supporting_facts and answers are evaluator-only fields.",
        },
        "scope_note": "This is structured evidence acquisition over HotpotQA distractor contexts, not answer generation.",
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = "ce" if not args.no_ce else "noce"
    stem = f"hotpot_structured_evidence_{args.max_rows}_{suffix}"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    payload = dict(result)
    payload["protocol_trace"] = build_trace(records, include_ce=not args.no_ce)
    payload["examples"] = [
        {
            "id": rec["id"],
            "question": rec["question"],
            "type": rec["type"],
            "level": rec["level"],
            "support_titles": rec["support_titles"],
            "oracle_view": rec["oracle_view"],
            "view_utility": {v: round(rec["views"][v]["utility"], 4) for v in view_order(not args.no_ce)},
            "top_titles": {v: rec["views"][v]["ranked_titles"] for v in view_order(not args.no_ce)},
        }
        for rec in records[: args.example_rows]
    ]
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)

    policy_cols = ["policy", "utility", "raw_ndcg", "support_recall", "support_f1", "support_sentence_f1", "cost", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "support_recall", "support_f1", "support_sentence_f1", "cost", "regret", "diff_vs_best_control"]
    type_cols = ["type", "n", "best_fixed", "fixed_utility", "oracle_utility", "oracle_gap"]
    tier_cols = ["policy", "feature_tier", "utility", "cost", "regret", "diff_vs_best_fixed", "gap_closed", "view_share"]
    md = [
        "# HotpotQA Structured Evidence Acquisition Audit",
        "",
        "Candidate Protocol B structured-evidence slice over HotpotQA distractor contexts. A method sees the question and title sketch, then may buy title-shortlist, paragraph, sentence, full-context, or optional CE evidence before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.",
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
        "## Randomization controls",
        "",
        markdown_table(result["control_rows"], control_cols),
        "",
        "## Feature-tier learner readout",
        "",
        markdown_table(result["feature_tier_rows"], tier_cols),
        "",
        "## Question-type oracle headroom",
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_HOTPOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--paragraph-sentences", type=int, default=2)
    parser.add_argument("--no-ce", action="store_true")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--max-doc-chars", type=int, default=1800)
    parser.add_argument("--example-rows", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = load_hotpot_rows(args.data_dir, args.max_rows, args.split)
    records = score_records(rows, args)
    result = summarize(records, args)
    json_path, md_path = write_outputs(result, records, args)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "n_rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
