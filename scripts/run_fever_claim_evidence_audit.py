from __future__ import annotations

import argparse
import json
import os
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
    random_matched_scores,
    rank_from_scores,
    recall_at_k,
    split_indices,
    token_set,
    tokens,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_FEVER = Path(os.environ.get("M2SBENCH_FEVER_TRAIN_JSONL", "PATH_TO_FEVER_OFFICIAL_TRAIN_JSONL"))

VIEW_ORDER = ["summary", "title_page", "evidence_sentence", "full_evidence_page"]
VIEW_COST = {
    "summary": 0.0,
    "title_page": 0.05,
    "evidence_sentence": 0.22,
    "full_evidence_page": 0.72,
}
CONTROL_ORDER = [
    "claim_only",
    "real_title_page",
    "real_evidence_sentence",
    "full_evidence_page",
    "shuffled_evidence_sentence",
    "wrong_evidence_sentence",
    "same_count_random",
    "title_frequency_matched_random",
    "high_frequency_page",
]
CONTROL_COST = {
    "claim_only": VIEW_COST["summary"],
    "real_title_page": VIEW_COST["title_page"],
    "real_evidence_sentence": VIEW_COST["evidence_sentence"],
    "full_evidence_page": VIEW_COST["full_evidence_page"],
    "shuffled_evidence_sentence": VIEW_COST["evidence_sentence"],
    "wrong_evidence_sentence": VIEW_COST["evidence_sentence"],
    "same_count_random": VIEW_COST["evidence_sentence"],
    "title_frequency_matched_random": VIEW_COST["title_page"],
    "high_frequency_page": VIEW_COST["title_page"],
}
FEATURE_TIERS = ["B0_claim", "B1_title_meta", "B1_sentence_handle"]


def norm_title(title: str) -> str:
    return str(title).replace("_", " ").strip().lower()


def title_display(title: str, sid: int | None = None) -> str:
    base = str(title).replace("_", " ").strip()
    if sid is None:
        return base
    return f"{base} sentence {sid}"


def title_overlap_score(claim: str, title: str) -> float:
    q = token_set(claim)
    t = token_set(title_display(title))
    if not t:
        return 0.0
    score = len(q & t) / max(1, len(t))
    if norm_title(title) and norm_title(title) in claim.lower():
        score += 1.0
    return float(score)


def extract_handles(evidence: Any) -> list[tuple[str, int]]:
    handles: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    if not isinstance(evidence, list):
        return handles
    for group in evidence:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, list) or len(item) < 4:
                continue
            title, sid = item[2], item[3]
            if title is None or sid is None:
                continue
            try:
                key = (str(title), int(sid))
            except (TypeError, ValueError):
                continue
            if key not in seen:
                handles.append(key)
                seen.add(key)
    return handles


def load_fever_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max_rows:
                break
            raw = json.loads(line)
            handles = extract_handles(raw.get("evidence"))
            if not handles:
                continue
            rows.append(
                {
                    "id": str(raw.get("id")),
                    "claim": str(raw.get("claim", "")),
                    "label": str(raw.get("label", "")),
                    "verifiable": str(raw.get("verifiable", "")),
                    "evidence_handles": handles,
                    "evidence_titles": sorted({title for title, _ in handles}),
                }
            )
    return rows


def evidence_metrics(
    view: str,
    ranked_ids: list[str],
    ranked_titles: list[str],
    scores: np.ndarray,
    gold_ids: set[str],
    gold_titles: set[str],
    lambda_cost: float,
    cost: float,
    topk: int,
) -> dict[str, Any]:
    ndcg = ndcg_at_k(ranked_ids, gold_ids, topk)
    recall = recall_at_k(ranked_ids, gold_ids, topk)
    support_f1 = f1_at_k(ranked_ids, gold_ids, topk)
    title_recall = recall_at_k(ranked_titles, gold_titles, topk)
    title_f1 = f1_at_k(ranked_titles, gold_titles, topk)
    return {
        "view": view,
        "ndcg": float(ndcg),
        "support_recall": float(recall),
        "support_f1": float(support_f1),
        "support_title_recall": float(title_recall),
        "support_title_f1": float(title_f1),
        "utility": float(ndcg - lambda_cost * cost),
        "cost": float(cost),
        "score_top": float(np.max(scores)) if len(scores) else 0.0,
        "score_margin": margin(scores),
        "score_entropy": entropy_from_scores(scores),
        "ranked_ids": ranked_ids[:topk],
        "ranked_titles": ranked_titles[:topk],
    }


def build_candidates(
    row: dict[str, Any],
    global_handles: list[tuple[str, int]],
    title_freq: Counter[str],
    rng: np.random.Generator,
    candidate_pool: int,
) -> tuple[list[str], list[str], list[tuple[str, int]], set[str], set[str]]:
    own = [(f"g{i}", title, sid) for i, (title, sid) in enumerate(row["evidence_handles"])]
    own_set = {(title, sid) for title, sid in row["evidence_handles"]}
    distractor_pool = [(title, sid) for title, sid in global_handles if (title, sid) not in own_set]
    n_distractors = max(0, candidate_pool - len(own))
    distractors: list[tuple[str, str, int]] = []
    if distractor_pool and n_distractors:
        weights = np.asarray([1.0 / max(1, title_freq[t]) for t, _ in distractor_pool], dtype=np.float64)
        weights = weights / weights.sum()
        take = rng.choice(len(distractor_pool), size=min(n_distractors, len(distractor_pool)), replace=False, p=weights)
        distractors = [(f"d{j}", distractor_pool[int(i)][0], distractor_pool[int(i)][1]) for j, i in enumerate(take)]
    candidates = own + distractors
    cand_ids = [cid for cid, _, _ in candidates]
    cand_titles = [title for _, title, _ in candidates]
    cand_handles = [(title, sid) for _, title, sid in candidates]
    gold_ids = {cid for cid, _, _ in own}
    gold_titles = {norm_title(title) for title, _ in row["evidence_handles"]}
    return cand_ids, cand_titles, cand_handles, gold_ids, gold_titles


def score_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    global_handles = [handle for row in rows for handle in row["evidence_handles"]]
    title_freq: Counter[str] = Counter(title for title, _ in global_handles)
    max_freq = max(title_freq.values()) if title_freq else 1

    records: list[dict[str, Any]] = []
    for row in rows:
        cand_ids, cand_titles, cand_handles, gold_ids, gold_titles = build_candidates(
            row, global_handles, title_freq, rng, args.candidate_pool
        )
        title_texts = [title_display(title) for title in cand_titles]
        handle_texts = [title_display(title, sid) for title, sid in cand_handles]
        claim = row["claim"]

        overlap = np.asarray([title_overlap_score(claim, title) for title in cand_titles], dtype=np.float32)
        summary_scores = bm25_scores(claim, title_texts) + 0.25 * overlap
        title_scores = bm25_scores(claim, title_texts) + 0.45 * overlap
        evidence_scores = bm25_scores(claim, handle_texts) + 0.35 * overlap
        full_scores = np.asarray(evidence_scores, dtype=np.float32).copy()
        for pos, cid in enumerate(cand_ids):
            if cid in gold_ids:
                full_scores[pos] += float(args.full_evidence_boost)

        score_by_view = {
            "summary": summary_scores,
            "title_page": title_scores,
            "evidence_sentence": evidence_scores,
            "full_evidence_page": full_scores,
        }

        view_records: dict[str, dict[str, Any]] = {}
        for view in VIEW_ORDER:
            scores = np.asarray(score_by_view[view], dtype=np.float32)
            ranked_ids = rank_from_scores(cand_ids, scores)
            id_to_title = {cid: norm_title(title) for cid, title in zip(cand_ids, cand_titles)}
            ranked_titles = [id_to_title[cid] for cid in ranked_ids]
            view_records[view] = evidence_metrics(
                view,
                ranked_ids,
                ranked_titles,
                scores,
                gold_ids,
                gold_titles,
                args.lambda_cost,
                VIEW_COST[view],
                args.topk,
            )

        shuffled_scores = np.asarray(evidence_scores, dtype=np.float32).copy()
        rng.shuffle(shuffled_scores)
        wrong_scores = random_matched_scores(evidence_scores, rng)
        random_scores = rng.random(len(cand_ids), dtype=np.float32)
        freq_scores = np.asarray([title_freq[t] / max_freq for t in cand_titles], dtype=np.float32)
        high_freq_scores = np.asarray([np.log1p(title_freq[t]) for t in cand_titles], dtype=np.float32)
        control_scores = {
            "claim_only": summary_scores,
            "real_title_page": title_scores,
            "real_evidence_sentence": evidence_scores,
            "full_evidence_page": full_scores,
            "shuffled_evidence_sentence": shuffled_scores,
            "wrong_evidence_sentence": wrong_scores,
            "same_count_random": random_scores,
            "title_frequency_matched_random": freq_scores,
            "high_frequency_page": high_freq_scores,
        }

        controls: dict[str, dict[str, Any]] = {}
        for control in CONTROL_ORDER:
            scores = np.asarray(control_scores[control], dtype=np.float32)
            ranked_ids = rank_from_scores(cand_ids, scores)
            id_to_title = {cid: norm_title(title) for cid, title in zip(cand_ids, cand_titles)}
            ranked_titles = [id_to_title[cid] for cid in ranked_ids]
            controls[control] = evidence_metrics(
                control,
                ranked_ids,
                ranked_titles,
                scores,
                gold_ids,
                gold_titles,
                args.lambda_cost,
                CONTROL_COST[control],
                args.topk,
            )

        oracle_view = max(VIEW_ORDER, key=lambda v: view_records[v]["utility"])
        title_lens = [len(tokens(title_display(title))) for title in cand_titles]
        records.append(
            {
                "id": row["id"],
                "claim": row["claim"],
                "label": row["label"],
                "verifiable": row["verifiable"],
                "evidence_handles": row["evidence_handles"],
                "candidate_pool_size": len(cand_ids),
                "n_evidence_handles": len(row["evidence_handles"]),
                "n_evidence_titles": len(row["evidence_titles"]),
                "claim_len": len(tokens(row["claim"])),
                "context_stats": {
                    "mean_title_len": float(np.mean(title_lens)) if title_lens else 0.0,
                    "std_title_len": float(np.std(title_lens)) if title_lens else 0.0,
                    "max_title_frequency": float(max(title_freq[t] for t in cand_titles)) if cand_titles else 0.0,
                    "gold_title_frequency_mean": float(np.mean([title_freq[t] for t, _ in row["evidence_handles"]]))
                    if row["evidence_handles"]
                    else 0.0,
                },
                "views": view_records,
                "controls": controls,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def fixed_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str]) -> dict[str, Any]:
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
        "view_share": {v: round(counts[v] / max(1, len(chosen)), 4) for v in VIEW_ORDER if counts[v]},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray, tier: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    rows: list[list[float]] = []
    y: list[float] = []
    keys: list[tuple[int, str]] = []
    for idx in indices:
        rec = records[int(idx)]
        summary = rec["views"]["summary"]
        title_page = rec["views"]["title_page"]
        sent = rec["views"]["evidence_sentence"]
        stats = rec["context_stats"]
        base = [
            rec["claim_len"],
            rec["n_evidence_handles"],
            rec["n_evidence_titles"],
            rec["candidate_pool_size"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
        ]
        if tier in {"B1_title_meta", "B1_sentence_handle"}:
            base += [
                title_page["score_top"],
                title_page["score_margin"],
                title_page["score_entropy"],
                stats["mean_title_len"],
                stats["std_title_len"],
            ]
        if tier == "B1_sentence_handle":
            base += [
                sent["score_top"],
                sent["score_margin"],
                sent["score_entropy"],
                stats["max_title_frequency"],
                stats["gold_title_frequency_mean"],
            ]
        for view_i, view in enumerate(VIEW_ORDER):
            vr = rec["views"][view]
            rows.append(base + [1.0 if i == view_i else 0.0 for i in range(len(VIEW_ORDER))] + [vr["cost"]])
            y.append(vr["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def route_from_predictions(keys: list[tuple[int, str]], preds: np.ndarray) -> list[str]:
    by_idx: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (idx, view), pred in zip(keys, preds):
        by_idx[idx].append((view, float(pred)))
    return [max(by_idx[idx], key=lambda x: (x[1], -VIEW_ORDER.index(x[0])))[0] for idx in sorted(by_idx)]


def train_router(
    records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    tier_rows: list[dict[str, Any]] = []
    best_dev = -1e9
    best_name = ""
    best_tier = ""
    best_model: tuple[Any, str] | None = None
    for tier in FEATURE_TIERS:
        models = {
            f"rf_{tier}": RandomForestRegressor(n_estimators=180, min_samples_leaf=4, random_state=seed, n_jobs=-1),
            f"et_{tier}": ExtraTreesRegressor(n_estimators=240, min_samples_leaf=3, random_state=seed, n_jobs=-1),
            f"hgb_{tier}": HistGradientBoostingRegressor(max_iter=140, learning_rate=0.05, random_state=seed),
        }
        x_train, y_train, _ = make_features(records, train_idx, tier)
        x_dev, _, dev_keys = make_features(records, dev_idx, tier)
        x_test, _, test_keys = make_features(records, test_idx, tier)
        for name, model in models.items():
            model.fit(x_train, y_train)
            dev_route = route_from_predictions(dev_keys, model.predict(x_dev))
            dev_eval = fixed_policy(records, np.asarray(sorted(set(i for i, _ in dev_keys))), name, dev_route)
            test_route = route_from_predictions(test_keys, model.predict(x_test))
            test_eval = fixed_policy(records, np.asarray(sorted(set(i for i, _ in test_keys))), name, test_route)
            test_eval["feature_tier"] = tier
            test_eval["dev_utility"] = dev_eval["utility"]
            tier_rows.append(test_eval)
            if dev_eval["utility"] > best_dev:
                best_dev = dev_eval["utility"]
                best_name = name
                best_tier = tier
                best_model = (model, tier)
    if best_model is None:
        return fixed_policy(records, test_idx, "best_legal_router(empty)", "summary"), tier_rows
    model, tier = best_model
    x_test, _, test_keys = make_features(records, test_idx, tier)
    test_route = route_from_predictions(test_keys, model.predict(x_test))
    out = fixed_policy(records, np.asarray(sorted(set(i for i, _ in test_keys))), f"best_legal_router({best_name})", test_route)
    out["feature_tier"] = best_tier
    out["dev_utility"] = float(best_dev)
    out["routes"] = test_route
    return out, tier_rows


def train_threshold_gate(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    vals = np.asarray([records[int(i)]["views"]["summary"]["score_margin"] for i in train_idx], dtype=np.float64)
    thresholds = sorted(set(float(x) for x in np.quantile(vals, np.linspace(0.05, 0.95, 19)))) if len(vals) else [0.0]
    candidates: list[tuple[float, float, str, str]] = []
    for threshold in thresholds:
        for direction in ["high_summary", "low_summary"]:
            for rich in ["title_page", "evidence_sentence", "full_evidence_page"]:
                route = []
                for idx in dev_idx:
                    val = records[int(idx)]["views"]["summary"]["score_margin"]
                    choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
                    route.append("summary" if choose_summary else rich)
                dev_eval = fixed_policy(records, dev_idx, "dev_gate", route)
                candidates.append((dev_eval["utility"], threshold, direction, rich))
    if not candidates:
        return fixed_policy(records, test_idx, "threshold_gate(summary)", "summary")
    dev_utility, threshold, direction, rich = max(candidates, key=lambda x: x[0])
    route = []
    for idx in test_idx:
        val = records[int(idx)]["views"]["summary"]["score_margin"]
        choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
        route.append("summary" if choose_summary else rich)
    out = fixed_policy(records, test_idx, f"threshold_gate(summary,{rich})", route)
    out["gate_threshold"] = float(threshold)
    out["gate_direction"] = direction
    out["rich_view"] = rich
    out["dev_utility"] = float(dev_utility)
    return out


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    train_idx, dev_idx, test_idx = split_indices(len(records), args.seed)
    fixed_rows = [fixed_policy(records, test_idx, f"fixed_{view}", view) for view in VIEW_ORDER]
    best_fixed = max(fixed_rows, key=lambda row: row["utility"])
    gate = train_threshold_gate(records, train_idx, dev_idx, test_idx)
    learned, tier_rows = train_router(records, train_idx, dev_idx, test_idx, args.seed)
    best_adaptive = max([gate, learned], key=lambda row: row.get("dev_utility", -1e9))
    best_adaptive = dict(best_adaptive)
    best_adaptive["policy"] = "best_legal_adaptive"
    oracle_route = [records[int(i)]["oracle_view"] for i in test_idx]
    oracle = fixed_policy(records, test_idx, "oracle_route", oracle_route)
    denom = max(oracle["utility"] - best_fixed["utility"], 1e-12)
    for row in fixed_rows + [gate, learned, best_adaptive, oracle] + tier_rows:
        row["diff_vs_best_fixed"] = row["utility"] - best_fixed["utility"]
        row["gap_closed"] = (row["utility"] - best_fixed["utility"]) / denom

    repeat = []
    selected = []
    for r in range(args.repeats):
        tr, dv, te = split_indices(len(records), args.seed + 100 + r)
        fixed = [fixed_policy(records, te, f"fixed_{view}", view) for view in VIEW_ORDER]
        bf = max(fixed, key=lambda row: row["utility"])
        gate_r = train_threshold_gate(records, tr, dv, te)
        learned_r, _ = train_router(records, tr, dv, te, args.seed + r + 1)
        adaptive_r = max([gate_r, learned_r], key=lambda row: row.get("dev_utility", -1e9))
        repeat.append(adaptive_r["utility"] - bf["utility"])
        selected.append(adaptive_r["policy"])
    repeat_arr = np.asarray(repeat, dtype=np.float64)

    control_rows = []
    for control in CONTROL_ORDER:
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

    label_rows = []
    for label in sorted(set(rec["label"] for rec in records)):
        idxs = np.asarray([int(i) for i in test_idx if records[int(i)]["label"] == label])
        if not len(idxs):
            continue
        fixed = [fixed_policy(records, idxs, f"fixed_{view}", view) for view in VIEW_ORDER]
        bf = max(fixed, key=lambda row: row["utility"])
        oracle_label = fixed_policy(records, idxs, "oracle_route", [records[int(i)]["oracle_view"] for i in idxs])
        label_rows.append(
            {
                "label": label,
                "n": int(len(idxs)),
                "best_fixed": bf["policy"],
                "fixed_utility": bf["utility"],
                "oracle_utility": oracle_label["utility"],
                "oracle_gap": oracle_label["utility"] - bf["utility"],
            }
        )

    oracle_share = Counter(records[int(i)]["oracle_view"] for i in test_idx)
    return {
        "task": "fever_claim_evidence",
        "dataset": "FEVER official train JSONL",
        "status": "candidate_claim_verification_slice",
        "scope_note": "Exploratory claim-evidence handle acquisition over FEVER official JSONL; no wiki page text is included.",
        "n_rows": len(records),
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "candidate_pool": args.candidate_pool,
        "view_costs": VIEW_COST,
        "views": {
            "summary": "claim-only lexical/title sketch",
            "title_page": "paid candidate page-title handle view",
            "evidence_sentence": "paid evidence sentence-handle view over title/sentence-id candidates",
            "full_evidence_page": "expensive evidence-bundle handle view; the official JSONL does not include wiki sentence text",
        },
        "hidden_fields": ["gold evidence handles", "label", "unpaid evidence/full scores"],
        "policy_rows": fixed_rows + [gate, learned, best_adaptive, oracle],
        "best_fixed": best_fixed["policy"],
        "feature_tier_rows": sorted(tier_rows, key=lambda row: row["utility"], reverse=True)[:8],
        "control_rows": control_rows,
        "label_rows": label_rows,
        "oracle_view_share": {k: float(v / max(1, len(test_idx))) for k, v in oracle_share.items()},
        "repeat_summary": {
            "learned_minus_best_fixed_mean": float(np.mean(repeat_arr)) if len(repeat_arr) else 0.0,
            "ci95": [float(np.quantile(repeat_arr, 0.025)), float(np.quantile(repeat_arr, 0.975))] if len(repeat_arr) else [0.0, 0.0],
            "positive_share": float(np.mean(repeat_arr > 0)) if len(repeat_arr) else 0.0,
            "selected_policies": dict(Counter(selected)),
            "repeats": int(len(repeat_arr)),
        },
    }


def write_report(summary: dict[str, Any], args: argparse.Namespace, elapsed: float) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = f"{summary['n_rows']}_noce"
    md_path = REPORTS / f"fever_claim_evidence_{suffix}.md"
    json_path = REPORTS / f"fever_claim_evidence_{suffix}.json"
    payload = dict(summary)
    payload["runtime_s"] = elapsed
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def table(columns: list[str], body: list[list[Any]]) -> str:
        out = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in body:
            out.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(out)

    rows = []
    for row in summary["policy_rows"]:
        rows.append(
            [
                row["policy"],
                f"{row['utility']:.4f}",
                f"{row['raw_ndcg']:.4f}",
                f"{row['support_recall']:.4f}",
                f"{row['support_f1']:.4f}",
                f"{row['support_title_recall']:.4f}",
                f"{row['cost']:.4f}",
                f"{row['regret']:.4f}",
                f"{row.get('gap_closed', 0):.4f}",
                f"{row.get('diff_vs_best_fixed', 0):.4f}",
                row.get("view_share", {}),
            ]
        )
    control_rows = []
    for row in summary["control_rows"]:
        control_rows.append(
            [
                row["control"],
                f"{row['utility']:.4f}",
                f"{row['raw_ndcg']:.4f}",
                f"{row['support_recall']:.4f}",
                f"{row['support_f1']:.4f}",
                f"{row['support_title_recall']:.4f}",
                f"{row['cost']:.4f}",
                f"{row['regret']:.4f}",
                f"{row.get('diff_vs_best_control', 0):.4f}",
            ]
        )
    tier_rows = []
    for row in summary["feature_tier_rows"]:
        tier_rows.append(
            [
                row["policy"],
                row.get("feature_tier", ""),
                f"{row['utility']:.4f}",
                f"{row['cost']:.4f}",
                f"{row['regret']:.4f}",
                f"{row.get('diff_vs_best_fixed', 0):.4f}",
                f"{row.get('gap_closed', 0):.4f}",
                row.get("view_share", {}),
            ]
        )
    label_rows = []
    for row in summary["label_rows"]:
        label_rows.append(
            [
                row["label"],
                row["n"],
                row["best_fixed"],
                f"{row['fixed_utility']:.4f}",
                f"{row['oracle_utility']:.4f}",
                f"{row['oracle_gap']:.4f}",
            ]
        )
    lines = [
        "# FEVER Claim-Evidence Handle Audit",
        "",
        "Exploratory Protocol B claim-evidence slice over FEVER official JSONL. A method sees a claim and cheap title/entity overlap handles, then may buy page-title or evidence-sentence handles before evaluator-held evidence handles are scored. The official JSONL used here does not include wiki sentence text or full page text, so this is a handle-acquisition audit rather than a full FEVER retrieval benchmark.",
        "",
        f"- Status: {summary['status']}",
        f"- Rows: {summary['n_rows']} ({summary['split_sizes']})",
        f"- Top-k: {summary['topk']}; lambda: {summary['lambda_cost']}; candidate pool: {summary['candidate_pool']}",
        f"- Oracle view share: {summary['oracle_view_share']}",
        "",
        "## Fixed and adaptive policies",
        "",
        table(
            [
                "policy",
                "utility",
                "raw_ndcg",
                "support_recall",
                "support_f1",
                "title_recall",
                "cost",
                "regret",
                "gap_closed",
                "diff_vs_best_fixed",
                "view_share",
            ],
            rows,
        ),
        "",
        "## Randomization and shortcut controls",
        "",
        table(
            [
                "control",
                "utility",
                "raw_ndcg",
                "support_recall",
                "support_f1",
                "title_recall",
                "cost",
                "regret",
                "diff_vs_best_control",
            ],
            control_rows,
        ),
        "",
        "## Feature-tier learner readout",
        "",
        table(["policy", "feature_tier", "utility", "cost", "regret", "diff_vs_best_fixed", "gap_closed", "view_share"], tier_rows),
        "",
        "## Label-split oracle headroom",
        "",
        table(["label", "n", "best_fixed", "fixed_utility", "oracle_utility", "oracle_gap"], label_rows),
        "",
        "## Repeated split stability",
        "",
        "```json",
        json.dumps(summary["repeat_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Protocol B legal/illegal trace",
        "",
        "Legal route example:",
        "",
        "```json",
        json.dumps(
            {
                "query_id": "fever_75397",
                "cell_id": "claim_evidence_pool",
                "tier": "B0",
                "cost_menu": "fever-claim-v1-op",
                "ranked_views": ["title_page", "evidence_sentence", "full_evidence_page"],
                "route": "title_page",
            },
            indent=2,
        ),
        "```",
        "",
        "Invalid route example:",
        "",
        "```json",
        json.dumps(
            {
                "query_id": "fever_75397",
                "cell_id": "claim_evidence_pool",
                "tier": "B0",
                "cost_menu": "fever-claim-v1-op",
                "route": "evidence_sentence",
                "gold_evidence_handle": "Nikolaj_Coster-Waldau#7",
            },
            indent=2,
        ),
        "```",
        "",
        "Reason: `gold_evidence_handle` is evaluator-only before evidence is bought.",
        "",
        f"Runtime: {elapsed:.2f}s",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def write_pointer(summary: dict[str, Any], md_path: Path, json_path: Path) -> Path:
    pointer = REPORTS / "fever_claim_evidence.md"
    repeat = summary["repeat_summary"]
    policies = {row["policy"]: row for row in summary["policy_rows"]}
    best_fixed = max([row for row in summary["policy_rows"] if row["policy"].startswith("fixed_")], key=lambda row: row["utility"])
    adaptive = next(row for row in summary["policy_rows"] if row["policy"] == "best_legal_adaptive")
    oracle = next(row for row in summary["policy_rows"] if row["policy"] == "oracle_route")
    control_map = {row["control"]: row for row in summary["control_rows"]}
    text = f"""# FEVER Claim-Evidence Report Pointer

Purpose: exploratory claim-verification evidence-purchase slice over FEVER official JSONL. The artifact uses claim text plus title/sentence-id evidence handles available in the official file; it does not include wiki sentence text or full page text.

Canonical report:

- `{md_path.relative_to(ROOT).as_posix()}`
- `{json_path.relative_to(ROOT).as_posix()}`

Scope: FEVER is not part of the official leaderboard core and is not used as a main-paper validation row. It is retained as a final exploratory claim-evidence check because official evidence handles allow a narrow Protocol B readout, while lexical title shortcuts can be strong.

Canonical readout: on {summary['n_rows']} verifiable rows, best fixed {best_fixed['policy'].replace('fixed_', '')} reaches {best_fixed['utility']:.4f} utility, valid adaptive routing reaches {adaptive['utility']:.4f}, and the oracle reaches {oracle['utility']:.4f}. Repeated adaptive-minus-fixed is {repeat['learned_minus_best_fixed_mean']:+.4f} with 95% interval [{repeat['ci95'][0]:+.4f},{repeat['ci95'][1]:+.4f}] and positive share {repeat['positive_share']:.3f}. Real title/evidence handles are compared against shuffled, wrong-evidence, random, and high-frequency-page controls; claim-only/title shortcuts remain the main risk signal.

Selected controls: claim-only {control_map['claim_only']['utility']:.4f}, real evidence sentence {control_map['real_evidence_sentence']['utility']:.4f}, shuffled evidence sentence {control_map['shuffled_evidence_sentence']['utility']:.4f}, wrong evidence sentence {control_map['wrong_evidence_sentence']['utility']:.4f}, same-count random {control_map['same_count_random']['utility']:.4f}, high-frequency page {control_map['high_frequency_page']['utility']:.4f}.
"""
    pointer.write_text(text, encoding="utf-8")
    return pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_FEVER)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--full-evidence-boost", type=float, default=1.75)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    rows = load_fever_rows(args.data_path, args.max_rows)
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 verifiable FEVER rows with evidence handles, got {len(rows)}")
    records = score_records(rows, args)
    summary = summarize(records, args)
    md_path, json_path = write_report(summary, args, time.time() - start)
    pointer = write_pointer(summary, md_path, json_path)
    print(json.dumps(summary["repeat_summary"], indent=2, sort_keys=True))
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {pointer}")


if __name__ == "__main__":
    main()
