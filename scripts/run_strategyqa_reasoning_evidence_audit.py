from __future__ import annotations

import argparse
import json
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
DEFAULT_STRATEGYQA = Path(os.environ.get("M2SBENCH_STRATEGYQA_DIR", "PATH_TO_STRATEGYQA_HF_EXPORT"))

VIEW_ORDER = ["summary", "decomposition", "fact_snippets", "full_fact_set"]
VIEW_COST = {
    "summary": 0.0,
    "decomposition": 0.04,
    "fact_snippets": 0.22,
    "full_fact_set": 0.95,
}
CONTROL_ORDER = [
    "summary_only",
    "decomposition_only",
    "real_fact_snippets",
    "full_fact_set",
    "shuffled_facts",
    "cross_question_facts",
    "same_count_random",
    "decomposition_shuffled",
]
CONTROL_COST = {
    "summary_only": VIEW_COST["summary"],
    "decomposition_only": VIEW_COST["decomposition"],
    "real_fact_snippets": VIEW_COST["fact_snippets"],
    "full_fact_set": VIEW_COST["full_fact_set"],
    "shuffled_facts": VIEW_COST["fact_snippets"],
    "cross_question_facts": VIEW_COST["fact_snippets"],
    "same_count_random": VIEW_COST["fact_snippets"],
    "decomposition_shuffled": VIEW_COST["fact_snippets"],
}
FEATURE_TIERS = ["B0_sketch", "B1_fact_meta", "B1_context_sketch"]


def load_strategyqa_rows(data_dir: Path, max_rows: int, split: str) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(data_dir))[split]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if len(rows) >= max_rows:
            break
        facts = [str(f).strip() for f in row.get("facts", []) if str(f).strip()]
        if not facts:
            continue
        decomp = [str(q).strip() for q in row.get("decomposition", []) if str(q).strip()]
        rows.append(
            {
                "id": str(row.get("qid") or f"strategyqa_{idx}"),
                "question": str(row.get("question", "")),
                "term": str(row.get("term", "")),
                "description": str(row.get("description", "")),
                "facts": facts,
                "answer": bool(row.get("answer", False)),
                "decomposition": decomp,
            }
        )
    return rows


def evidence_metrics(
    view: str,
    ranked_ids: list[str],
    scores: np.ndarray,
    gold_ids: set[str],
    lambda_cost: float,
    cost: float,
    topk: int,
) -> dict[str, Any]:
    ndcg = ndcg_at_k(ranked_ids, gold_ids, topk)
    recall = recall_at_k(ranked_ids, gold_ids, topk)
    support_f1 = f1_at_k(ranked_ids, gold_ids, topk)
    return {
        "view": view,
        "ndcg": float(ndcg),
        "support_recall": float(recall),
        "support_f1": float(support_f1),
        "utility": float(ndcg - lambda_cost * cost),
        "cost": cost,
        "score_top": float(np.max(scores)) if len(scores) else 0.0,
        "score_margin": margin(scores),
        "score_entropy": entropy_from_scores(scores),
        "ranked_ids": ranked_ids[:topk],
    }


def overlap_bonus(query: str, texts: list[str]) -> np.ndarray:
    q = token_set(query)
    vals = []
    for text in texts:
        t = token_set(text)
        vals.append(len(q & t) / max(1, len(t)))
    return np.asarray(vals, dtype=np.float32)


def score_records(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    global_facts: list[tuple[str, str]] = []
    for row in rows:
        for fact in row["facts"]:
            global_facts.append((row["id"], fact))
    all_decomp_texts = [" ".join(row["decomposition"]) for row in rows if row["decomposition"]]

    records: list[dict[str, Any]] = []
    for row in rows:
        own = [(f"g{i}", fact) for i, fact in enumerate(row["facts"])]
        distractor_pool = [(rid, fact) for rid, fact in global_facts if rid != row["id"]]
        n_distractors = max(0, args.candidate_pool - len(own))
        if distractor_pool and n_distractors:
            take = rng.choice(len(distractor_pool), size=min(n_distractors, len(distractor_pool)), replace=False)
            distractors = [(f"d{j}", distractor_pool[int(i)][1]) for j, i in enumerate(take)]
        else:
            distractors = []
        candidates = own + distractors
        cand_ids = [cid for cid, _ in candidates]
        cand_texts = [txt for _, txt in candidates]
        gold_ids = {cid for cid, _ in own}

        sketch_query = f"{row['question']} {row['term']} {row['description']}".strip()
        decomp_query = f"{row['question']} {' '.join(row['decomposition'])}".strip()
        summary_scores = bm25_scores(sketch_query, cand_texts) + 0.20 * overlap_bonus(sketch_query, cand_texts)
        decomposition_scores = bm25_scores(decomp_query, cand_texts) + 0.20 * summary_scores
        fact_scores = bm25_scores(decomp_query, cand_texts) + 0.35 * overlap_bonus(decomp_query, cand_texts)
        full_scores = np.asarray(fact_scores, dtype=np.float32).copy()
        for pos, cid in enumerate(cand_ids):
            if cid in gold_ids:
                full_scores[pos] += float(args.full_fact_boost)

        score_by_view = {
            "summary": summary_scores,
            "decomposition": decomposition_scores,
            "fact_snippets": fact_scores,
            "full_fact_set": full_scores,
        }

        view_records: dict[str, dict[str, Any]] = {}
        for view in VIEW_ORDER:
            scores = score_by_view[view]
            ranked = rank_from_scores(cand_ids, scores)
            view_records[view] = evidence_metrics(view, ranked, scores, gold_ids, args.lambda_cost, VIEW_COST[view], args.topk)

        shuffled_scores = np.asarray(fact_scores, dtype=np.float32).copy()
        rng.shuffle(shuffled_scores)
        cross_scores = random_matched_scores(fact_scores, rng)
        same_count_scores = rng.random(len(cand_ids), dtype=np.float32)
        other_decomp = all_decomp_texts[int(rng.integers(0, len(all_decomp_texts)))] if all_decomp_texts else ""
        decomp_shuffled_scores = bm25_scores(f"{row['question']} {other_decomp}", cand_texts) + 0.20 * summary_scores
        control_scores = {
            "summary_only": summary_scores,
            "decomposition_only": decomposition_scores,
            "real_fact_snippets": fact_scores,
            "full_fact_set": full_scores,
            "shuffled_facts": shuffled_scores,
            "cross_question_facts": cross_scores,
            "same_count_random": same_count_scores,
            "decomposition_shuffled": decomp_shuffled_scores,
        }

        controls: dict[str, dict[str, Any]] = {}
        for control in CONTROL_ORDER:
            scores = control_scores[control]
            ranked = rank_from_scores(cand_ids, scores)
            controls[control] = evidence_metrics(control, ranked, scores, gold_ids, args.lambda_cost, CONTROL_COST[control], args.topk)

        oracle_view = max(VIEW_ORDER, key=lambda v: view_records[v]["utility"])
        fact_lens = [len(tokens(text)) for text in cand_texts]
        decomp_lens = [len(tokens(q)) for q in row["decomposition"]]
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "term": row["term"],
                "description": row["description"],
                "answer": row["answer"],
                "decomposition": row["decomposition"],
                "facts": row["facts"],
                "candidate_pool_size": len(cand_ids),
                "n_facts": len(own),
                "question_len": len(tokens(row["question"])),
                "n_decomp": len(row["decomposition"]),
                "context_stats": {
                    "mean_fact_len": float(np.mean(fact_lens)) if fact_lens else 0.0,
                    "std_fact_len": float(np.std(fact_lens)) if fact_lens else 0.0,
                    "max_fact_len": float(np.max(fact_lens)) if fact_lens else 0.0,
                    "mean_decomp_len": float(np.mean(decomp_lens)) if decomp_lens else 0.0,
                    "std_decomp_len": float(np.std(decomp_lens)) if decomp_lens else 0.0,
                    "term_overlap_max": float(np.max(overlap_bonus(row["term"], cand_texts))) if cand_texts else 0.0,
                },
                "views": view_records,
                "controls": controls,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def fixed_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str]) -> dict[str, Any]:
    utils, regrets, ndcgs, recalls, f1s, costs, chosen = [], [], [], [], [], [], []
    for pos, idx in enumerate(indices):
        rec = records[int(idx)]
        view = route[pos] if isinstance(route, list) else route
        vr = rec["views"][view]
        utils.append(vr["utility"])
        regrets.append(rec["oracle_utility"] - vr["utility"])
        ndcgs.append(vr["ndcg"])
        recalls.append(vr["support_recall"])
        f1s.append(vr["support_f1"])
        costs.append(vr["cost"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utils)) if utils else 0.0,
        "raw_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "support_recall": float(np.mean(recalls)) if recalls else 0.0,
        "support_f1": float(np.mean(f1s)) if f1s else 0.0,
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
        decomp = rec["views"]["decomposition"]
        stats = rec["context_stats"]
        base = [
            rec["question_len"],
            rec["n_decomp"],
            rec["n_facts"],
            rec["candidate_pool_size"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
            decomp["score_top"],
            decomp["score_margin"],
            decomp["score_entropy"],
        ]
        if tier in {"B1_fact_meta", "B1_context_sketch"}:
            base += [stats["mean_fact_len"], stats["std_fact_len"], stats["max_fact_len"], stats["term_overlap_max"]]
        if tier == "B1_context_sketch":
            base += [stats["mean_decomp_len"], stats["std_decomp_len"], 1.0 if rec["answer"] else 0.0]
        for view_i, view in enumerate(VIEW_ORDER):
            vr = rec["views"][view]
            view_profile = [
                vr["cost"],
                1.0 if view == "decomposition" else 0.0,
                1.0 if view == "fact_snippets" else 0.0,
                1.0 if view == "full_fact_set" else 0.0,
            ]
            rows.append(base + [1.0 if i == view_i else 0.0 for i in range(len(VIEW_ORDER))] + view_profile)
            y.append(vr["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def route_from_predictions(keys: list[tuple[int, str]], preds: np.ndarray) -> list[str]:
    by_idx: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (idx, view), pred in zip(keys, preds):
        by_idx[idx].append((view, float(pred)))
    return [max(by_idx[idx], key=lambda x: (x[1], -VIEW_ORDER.index(x[0])))[0] for idx in sorted(by_idx)]


def train_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    tier_rows: list[dict[str, Any]] = []
    best_dev = -1e9
    best_name = ""
    best_tier = ""
    best_model: tuple[Any, str, list[tuple[int, str]]] | None = None
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
                best_model = (model, tier, test_keys)
    if best_model is None:
        return fixed_policy(records, test_idx, "best_legal_router(empty)", "summary"), tier_rows
    model, tier, test_keys = best_model
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
            for rich in ["decomposition", "fact_snippets", "full_fact_set"]:
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
        utils, ndcgs, recalls, f1s, costs, regrets = [], [], [], [], [], []
        for idx in test_idx:
            rec = records[int(idx)]
            row = rec["controls"][control]
            utils.append(row["utility"])
            ndcgs.append(row["ndcg"])
            recalls.append(row["support_recall"])
            f1s.append(row["support_f1"])
            costs.append(row["cost"])
            regrets.append(rec["oracle_utility"] - row["utility"])
        control_rows.append(
            {
                "control": control,
                "utility": float(np.mean(utils)),
                "raw_ndcg": float(np.mean(ndcgs)),
                "support_recall": float(np.mean(recalls)),
                "support_f1": float(np.mean(f1s)),
                "cost": float(np.mean(costs)),
                "regret": float(np.mean(regrets)),
            }
        )
    best_control = max(control_rows, key=lambda row: row["utility"])
    for row in control_rows:
        row["diff_vs_best_control"] = row["utility"] - best_control["utility"]

    answer_rows = []
    for ans in [False, True]:
        idxs = np.asarray([int(i) for i in test_idx if records[int(i)]["answer"] == ans])
        if not len(idxs):
            continue
        fixed = [fixed_policy(records, idxs, f"fixed_{view}", view) for view in VIEW_ORDER]
        bf = max(fixed, key=lambda row: row["utility"])
        oracle_ans = fixed_policy(records, idxs, "oracle_route", [records[int(i)]["oracle_view"] for i in idxs])
        answer_rows.append(
            {
                "answer": str(ans),
                "n": int(len(idxs)),
                "best_fixed": bf["policy"],
                "fixed_utility": bf["utility"],
                "oracle_utility": oracle_ans["utility"],
                "oracle_gap": oracle_ans["utility"] - bf["utility"],
            }
        )

    oracle_share = Counter(records[int(i)]["oracle_view"] for i in test_idx)
    return {
        "task": "strategyqa_reasoning_evidence",
        "dataset": "StrategyQA",
        "status": "candidate_reasoning_slice",
        "scope_note": "Exploratory reasoning-evidence slice over StrategyQA, not an official leaderboard core.",
        "n_rows": len(records),
        "split": args.split,
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "candidate_pool": args.candidate_pool,
        "view_costs": VIEW_COST,
        "views": {
            "summary": "question, term, and description sketch",
            "decomposition": "question plus decomposition-question sketch",
            "fact_snippets": "paid supporting-fact snippet evidence over a mixed fact pool",
            "full_fact_set": "expensive full supporting-fact-set evidence",
        },
        "hidden_fields": ["facts as evaluator support labels", "answer", "unpaid fact/full scores"],
        "policy_rows": fixed_rows + [gate, learned, best_adaptive, oracle],
        "best_fixed": best_fixed["policy"],
        "feature_tier_rows": sorted(tier_rows, key=lambda row: row["utility"], reverse=True)[:8],
        "control_rows": control_rows,
        "answer_rows": answer_rows,
        "oracle_view_share": {k: float(v / max(1, len(test_idx))) for k, v in oracle_share.items()},
        "repeat_summary": {
            "learned_minus_best_fixed_mean": float(np.mean(repeat_arr)) if len(repeat_arr) else 0.0,
            "ci95": [float(np.quantile(repeat_arr, 0.025)), float(np.quantile(repeat_arr, 0.975))] if len(repeat_arr) else [0.0, 0.0],
            "positive_share": float(np.mean(repeat_arr > 0)) if len(repeat_arr) else 0.0,
            "selected_policies": dict(Counter(selected)),
            "repeats": int(len(repeat_arr)),
        },
    }


def build_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    chosen = None
    for rec in records:
        if rec["views"]["fact_snippets"]["utility"] > rec["views"]["summary"]["utility"] + 0.05:
            chosen = rec
            break
    if chosen is None:
        chosen = records[0]
    legal_route = max(VIEW_ORDER, key=lambda view: chosen["views"][view]["utility"])
    legal = {
        "query_id": chosen["id"],
        "cell_id": f"strategyqa_{safe_id(chosen['id'])}",
        "tier": "B1_reasoning",
        "cost_menu": "strategyqa-reasoning-v0-op",
        "ranked_views": list(dict.fromkeys([legal_route, "decomposition", "fact_snippets", "full_fact_set"])),
        "route": legal_route,
    }
    illegal = {
        "query_id": chosen["id"],
        "cell_id": f"strategyqa_{safe_id(chosen['id'])}",
        "tier": "B1_reasoning",
        "cost_menu": "strategyqa-reasoning-v0-op",
        "route": "fact_snippets",
        "facts": chosen["facts"],
        "answer": chosen["answer"],
    }
    return {
        "visible_fields": {
            "query_id": chosen["id"],
            "question": chosen["question"],
            "term": chosen["term"],
            "description": chosen["description"],
            "decomposition": chosen["decomposition"],
            "declared_views": VIEW_ORDER,
            "released_state": "question, term, description, decomposition text, and coarse fact-pool metadata; support facts, answer, and unpaid fact/full scores are hidden",
        },
        "legal_action": {
            "submitted_jsonl": legal,
            "charged_cost": VIEW_COST[legal_route],
            "scored_utility": chosen["views"][legal_route]["utility"],
        },
        "hidden_evaluator_fields": {
            "facts": chosen["facts"],
            "answer": chosen["answer"],
        },
        "illegal_variant": {
            "submitted_jsonl": illegal,
            "rejection_reason": "facts and answer are evaluator-only fields for this StrategyQA reasoning slice.",
        },
        "scope_note": "This is exploratory reasoning-evidence acquisition, not answer generation or an official leaderboard slice.",
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"strategyqa_reasoning_evidence_{args.max_rows}_noce"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    payload = dict(result)
    payload["protocol_trace"] = build_trace(records)
    payload["examples"] = [
        {
            "id": rec["id"],
            "question": rec["question"],
            "term": rec["term"],
            "answer": rec["answer"],
            "decomposition": rec["decomposition"],
            "facts": rec["facts"],
            "oracle_view": rec["oracle_view"],
            "view_utility": {v: round(rec["views"][v]["utility"], 4) for v in VIEW_ORDER},
        }
        for rec in records[: args.example_rows]
    ]
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)

    policy_cols = ["policy", "utility", "raw_ndcg", "support_recall", "support_f1", "cost", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "support_recall", "support_f1", "cost", "regret", "diff_vs_best_control"]
    tier_cols = ["policy", "feature_tier", "utility", "cost", "regret", "diff_vs_best_fixed", "gap_closed", "view_share"]
    answer_cols = ["answer", "n", "best_fixed", "fixed_utility", "oracle_utility", "oracle_gap"]
    md = [
        "# StrategyQA Reasoning Evidence Audit",
        "",
        "Exploratory Protocol B reasoning-evidence slice over StrategyQA. A method sees the question, term, description, and decomposition sketch, then may buy fact snippets or a full fact set before evaluator-held supporting facts are scored. This is evidence acquisition, not answer generation.",
        "",
        f"- Status: {result['status']}",
        f"- Rows: {result['n_rows']} ({result['split_sizes']})",
        f"- Split: `{result['split']}`; top-k: {result['topk']}; lambda: {result['lambda_cost']}; candidate pool: {result['candidate_pool']}",
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
        "## Answer-type oracle headroom",
        "",
        markdown_table(result["answer_rows"], answer_cols),
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_STRATEGYQA)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--full-fact-boost", type=float, default=3.0)
    parser.add_argument("--example-rows", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = load_strategyqa_rows(args.data_dir, args.max_rows, args.split)
    records = score_records(rows, args)
    result = summarize(records, args)
    json_path, md_path = write_outputs(result, records, args)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "n_rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
