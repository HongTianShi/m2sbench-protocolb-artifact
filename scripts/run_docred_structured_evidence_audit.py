from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_docred_evidence_access_audit import build_queries, encode_texts, ndcg_at_k


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TOKEN_RE = re.compile(r"[a-z0-9]+")

BASE_VIEW_ORDER = ["summary", "co_mention", "evidence_sentences", "full_context"]
VIEW_COST = {
    "summary": 0.0,
    "co_mention": 0.12,
    "evidence_sentences": 0.22,
    "full_context": 0.55,
    "ce": 0.95,
}
CONTROL_ORDER = [
    "title_position",
    "real_co_mention",
    "shuffled_co_mention",
    "degree_random_co_mention",
    "same_doc_random",
    "evidence_sentences",
    "full_context",
    "ce",
]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def stable_rng(text: str, seed: int, salt: str) -> np.random.Generator:
    import hashlib

    digest = hashlib.sha1(f"{text}:{seed}:{salt}".encode("utf-8")).hexdigest()[:16]
    return np.random.default_rng(int(digest, 16) % (2**32))


def bm25_scores(query: str, docs: list[str]) -> np.ndarray:
    q_terms = tokens(query)
    if not docs:
        return np.zeros(0, dtype=np.float32)
    doc_terms = [tokens(doc) for doc in docs]
    df: Counter[str] = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    avgdl = float(np.mean([len(x) for x in doc_terms]) or 1.0)
    scores = np.zeros(len(docs), dtype=np.float32)
    k1 = 1.2
    b = 0.75
    for i, terms in enumerate(doc_terms):
        tf = Counter(terms)
        dl = len(terms) or 1
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * tf[term] * (k1 + 1) / denom
    return scores


def margin(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    vals = np.sort(scores.astype(np.float64))[::-1]
    return float(vals[0] - vals[1])


def entropy(scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    x = scores.astype(np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def rank(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores.astype(np.float64), kind="mergesort")


def f1_at_k(order: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    pred = {int(x) for x in order[:k]}
    hit = len(pred & relevant)
    precision = hit / max(1, min(k, len(order)))
    recall = hit / max(1, len(relevant))
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def hit_at_k(order: np.ndarray, relevant: set[int], k: int) -> float:
    return float(any(int(item) in relevant for item in order[:k]))


def random_matched(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    values = sorted([float(x) for x in scores if float(x) > 0], reverse=True)
    if not values:
        return out
    ids = rng.choice(len(scores), size=min(len(values), len(scores)), replace=False)
    for idx, value in zip(ids, values):
        out[int(idx)] = value
    return out


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.60 * n)
    n_dev = int(0.20 * n)
    return idx[:n_train], idx[n_train : n_train + n_dev], idx[n_train + n_dev :]


def summarize_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str]) -> dict[str, Any]:
    utilities: list[float] = []
    ndcgs: list[float] = []
    f1s: list[float] = []
    costs: list[float] = []
    regrets: list[float] = []
    chosen: list[str] = []
    for pos, idx in enumerate(indices):
        rec = records[int(idx)]
        view = route[pos] if isinstance(route, list) else route
        row = rec["views"][view]
        utilities.append(row["utility"])
        ndcgs.append(row["ndcg"])
        f1s.append(row["evidence_f1"])
        costs.append(row["cost"])
        regrets.append(rec["oracle_utility"] - row["utility"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utilities)),
        "raw_ndcg": float(np.mean(ndcgs)),
        "evidence_f1": float(np.mean(f1s)),
        "cost": float(np.mean(costs)),
        "regret": float(np.mean(regrets)),
        "view_share": {k: round(v / len(chosen), 4) for k, v in counts.items()},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray, tier: str, view_order: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    rows: list[list[float]] = []
    y: list[float] = []
    keys: list[tuple[int, str]] = []
    for idx in indices:
        rec = records[int(idx)]
        stats = rec["visible_stats"]
        base = [
            stats["candidate_count"],
            stats["query_len"],
            stats["mean_sentence_len"],
            stats["summary_margin"],
            stats["summary_entropy"],
        ]
        if tier in {"B1_entity", "B1_sentence", "B1_context"}:
            base += [
                stats["co_top"],
                stats["co_margin"],
                stats["co_entropy"],
                stats["co_positive_share"],
            ]
        if tier in {"B1_sentence", "B1_context"}:
            base += [
                stats["bm25_top"],
                stats["bm25_margin"],
                stats["bm25_entropy"],
                stats["co_bm25_agreement"],
            ]
        if tier == "B1_context":
            base += [
                stats["max_sentence_len"],
                stats["std_sentence_len"],
                stats["relation_token_count"],
            ]
        for view_i, view in enumerate(view_order):
            one_hot = [1.0 if j == view_i else 0.0 for j in range(len(view_order))]
            profile = [
                VIEW_COST[view],
                1.0 if view == "co_mention" else 0.0,
                1.0 if view == "evidence_sentences" else 0.0,
                1.0 if view == "full_context" else 0.0,
                1.0 if view == "ce" else 0.0,
            ]
            rows.append(base + one_hot + profile)
            y.append(rec["views"][view]["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def route_from_predictions(keys: list[tuple[int, str]], pred: np.ndarray) -> list[str]:
    grouped: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (idx, view), value in zip(keys, pred):
        grouped[idx].append((view, float(value)))
    return [max(grouped[idx], key=lambda x: x[1])[0] for idx in sorted(grouped)]


def train_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int, tier: str, view_order: list[str]) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    models = {
        f"ridge_{tier}": Ridge(alpha=1.0),
        f"rf_{tier}": RandomForestRegressor(n_estimators=220, min_samples_leaf=4, random_state=seed, n_jobs=-1),
        f"et_{tier}": ExtraTreesRegressor(n_estimators=320, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        f"hgb_{tier}": HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, random_state=seed),
    }
    x_train, y_train, _ = make_features(records, train_idx, tier, view_order)
    x_dev, _, dev_keys = make_features(records, dev_idx, tier, view_order)
    x_test, _, test_keys = make_features(records, test_idx, tier, view_order)
    best_name = ""
    best_dev = -1e9
    best_model: Any = None
    for name, model in models.items():
        model.fit(x_train, y_train)
        route = route_from_predictions(dev_keys, model.predict(x_dev))
        dev_policy = summarize_policy(records, dev_idx, name, route)
        if dev_policy["utility"] > best_dev:
            best_dev = dev_policy["utility"]
            best_name = name
            best_model = model
    assert best_model is not None
    test_route = route_from_predictions(test_keys, best_model.predict(x_test))
    out = summarize_policy(records, test_idx, best_name, test_route)
    out["feature_tier"] = tier
    return out


def score_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from run_docred_evidence_access_audit import DocredEvidenceConfig

    config = DocredEvidenceConfig(max_queries=args.max_queries, batch_size=args.batch_size, model_name=args.model_name)
    queries, query_texts, sentence_texts, manifest = build_queries(config)
    query_emb = encode_texts(args.model_name, query_texts, args.batch_size)
    sent_emb = encode_texts(args.model_name, sentence_texts, args.batch_size)
    query_emb = query_emb.astype(np.float32)
    sent_emb = sent_emb.astype(np.float32)

    ce_scores_by_q: dict[int, np.ndarray] = {}
    if args.with_ce:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(args.ce_model, device=device)
        pairs: list[tuple[str, str]] = []
        slices: list[tuple[int, int, int]] = []
        cursor = 0
        for qi, q in enumerate(queries):
            start = cursor
            for sid in q["candidate_sentence_ids"]:
                pairs.append((q["query_text"], sentence_texts[sid]))
                cursor += 1
            slices.append((qi, start, cursor))
        scores = np.asarray(model.predict(pairs, batch_size=args.ce_batch_size, show_progress_bar=True), dtype=np.float32)
        for qi, start, end in slices:
            ce_scores_by_q[qi] = scores[start:end]

    view_order = BASE_VIEW_ORDER + (["ce"] if args.with_ce else [])
    records: list[dict[str, Any]] = []
    for qi, q in enumerate(queries):
        cand = np.asarray(q["candidate_sentence_ids"], dtype=np.int64)
        candidate_texts = [sentence_texts[int(sid)] for sid in cand]
        relevant_local = {q["candidate_sentence_ids"].index(sid) for sid in q["relevant_sentence_ids"]}
        n = len(cand)
        summary_scores = -0.001 * np.arange(n, dtype=np.float32)
        co_scores = np.asarray(q["summary_scores"], dtype=np.float32)
        bm25 = bm25_scores(q["query_text"], candidate_texts)
        full = sent_emb[cand] @ query_emb[qi]
        view_scores = {
            "summary": summary_scores,
            "co_mention": co_scores,
            "evidence_sentences": bm25,
            "full_context": full,
        }
        if args.with_ce:
            view_scores["ce"] = ce_scores_by_q[qi]
        controls = {
            "title_position": summary_scores,
            "real_co_mention": co_scores,
            "shuffled_co_mention": stable_rng(q["query_id"], args.seed, "shuffle").permutation(co_scores),
            "degree_random_co_mention": random_matched(co_scores, stable_rng(q["query_id"], args.seed, "degree")),
            "same_doc_random": stable_rng(q["query_id"], args.seed, "random").random(n).astype(np.float32),
            "evidence_sentences": bm25,
            "full_context": full,
        }
        if args.with_ce:
            controls["ce"] = ce_scores_by_q[qi]

        views: dict[str, dict[str, float]] = {}
        for view, scores in view_scores.items():
            order = rank(scores)
            ndcg = ndcg_at_k(order, relevant_local, args.topk)
            util = ndcg - args.lambda_cost * VIEW_COST[view]
            views[view] = {
                "ndcg": float(ndcg),
                "evidence_f1": f1_at_k(order, relevant_local, args.topk),
                "hit": hit_at_k(order, relevant_local, args.topk),
                "utility": float(util),
                "cost": VIEW_COST[view],
            }
        control_rows: dict[str, dict[str, float]] = {}
        for name, scores in controls.items():
            order = rank(scores)
            cost = VIEW_COST.get(name, VIEW_COST.get(name.replace("real_", ""), 0.0))
            if name in {"shuffled_co_mention", "degree_random_co_mention"}:
                cost = VIEW_COST["co_mention"]
            if name == "same_doc_random":
                cost = VIEW_COST["evidence_sentences"]
            ndcg = ndcg_at_k(order, relevant_local, args.topk)
            control_rows[name] = {
                "control": name,
                "ndcg": float(ndcg),
                "evidence_f1": f1_at_k(order, relevant_local, args.topk),
                "hit": hit_at_k(order, relevant_local, args.topk),
                "utility": float(ndcg - args.lambda_cost * cost),
                "cost": cost,
            }
        oracle = max(view_order, key=lambda v: views[v]["utility"])
        lengths = np.asarray([len(tokens(text)) for text in candidate_texts], dtype=np.float32)
        co_rank = rank(co_scores)
        bm_rank = rank(bm25)
        records.append(
            {
                "id": q["query_id"],
                "split": q["split"],
                "relation": q["relation"],
                "candidate_count": n,
                "views": views,
                "controls": control_rows,
                "oracle_view": oracle,
                "oracle_utility": views[oracle]["utility"],
                "visible_stats": {
                    "candidate_count": float(n),
                    "query_len": float(len(tokens(q["query_text"]))),
                    "mean_sentence_len": float(np.mean(lengths)),
                    "std_sentence_len": float(np.std(lengths)),
                    "max_sentence_len": float(np.max(lengths)),
                    "relation_token_count": float(len(tokens(q["relation"]))),
                    "summary_margin": margin(summary_scores),
                    "summary_entropy": entropy(summary_scores),
                    "co_top": float(np.max(co_scores)) if len(co_scores) else 0.0,
                    "co_margin": margin(co_scores),
                    "co_entropy": entropy(co_scores),
                    "co_positive_share": float(np.mean(co_scores > 0)) if len(co_scores) else 0.0,
                    "bm25_top": float(np.max(bm25)) if len(bm25) else 0.0,
                    "bm25_margin": margin(bm25),
                    "bm25_entropy": entropy(bm25),
                    "co_bm25_agreement": float(len(set(co_rank[: args.topk]) & set(bm_rank[: args.topk])) / max(1, args.topk)),
                },
            }
        )
    meta = {
        "sources": manifest,
        "model": args.model_name,
        "with_ce": bool(args.with_ce),
        "view_order": view_order,
        "queries": len(records),
        "candidate_sentences": len(sentence_texts),
    }
    return records, meta


def aggregate_view(records: list[dict[str, Any]], indices: np.ndarray, view: str) -> dict[str, Any]:
    vals = [records[int(i)]["views"][view] for i in indices]
    return {
        "view": view,
        "utility": float(np.mean([v["utility"] for v in vals])),
        "raw_ndcg": float(np.mean([v["ndcg"] for v in vals])),
        "evidence_f1": float(np.mean([v["evidence_f1"] for v in vals])),
        "hit": float(np.mean([v["hit"] for v in vals])),
        "cost": float(np.mean([v["cost"] for v in vals])),
    }


def aggregate_control(records: list[dict[str, Any]], indices: np.ndarray, control: str) -> dict[str, Any]:
    vals = [records[int(i)]["controls"][control] for i in indices if control in records[int(i)]["controls"]]
    return {
        "control": control,
        "utility": float(np.mean([v["utility"] for v in vals])),
        "raw_ndcg": float(np.mean([v["ndcg"] for v in vals])),
        "evidence_f1": float(np.mean([v["evidence_f1"] for v in vals])),
        "hit": float(np.mean([v["hit"] for v in vals])),
        "cost": float(np.mean([v["cost"] for v in vals])),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    records, meta = score_records(args)
    train_idx, dev_idx, test_idx = split_indices(len(records), args.seed)
    view_order = meta["view_order"]

    fixed_rows = [summarize_policy(records, test_idx, f"fixed_{view}", view) for view in view_order]
    best_fixed = max(fixed_rows, key=lambda row: row["utility"])
    oracle_route = [records[int(i)]["oracle_view"] for i in test_idx]
    oracle = summarize_policy(records, test_idx, "oracle_route", oracle_route)
    denom = max(1e-9, oracle["utility"] - best_fixed["utility"])
    for row in fixed_rows:
        row["gap_closed"] = (row["utility"] - best_fixed["utility"]) / denom
        row["diff_vs_best_fixed"] = row["utility"] - best_fixed["utility"]

    tier_rows = []
    for tier in ["B0_summary", "B1_entity", "B1_sentence", "B1_context"]:
        row = train_router(records, train_idx, dev_idx, test_idx, args.seed, tier, view_order)
        row["gap_closed"] = (row["utility"] - best_fixed["utility"]) / denom
        row["diff_vs_best_fixed"] = row["utility"] - best_fixed["utility"]
        tier_rows.append(row)
    best_learner = max(tier_rows, key=lambda row: row["utility"])
    oracle["gap_closed"] = 1.0
    oracle["diff_vs_best_fixed"] = oracle["utility"] - best_fixed["utility"]
    policy_rows = fixed_rows + [best_learner, oracle]

    control_rows = [aggregate_control(records, test_idx, c) for c in CONTROL_ORDER if c in records[0]["controls"]]
    best_control = max(control_rows, key=lambda row: row["utility"])
    for row in control_rows:
        row["diff_vs_best_control"] = row["utility"] - best_control["utility"]

    repeat_deltas = []
    repeat_policies = []
    for rep in range(args.repeats):
        tr, dv, te = split_indices(len(records), args.seed + rep + 1)
        fixed = [summarize_policy(records, te, f"fixed_{view}", view) for view in view_order]
        bf = max(fixed, key=lambda row: row["utility"])
        tier_candidates = [train_router(records, tr, dv, te, args.seed + rep + 1, tier, view_order) for tier in ["B0_summary", "B1_entity", "B1_sentence", "B1_context"]]
        bl = max(tier_candidates, key=lambda row: row["utility"])
        repeat_deltas.append(bl["utility"] - bf["utility"])
        repeat_policies.append(bl["policy"])
    repeat_arr = np.asarray(repeat_deltas, dtype=np.float64)
    repeat = {
        "mean_learned_minus_best_fixed": float(repeat_arr.mean()),
        "ci95": [float(np.percentile(repeat_arr, 2.5)), float(np.percentile(repeat_arr, 97.5))],
        "positive_share": float(np.mean(repeat_arr > 0)),
        "selected_policies": dict(Counter(repeat_policies)),
    }

    oracle_counts = Counter(records[int(i)]["oracle_view"] for i in test_idx)
    result = {
        "audit": "docred_structured_evidence",
        "status": "candidate_structured_evidence_slice",
        "purpose": "DocRED structured evidence acquisition under Protocol B with exact evidence-sentence qrels.",
        "config": {
            "max_queries": args.max_queries,
            "topk": args.topk,
            "lambda_cost": args.lambda_cost,
            "seed": args.seed,
            "repeats": args.repeats,
            "with_ce": args.with_ce,
            "model_name": args.model_name,
            "ce_model": args.ce_model if args.with_ce else None,
            "view_cost": {v: VIEW_COST[v] for v in view_order},
        },
        "sources": meta["sources"],
        "queries": meta["queries"],
        "candidate_sentences": meta["candidate_sentences"],
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "view_order": view_order,
        "fixed_and_adaptive": policy_rows,
        "feature_tier_rows": tier_rows,
        "control_rows": control_rows,
        "oracle_view_share": {k: float(v / len(test_idx)) for k, v in oracle_counts.items()},
        "repeated_split_stability": repeat,
        "elapsed_seconds": time.time() - start,
        "reading": (
            "DocRED is a second structured-evidence family: methods buy entity/co-mention, candidate-sentence, "
            "full-context, or optional CE evidence before evaluator-held evidence-sentence qrels are scored."
        ),
    }
    return result


def write_report(result: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = "ce" if result["config"]["with_ce"] else "noce"
    json_path = REPORTS / f"docred_structured_evidence_{suffix}.json"
    md_path = REPORTS / f"docred_structured_evidence_{suffix}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    policy_cols = ["policy", "utility", "raw_ndcg", "evidence_f1", "cost", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    tier_cols = ["feature_tier", "policy", "utility", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "evidence_f1", "hit", "cost", "diff_vs_best_control"]
    stability = result["repeated_split_stability"]
    md = f"""# DocRED Structured Evidence Acquisition Audit

Protocol B slice over DocRED relation evidence annotations. A method sees a document/relation/entity-pair sketch and may buy declared evidence views before evaluator-held evidence sentence ids are scored. This is evidence acquisition, not relation extraction training.

- Queries: {result['queries']:,}; candidate sentences: {result['candidate_sentences']:,}; split: {result['split_sizes']}
- Top-k: {result['config']['topk']}; lambda: {result['config']['lambda_cost']}
- Views: {result['view_order']}
- View costs: {result['config']['view_cost']}
- Oracle view share on hidden test: {result['oracle_view_share']}

## Fixed and adaptive policies

{markdown_table(result['fixed_and_adaptive'], policy_cols)}

## Legal feature-tier learner ablation

Rows use only released manifest summaries: candidate counts, entity/co-mention score summaries, candidate-sentence score summaries, and context-length metadata. Evidence sentence ids, relation labels used for scoring, unpaid full/CE scores, and oracle views remain evaluator-held.

{markdown_table(result['feature_tier_rows'], tier_cols)}

## Real evidence views versus controls

{markdown_table(result['control_rows'], control_cols)}

## Repeated split stability

```json
{json.dumps(stability, indent=2)}
```

## Reading

{result['reading']}
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=30_000)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--with-ce", action="store_true")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    args = parser.parse_args()
    result = evaluate(args)
    write_report(result)


if __name__ == "__main__":
    main()
