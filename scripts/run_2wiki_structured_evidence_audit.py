from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_HF_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", "data/external/hf_cache/datasets"))
DEFAULT_CORPUS = Path(
    os.environ.get(
        "M2SBENCH_2WIKI_HYPERLINK_CORPUS",
        "data/external/2WikiMultiHopQA/para_with_hyperlink/para_with_hyperlink.jsonl",
    )
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_FIELD_RE = re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"')
VIEW_ORDER = ["summary", "one_hop", "two_hop", "full_context", "ce"]
VIEW_COST = {
    "summary": 0.0,
    "one_hop": 0.16,
    "two_hop": 0.26,
    "full_context": 0.55,
    "ce": 0.95,
}
CONTROL_ORDER = [
    "title_only",
    "real_1hop",
    "real_2hop",
    "shuffled_1hop",
    "degree_random_1hop",
    "random_2hop",
    "full_context",
    "ce",
]
CONTROL_COST = {
    "title_only": 0.0,
    "real_1hop": VIEW_COST["one_hop"],
    "real_2hop": VIEW_COST["two_hop"],
    "shuffled_1hop": VIEW_COST["one_hop"],
    "degree_random_1hop": VIEW_COST["one_hop"],
    "random_2hop": VIEW_COST["two_hop"],
    "full_context": VIEW_COST["full_context"],
    "ce": VIEW_COST["ce"],
}
FEATURE_TIERS = ["B0_title", "B1_entity", "B1_graph", "B1_path", "B1_context"]


def norm_title(text: str) -> str:
    text = str(text)
    text = text.replace("_", " ").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|")

    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(out)


def dcg_at_k(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    score = 0.0
    for rank, title in enumerate(ranked_titles[:k], start=1):
        rel = 1.0 if norm_title(title) in gold_titles else 0.0
        if rel:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_titles))))
    return dcg_at_k(ranked_titles, gold_titles, k) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    got = {norm_title(t) for t in ranked_titles[:k]}
    return len(got & gold_titles) / len(gold_titles)


def mrr_at_k(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    for rank, title in enumerate(ranked_titles[:k], start=1):
        if norm_title(title) in gold_titles:
            return 1.0 / rank
    return 0.0


def f1_at_k(ranked_titles: list[str], gold_titles: set[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    got = {norm_title(t) for t in ranked_titles[:k]}
    hits = len(got & gold_titles)
    precision = hits / max(1, min(k, len(ranked_titles)))
    recall = hits / len(gold_titles)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def deterministic_rng(row_id: str, seed: int, salt: str) -> np.random.Generator:
    digest = hashlib.sha1(f"{row_id}:{seed}:{salt}".encode("utf-8")).hexdigest()[:16]
    return np.random.default_rng(int(digest, 16) % (2**32))


def random_matched_scores(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(source, dtype=np.float32)
    weights = [float(x) for x in source if float(x) > 0]
    if not weights:
        return out
    weights = sorted(weights, reverse=True)
    chosen = rng.choice(len(source), size=min(len(weights), len(source)), replace=False)
    for idx, weight in zip(chosen, weights):
        out[int(idx)] = weight
    return out


def bm25_scores(query: str, docs: list[str]) -> np.ndarray:
    q_terms = tokens(query)
    if not docs:
        return np.zeros(0, dtype=np.float32)
    doc_terms = [tokens(d) for d in docs]
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    avgdl = float(np.mean([len(t) for t in doc_terms]) or 1.0)
    k1 = 1.2
    b = 0.75
    scores = np.zeros(n_docs, dtype=np.float32)
    for i, terms in enumerate(doc_terms):
        tf = Counter(terms)
        dl = len(terms) or 1
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * tf[term] * (k1 + 1) / denom
    return scores


def rank_from_scores(titles: list[str], scores: np.ndarray) -> list[str]:
    order = sorted(range(len(titles)), key=lambda i: (-float(scores[i]), norm_title(titles[i])))
    return [titles[i] for i in order]


def margin(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    vals = np.sort(scores)[::-1]
    return float(vals[0] - vals[1])


def entropy_from_scores(scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    x = scores.astype(np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    p = p / max(float(p.sum()), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def load_2wiki_rows(max_rows: int, cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        "voidful/2wikimultihopqa",
        split=f"train[:{max_rows}]",
        cache_dir=str(cache_dir),
    )
    rows: list[dict[str, Any]] = []
    for row in ds:
        contexts = []
        for title, sentences in row["context"]:
            contexts.append({"title": str(title), "sentences": [str(s) for s in sentences]})
        support_titles = set()
        for fact in row["supporting_facts"]:
            if not fact:
                continue
            title = fact[0] if isinstance(fact[0], str) else (fact[1] if len(fact) > 1 and isinstance(fact[1], str) else fact[0])
            support_titles.add(norm_title(title))
        evidence_entities = set()
        for triple in row["evidences"]:
            if len(triple) >= 3:
                evidence_entities.add(norm_title(str(triple[0])))
                evidence_entities.add(norm_title(str(triple[2])))
        rows.append(
            {
                "id": row["_id"],
                "type": row["type"],
                "question": row["question"],
                "answer": row["answer"],
                "contexts": contexts,
                "supporting_facts": row["supporting_facts"],
                "evidences": row["evidences"],
                "support_titles": support_titles,
                "evidence_entities": evidence_entities,
            }
        )
    return rows


def build_link_map(rows: list[dict[str, Any]], corpus_path: Path, cache_path: Path) -> dict[str, set[str]]:
    needed = {norm_title(ctx["title"]) for row in rows for ctx in row["contexts"]}
    if cache_path.exists():
        data = json.load(cache_path.open("r", encoding="utf-8"))
        cached_needed = set(data.get("needed_titles", []))
        if cached_needed == needed:
            return {k: set(v) for k, v in data["links"].items()}

    links: dict[str, set[str]] = {title: set() for title in needed}
    found = set()
    start = time.time()
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            # The hyperlink corpus is roughly 7GB.  Avoid full JSON parsing for
            # non-target rows; parse the title field first and only decode rows
            # that match a title in the sampled 2Wiki contexts.
            match = TITLE_FIELD_RE.search(line)
            if not match:
                continue
            try:
                title = norm_title(json.loads('"' + match.group(1) + '"'))
            except json.JSONDecodeError:
                continue
            if title not in needed:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            found.add(title)
            out = links.setdefault(title, set())
            for mention in obj.get("mentions", []) or []:
                ref_url = str(mention.get("ref_url", "")).strip()
                if ref_url:
                    out.add(norm_title(ref_url))
                for ref_id in mention.get("ref_ids", []) or []:
                    # Keep ids out of visible text, but count the presence of a link target.
                    if ref_id:
                        pass
            if len(found) == len(needed):
                break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": str(corpus_path),
            "seconds": round(time.time() - start, 3),
            "needed_titles": sorted(needed),
            "found_titles": sorted(found),
            "links": {k: sorted(v) for k, v in links.items()},
        },
        cache_path.open("w", encoding="utf-8"),
        indent=2,
    )
    return links


def context_graph_scores(row: dict[str, Any], link_map: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    titles = [ctx["title"] for ctx in row["contexts"]]
    norms = [norm_title(t) for t in titles]
    q_norm = norm_title(row["question"])
    q_tokens = token_set(row["question"])
    title_tokens = [token_set(t) for t in titles]
    anchors = []
    for i, nt in enumerate(norms):
        overlap = len(title_tokens[i] & q_tokens) / max(1, len(title_tokens[i]))
        if nt and nt in q_norm:
            anchors.append(i)
        elif overlap >= 0.50 and len(title_tokens[i] & q_tokens) >= 1:
            anchors.append(i)
    if not anchors:
        overlaps = [len(ts & q_tokens) for ts in title_tokens]
        if overlaps:
            anchors = [int(np.argmax(overlaps))]

    norm_to_idx = {nt: i for i, nt in enumerate(norms)}
    outgoing = {nt: {x for x in link_map.get(nt, set()) if x in norm_to_idx} for nt in norms}
    incoming: dict[str, set[str]] = defaultdict(set)
    for src, dsts in outgoing.items():
        for dst in dsts:
            incoming[dst].add(src)

    one = np.zeros(len(titles), dtype=np.float32)
    two = np.zeros(len(titles), dtype=np.float32)
    anchor_norms = {norms[i] for i in anchors}
    for i, nt in enumerate(norms):
        direct = 0
        reverse = 0
        for a in anchor_norms:
            direct += int(nt in outgoing.get(a, set()))
            reverse += int(a in outgoing.get(nt, set()))
        one[i] = direct + 0.75 * reverse

    for i, nt in enumerate(norms):
        paths = 0
        for a in anchor_norms:
            for mid in outgoing.get(a, set()):
                paths += int(nt in outgoing.get(mid, set()))
                paths += int(a in outgoing.get(mid, set()) and mid == nt)
        central = len(outgoing.get(nt, set())) + len(incoming.get(nt, set()))
        two[i] = one[i] + 0.5 * paths + 0.05 * central

    stats = {
        "anchor_count": float(len(anchors)),
        "edge_count": float(sum(len(v) for v in outgoing.values())),
        "graph_density": float(sum(len(v) for v in outgoing.values()) / max(1, len(titles) * max(1, len(titles) - 1))),
        "twohop_paths": float(two.sum() - one.sum()),
        "onehop_coverage": float(np.mean(one > 0)) if len(one) else 0.0,
        "twohop_coverage": float(np.mean(two > 0)) if len(two) else 0.0,
    }
    return one, two, stats


def score_views(rows: list[dict[str, Any]], link_map: dict[str, set[str]], args: argparse.Namespace) -> list[dict[str, Any]]:
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
            docs = [" ".join(ctx["sentences"])[: args.max_doc_chars] for ctx in row["contexts"]]
            start = cursor
            for text in docs:
                pairs.append((row["question"], text))
                cursor += 1
            row_slices.append((row["id"], start, cursor))
        scores = model.predict(pairs, batch_size=args.ce_batch_size, show_progress_bar=True)
        scores = np.asarray(scores, dtype=np.float32)
        for row_id, start, end in row_slices:
            ce_scores_by_id[row_id] = scores[start:end]

    records: list[dict[str, Any]] = []
    for row in rows:
        titles = [ctx["title"] for ctx in row["contexts"]]
        docs = [" ".join(ctx["sentences"]) for ctx in row["contexts"]]
        gold_titles = set(row["support_titles"])
        q_title_scores = bm25_scores(row["question"], titles)
        full_scores = bm25_scores(row["question"], docs)
        one_graph, two_graph, graph_stats = context_graph_scores(row, link_map)
        # Keep graph views anchored in the visible title sketch.
        one_scores = one_graph + 0.25 * q_title_scores
        two_scores = two_graph + 0.20 * q_title_scores
        ce_scores = ce_scores_by_id.get(row["id"], full_scores.copy())

        shuffled_one = deterministic_rng(row["id"], args.seed, "shuffle1").permutation(one_graph) + 0.25 * q_title_scores
        degree_random_one = random_matched_scores(one_graph, deterministic_rng(row["id"], args.seed, "degree1")) + 0.25 * q_title_scores
        random_two = random_matched_scores(two_graph, deterministic_rng(row["id"], args.seed, "random2")) + 0.20 * q_title_scores

        view_scores = {
            "summary": q_title_scores,
            "one_hop": one_scores,
            "two_hop": two_scores,
            "full_context": full_scores,
            "ce": ce_scores,
        }
        control_scores = {
            "title_only": q_title_scores,
            "real_1hop": one_scores,
            "real_2hop": two_scores,
            "shuffled_1hop": shuffled_one,
            "degree_random_1hop": degree_random_one,
            "random_2hop": random_two,
            "full_context": full_scores,
            "ce": ce_scores,
        }
        view_records = {}
        for view, scores in view_scores.items():
            ranked = rank_from_scores(titles, scores)
            ndcg = ndcg_at_k(ranked, gold_titles, args.topk)
            rec = recall_at_k(ranked, gold_titles, args.topk)
            mrr = mrr_at_k(ranked, gold_titles, args.topk)
            supp_f1 = f1_at_k(ranked, gold_titles, args.topk)
            utility = ndcg - args.lambda_cost * VIEW_COST[view]
            endpoint_gold = set(row["evidence_entities"])
            endpoint_hit = len({norm_title(t) for t in ranked[: args.topk]} & endpoint_gold) / max(1, len(endpoint_gold))
            endpoint_f1 = f1_at_k(ranked, endpoint_gold, args.topk)
            view_records[view] = {
                "view": view,
                "ndcg": float(ndcg),
                "recall": float(rec),
                "mrr": float(mrr),
                "support_f1": float(supp_f1),
                "endpoint_recall": float(endpoint_hit),
                "endpoint_f1": float(endpoint_f1),
                "utility": float(utility),
                "cost": VIEW_COST[view],
                "score_top": float(np.max(scores)) if len(scores) else 0.0,
                "score_margin": margin(scores),
                "score_entropy": entropy_from_scores(scores),
                "ranked_titles": ranked[: args.topk],
            }
        control_records = {}
        for view, scores in control_scores.items():
            ranked = rank_from_scores(titles, scores)
            endpoint_gold = set(row["evidence_entities"])
            control_records[view] = {
                "view": view,
                "ndcg": float(ndcg_at_k(ranked, gold_titles, args.topk)),
                "recall": float(recall_at_k(ranked, gold_titles, args.topk)),
                "support_f1": float(f1_at_k(ranked, gold_titles, args.topk)),
                "endpoint_recall": float(len({norm_title(t) for t in ranked[: args.topk]} & endpoint_gold) / max(1, len(endpoint_gold))),
                "endpoint_f1": float(f1_at_k(ranked, endpoint_gold, args.topk)),
                "utility": float(ndcg_at_k(ranked, gold_titles, args.topk) - args.lambda_cost * CONTROL_COST[view]),
                "cost": CONTROL_COST[view],
            }
        oracle_view = max(VIEW_ORDER, key=lambda v: view_records[v]["utility"])
        doc_lens = [len(tokens(d)) for d in docs]
        records.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "answer": row["answer"],
                "support_titles": sorted(gold_titles),
                "evidences": row["evidences"],
                "n_context": len(titles),
                "question_len": len(tokens(row["question"])),
                "graph_stats": graph_stats,
                "context_stats": {
                    "mean_doc_len": float(np.mean(doc_lens)) if doc_lens else 0.0,
                    "std_doc_len": float(np.std(doc_lens)) if doc_lens else 0.0,
                    "max_doc_len": float(np.max(doc_lens)) if doc_lens else 0.0,
                },
                "views": view_records,
                "controls": control_records,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.60 * n)
    n_dev = int(0.20 * n)
    return idx[:n_train], idx[n_train : n_train + n_dev], idx[n_train + n_dev :]


def fixed_summary(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: list[str]) -> dict[str, Any]:
    utils = []
    regrets = []
    ndcgs = []
    recalls = []
    mrrs = []
    costs = []
    chosen = []
    for pos, i in enumerate(indices):
        rec = records[int(i)]
        view = route[pos] if isinstance(route, list) else route
        vr = rec["views"][view]
        utils.append(vr["utility"])
        regrets.append(rec["oracle_utility"] - vr["utility"])
        ndcgs.append(vr["ndcg"])
        recalls.append(vr["recall"])
        mrrs.append(vr["mrr"])
        costs.append(vr["cost"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utils)) if utils else 0.0,
        "raw_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
        "regret": float(np.mean(regrets)) if regrets else 0.0,
        "ce_buy": float(counts["ce"] / max(1, len(chosen))),
        "view_share": {v: round(counts[v] / max(1, len(chosen)), 4) for v in VIEW_ORDER if counts[v]},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray, tier: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    rows = []
    y = []
    keys: list[tuple[int, str]] = []
    for idx in indices:
        rec = records[int(idx)]
        graph_stats = rec["graph_stats"]
        context_stats = rec["context_stats"]
        summary = rec["views"]["summary"]
        base = [
            rec["question_len"],
            rec["n_context"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
        ]
        if tier in {"B1_entity", "B1_graph", "B1_path", "B1_context"}:
            base += [graph_stats["anchor_count"]]
        if tier in {"B1_graph", "B1_path", "B1_context"}:
            base += [
                graph_stats["edge_count"],
                graph_stats["graph_density"],
                graph_stats["onehop_coverage"],
            ]
        if tier in {"B1_path", "B1_context"}:
            base += [graph_stats["twohop_paths"], graph_stats["twohop_coverage"]]
        if tier == "B1_context":
            # Released context metadata only: lengths and dispersion, not full
            # paragraph text, CE scores, or evaluator qrels.
            base += [
                context_stats["mean_doc_len"],
                context_stats["std_doc_len"],
                context_stats["max_doc_len"],
            ]
        for view_i, view in enumerate(VIEW_ORDER):
            vr = rec["views"][view]
            one_hot = [1.0 if j == view_i else 0.0 for j in range(len(VIEW_ORDER))]
            view_profile = [
                vr["cost"],
                1.0 if view in {"one_hop", "two_hop"} else 0.0,
                1.0 if view == "two_hop" else 0.0,
                1.0 if view in {"full_context", "ce"} else 0.0,
                1.0 if view == "ce" else 0.0,
            ]
            if tier in {"B1_graph", "B1_path", "B1_context"}:
                view_profile += [
                    graph_stats["onehop_coverage"] if view == "one_hop" else 0.0,
                    graph_stats["edge_count"] if view == "one_hop" else 0.0,
                ]
            if tier in {"B1_path", "B1_context"}:
                view_profile += [
                    graph_stats["twohop_coverage"] if view == "two_hop" else 0.0,
                    graph_stats["twohop_paths"] if view == "two_hop" else 0.0,
                ]
            if tier == "B1_context":
                view_profile += [
                    context_stats["mean_doc_len"] if view in {"full_context", "ce"} else 0.0,
                    context_stats["std_doc_len"] if view in {"full_context", "ce"} else 0.0,
                ]
            rows.append(
                base
                + one_hot
                + view_profile
            )
            y.append(vr["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def train_utility_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int, tier: str) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    models = {
        f"ridge_{tier}": Ridge(alpha=1.0),
        f"rf_{tier}": RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=seed, n_jobs=-1),
        f"et_{tier}": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        f"hgb_{tier}": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, random_state=seed),
    }
    x_train, y_train, _ = make_features(records, train_idx, tier)
    x_dev, _, dev_keys = make_features(records, dev_idx, tier)
    x_test, _, test_keys = make_features(records, test_idx, tier)

    def route_from_preds(keys: list[tuple[int, str]], pred: np.ndarray) -> list[str]:
        by_idx: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for (idx, view), val in zip(keys, pred):
            by_idx[idx].append((view, float(val)))
        routes = []
        for idx in sorted(by_idx):
            routes.append(max(by_idx[idx], key=lambda x: x[1])[0])
        return routes

    best_name = None
    best_dev = -1e9
    best_model = None
    for name, model in models.items():
        model.fit(x_train, y_train)
        dev_route = route_from_preds(dev_keys, model.predict(x_dev))
        dev_eval = fixed_summary(records, np.asarray(sorted(set(i for i, _ in dev_keys))), name, dev_route)
        if dev_eval["utility"] > best_dev:
            best_name = name
            best_dev = dev_eval["utility"]
            best_model = model

    assert best_name and best_model
    test_route = route_from_preds(test_keys, best_model.predict(x_test))
    out = fixed_summary(records, np.asarray(sorted(set(i for i, _ in test_keys))), best_name, test_route)
    out["dev_utility"] = float(best_dev)
    out["feature_tier"] = tier
    out["routes"] = test_route
    return out


def train_best_utility_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tier_rows = [
        train_utility_router(records, train_idx, dev_idx, test_idx, seed, tier)
        for tier in FEATURE_TIERS
    ]
    best = max(tier_rows, key=lambda row: row["dev_utility"])
    return best, tier_rows


def choose_threshold_gate(records: list[dict[str, Any]], dev_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    thresholds = [-0.5, 0.0, 0.05, 0.10, 0.20, 0.40, 0.80]

    def route(indices: np.ndarray, t_summary: float, t_graph: float, t_full: float) -> list[str]:
        chosen = []
        for i in indices:
            rec = records[int(i)]
            if rec["views"]["summary"]["score_margin"] >= t_summary:
                chosen.append("summary")
            elif rec["views"]["two_hop"]["score_margin"] >= t_graph:
                chosen.append("two_hop")
            elif rec["views"]["full_context"]["score_margin"] >= t_full:
                chosen.append("full_context")
            else:
                chosen.append("ce")
        return chosen

    best = None
    best_u = -1e9
    for ts in thresholds:
        for tg in thresholds:
            for tf in thresholds:
                r = route(dev_idx, ts, tg, tf)
                ev = fixed_summary(records, dev_idx, "structured_gate", r)
                if ev["utility"] > best_u:
                    best_u = ev["utility"]
                    best = (ts, tg, tf)
    assert best is not None
    test_route = route(test_idx, *best)
    out = fixed_summary(records, test_idx, "structured_gate", test_route)
    out["thresholds"] = best
    out["dev_utility"] = float(best_u)
    return out


def bootstrap_diff(records: list[dict[str, Any]], test_idx: np.ndarray, route_a: str, route_b: str, seed: int, n_boot: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    vals = []
    for i in test_idx:
        rec = records[int(i)]
        vals.append(rec["views"][route_a]["utility"] - rec["views"][route_b]["utility"])
    vals = np.asarray(vals, dtype=np.float64)
    mean = float(vals.mean())
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        diffs.append(float(sample.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean, float(lo), float(hi)


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    train_idx, dev_idx, test_idx = split_indices(len(records), args.seed)
    policy_rows = []
    for view in VIEW_ORDER:
        policy_rows.append(fixed_summary(records, test_idx, f"fixed_{view}", view))
    oracle_route = [records[int(i)]["oracle_view"] for i in test_idx]
    oracle = fixed_summary(records, test_idx, "oracle_route", oracle_route)
    best_fixed = max(policy_rows, key=lambda r: r["utility"])
    gate = choose_threshold_gate(records, dev_idx, test_idx)
    learner, feature_tier_rows = train_best_utility_router(records, train_idx, dev_idx, test_idx, args.seed)
    policy_rows.extend([gate, learner, oracle])

    best_fixed_u = best_fixed["utility"]
    oracle_u = oracle["utility"]
    denom = max(1e-9, oracle_u - best_fixed_u)
    for row in policy_rows:
        row["gap_closed"] = float((row["utility"] - best_fixed_u) / denom)
        row["diff_vs_best_fixed"] = float(row["utility"] - best_fixed_u)

    fixed_diffs = {}
    for view in VIEW_ORDER:
        fixed_diffs[f"{view}_minus_{best_fixed['policy']}"] = bootstrap_diff(
            records, test_idx, view, best_fixed["policy"].replace("fixed_", ""), args.seed
        )

    # Repeated split stability for the best legal learned family.
    repeat_deltas = []
    repeat_best = []
    repeat_tiers = []
    for r in range(args.repeats):
        tr, dv, te = split_indices(len(records), args.seed + r + 1)
        fixeds = [fixed_summary(records, te, f"fixed_{v}", v) for v in VIEW_ORDER]
        bf = max(fixeds, key=lambda x: x["utility"])
        learned, _ = train_best_utility_router(records, tr, dv, te, args.seed + r + 1)
        repeat_deltas.append(learned["utility"] - bf["utility"])
        repeat_best.append(learned["policy"])
        repeat_tiers.append(learned.get("feature_tier", ""))
    if repeat_deltas:
        lo, hi = np.percentile(repeat_deltas, [2.5, 97.5])
        repeat_summary = {
            "mean_learned_minus_best_fixed": float(np.mean(repeat_deltas)),
            "ci95": [float(lo), float(hi)],
            "positive_share": float(np.mean(np.asarray(repeat_deltas) > 0)),
            "selected_learners": dict(Counter(repeat_best)),
            "selected_tiers": dict(Counter(repeat_tiers)),
        }
    else:
        repeat_summary = {}

    for row in feature_tier_rows:
        row["diff_vs_best_fixed"] = float(row["utility"] - best_fixed_u)
        row["gap_closed"] = float((row["utility"] - best_fixed_u) / denom)

    oracle_shares = Counter(records[int(i)]["oracle_view"] for i in test_idx)
    control_rows = []
    best_control_u = -1e9
    for control in CONTROL_ORDER:
        utils = []
        ndcgs = []
        recalls = []
        support_f1s = []
        endpoint_f1s = []
        regrets = []
        for i in test_idx:
            rec = records[int(i)]
            cv = rec["controls"][control]
            utils.append(cv["utility"])
            ndcgs.append(cv["ndcg"])
            recalls.append(cv["recall"])
            support_f1s.append(cv["support_f1"])
            endpoint_f1s.append(cv["endpoint_f1"])
            regrets.append(rec["oracle_utility"] - cv["utility"])
        row = {
            "control": control,
            "utility": float(np.mean(utils)),
            "raw_ndcg": float(np.mean(ndcgs)),
            "support_recall": float(np.mean(recalls)),
            "support_f1": float(np.mean(support_f1s)),
            "triple_endpoint_f1": float(np.mean(endpoint_f1s)),
            "cost": CONTROL_COST[control],
            "regret": float(np.mean(regrets)),
        }
        best_control_u = max(best_control_u, row["utility"])
        control_rows.append(row)
    for row in control_rows:
        row["diff_vs_best_control"] = float(row["utility"] - best_control_u)

    learner_routes = learner.get("routes", [])
    type_rows = []
    by_type: dict[str, list[int]] = defaultdict(list)
    sorted_test = list(map(int, test_idx))
    route_by_idx = {idx: learner_routes[pos] for pos, idx in enumerate(sorted(set(sorted_test))) if pos < len(learner_routes)}
    for i in sorted_test:
        by_type[str(records[i]["type"])].append(i)
    for qtype, idxs in sorted(by_type.items()):
        fixed_utils = {
            v: float(np.mean([records[i]["views"][v]["utility"] for i in idxs]))
            for v in VIEW_ORDER
        }
        best_v, best_u = max(fixed_utils.items(), key=lambda x: x[1])
        learned_u = float(np.mean([records[i]["views"][route_by_idx.get(i, best_v)]["utility"] for i in idxs]))
        oracle_u_type = float(np.mean([records[i]["oracle_utility"] for i in idxs]))
        type_rows.append(
            {
                "type": qtype,
                "n": len(idxs),
                "best_fixed": best_v,
                "fixed_utility": best_u,
                "adaptive_utility": learned_u,
                "oracle_utility": oracle_u_type,
                "gain": learned_u - best_u,
                "oracle_gap": oracle_u_type - best_u,
            }
        )
    hard_rows = []
    for bucket_name, key in [
        ("low title margin", ("summary", "score_margin")),
        ("high graph density", None),
        ("high full-context entropy", ("full_context", "score_entropy")),
    ]:
        if key is None:
            vals = np.asarray([records[int(i)]["graph_stats"]["graph_density"] for i in test_idx])
        else:
            vals = np.asarray([records[int(i)]["views"][key[0]][key[1]] for i in test_idx])
        if bucket_name.startswith("low"):
            mask = vals <= np.quantile(vals, 0.33)
        else:
            mask = vals >= np.quantile(vals, 0.67)
        idxs = test_idx[mask]
        if len(idxs) == 0:
            continue
        ce_worth = np.mean(
            [
                records[int(i)]["views"]["ce"]["utility"]
                > max(records[int(i)]["views"][v]["utility"] for v in VIEW_ORDER if v != "ce")
                for i in idxs
            ]
        )
        hard_rows.append(
            {
                "bucket": bucket_name,
                "share": float(len(idxs) / len(test_idx)),
                "ce_worth_share": float(ce_worth),
                "oracle_regret_fixed_best": float(np.mean([records[int(i)]["oracle_utility"] - records[int(i)]["views"][best_fixed["policy"].replace("fixed_", "")]["utility"] for i in idxs])),
                "summary_ndcg": float(np.mean([records[int(i)]["views"]["summary"]["ndcg"] for i in idxs])),
                "two_hop_ndcg": float(np.mean([records[int(i)]["views"]["two_hop"]["ndcg"] for i in idxs])),
                "ce_ndcg": float(np.mean([records[int(i)]["views"]["ce"]["ndcg"] for i in idxs])),
            }
        )

    return {
        "task": "2wiki_structured_evidence_access",
        "dataset": "voidful/2wikimultihopqa",
        "n_rows": len(records),
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "view_costs": VIEW_COST,
        "views": {
            "summary": "title/entity sketch from question and candidate titles",
            "one_hop": "paid hyperlink edges among candidate titles",
            "two_hop": "paid two-hop hyperlink paths among candidate titles",
            "full_context": "paid full context paragraphs",
            "ce": "paid cross-encoder rerank over context paragraphs",
        },
        "qrels": "Evaluator-held 2Wiki supporting_facts and evidence triples; primary score is support-title NDCG@k.",
        "policy_rows": policy_rows,
        "best_fixed": best_fixed["policy"],
        "feature_tier_rows": feature_tier_rows,
        "control_rows": control_rows,
        "type_rows": type_rows,
        "oracle_view_share": {k: float(v / len(test_idx)) for k, v in oracle_shares.items()},
        "fixed_bootstrap_diffs": fixed_diffs,
        "repeat_summary": repeat_summary,
        "hard_bucket_rows": hard_rows,
    }


def build_protocol_trace(records: list[dict[str, Any]], result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Create one compact legal/illegal Protocol B trace for the paper/artifact.

    The trace is generated from an audited row rather than hand-written so the
    visible fields, route, costs, and hidden fields stay synchronized with the
    structured-evidence slice.
    """
    if not records:
        return {}
    # Prefer a row where graph evidence changes the cheap summary decision.
    chosen = None
    for rec in records:
        if rec["views"]["one_hop"]["utility"] > rec["views"]["summary"]["utility"] + 0.05:
            chosen = rec
            break
    if chosen is None:
        chosen = records[0]

    titles = []
    # The ranked titles are the method-visible title/entity sketch; do not
    # expose support flags or evidence-triple labels in this field.
    seen = set()
    for view in ["summary", "one_hop", "two_hop"]:
        for title in chosen["views"][view]["ranked_titles"]:
            nt = norm_title(title)
            if nt not in seen:
                titles.append(title)
                seen.add(nt)
            if len(titles) >= args.topk:
                break
        if len(titles) >= args.topk:
            break

    legal_route = max(["summary", "one_hop", "two_hop"], key=lambda v: chosen["views"][v]["utility"])
    legal_submission = {
        "query_id": chosen["id"],
        "cell_id": f"2wiki_{safe_id(chosen['id'])}",
        "tier": "B1_path",
        "cost_menu": "2wiki-structured-v1-op",
        "ranked_views": list(dict.fromkeys([legal_route, "summary", "one_hop", "two_hop"])),
        "route": legal_route,
    }
    illegal_submission = {
        "query_id": chosen["id"],
        "cell_id": f"2wiki_{safe_id(chosen['id'])}",
        "tier": "B1_path",
        "cost_menu": "2wiki-structured-v1-op",
        "route": "ce",
        "ce_score_margin": round(chosen["views"]["ce"]["score_margin"], 4),
        "support_titles": chosen["support_titles"],
    }
    return {
        "visible_fields": {
            "query_id": chosen["id"],
            "question": chosen["question"],
            "tier": "B1_path",
            "cost_menu": "2wiki-structured-v1-op",
            "declared_views": VIEW_ORDER,
            "visible_title_sketch": titles,
            "released_graph_summaries": [
                "anchor_count",
                "edge_count",
                "graph_density",
                "onehop_coverage",
                "twohop_paths",
                "twohop_coverage",
            ],
        },
        "legal_action": {
            "allowed_purchase": legal_route,
            "submitted_jsonl": legal_submission,
            "charged_cost": VIEW_COST[legal_route],
            "scored_utility": chosen["views"][legal_route]["utility"],
        },
        "hidden_evaluator_fields": {
            "support_titles": chosen["support_titles"],
            "evidence_triples": chosen["evidences"],
            "answer": chosen["answer"],
            "ce_scores": "hidden unless CE is bought",
        },
        "illegal_variant": {
            "submitted_jsonl": illegal_submission,
            "rejection_reason": "CE score margin and support titles are evaluator-only before the CE/support view is bought.",
        },
        "scope_note": "This trace scores support/path evidence acquisition, not generated answer faithfulness or an end-to-end agent transcript.",
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"2wiki_structured_evidence_{args.max_rows}"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    compact_records = []
    for rec in records[: args.example_rows]:
        compact_records.append(
            {
                "id": rec["id"],
                "question": rec["question"],
                "answer": rec["answer"],
                "support_titles": rec["support_titles"],
                "oracle_view": rec["oracle_view"],
                "view_ndcg": {v: round(rec["views"][v]["ndcg"], 4) for v in VIEW_ORDER},
                "top_titles": {v: rec["views"][v]["ranked_titles"][: args.topk] for v in VIEW_ORDER},
                "evidences": rec["evidences"],
            }
        )
    payload = dict(result)
    payload["examples"] = compact_records
    payload["protocol_trace"] = build_protocol_trace(records, result, args)
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)

    policy_cols = ["policy", "utility", "raw_ndcg", "recall", "cost", "regret", "gap_closed", "ce_buy", "view_share"]
    tier_cols = ["feature_tier", "policy", "utility", "regret", "gap_closed", "diff_vs_best_fixed", "ce_buy", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "support_recall", "support_f1", "triple_endpoint_f1", "cost", "regret", "diff_vs_best_control"]
    type_cols = ["type", "n", "best_fixed", "fixed_utility", "adaptive_utility", "oracle_utility", "gain", "oracle_gap"]
    hard_cols = ["bucket", "share", "ce_worth_share", "oracle_regret_fixed_best", "summary_ndcg", "two_hop_ndcg", "ce_ndcg"]
    md = [
        "# 2Wiki Structured Evidence Access Audit",
        "",
        "Protocol B structured-evidence slice over `voidful/2wikimultihopqa`. A method sees a question plus a cheap title/entity sketch and may buy one of five declared views: summary, one-hop hyperlinks, two-hop paths, full context, or cross-encoder reranking. Evaluator-held supporting facts and evidence triples define the qrels; the primary score is support-title NDCG@k with cost-adjusted utility.",
        "",
        f"- Rows: {result['n_rows']} ({result['split_sizes']})",
        f"- Top-k: {result['topk']}; lambda: {result['lambda_cost']}",
        f"- View costs: {result['view_costs']}",
        f"- Oracle view share on hidden test: {result['oracle_view_share']}",
        "",
        "## Fixed and adaptive policies",
        "",
        markdown_table(result["policy_rows"], policy_cols),
        "",
        "## Legal feature-tier learner ablation",
        "",
        "Rows use only released features at the named tier; paid CE/full/view scores are not visible before purchase.",
        "",
        markdown_table(result["feature_tier_rows"], tier_cols),
        "",
        "## Anti-token and path controls",
        "",
        markdown_table(result["control_rows"], control_cols),
        "",
        "## Question-type breakdown",
        "",
        markdown_table(result["type_rows"], type_cols),
        "",
        "## Repeated split stability",
        "",
        "```json",
        json.dumps(result["repeat_summary"], indent=2),
        "```",
        "",
        "## Hard structured-evidence buckets",
        "",
        markdown_table(result["hard_bucket_rows"], hard_cols),
        "",
        "## Protocol B legal/illegal trace",
        "",
        "The JSON report contains a generated trace with visible fields, an allowed 2Wiki view purchase, the submitted JSONL route, charged cost, hidden evaluator fields, and an illegal variant containing unpaid CE/support fields. This is an evidence-acquisition trace rather than an end-to-end RAG transcript.",
        "",
        "## Notes",
        "",
        "- This is a structured evidence-acquisition slice, not an end-to-end RAG or generation benchmark.",
        "- The hyperlink corpus is used only to construct paid 1-hop/2-hop graph views; gold supporting facts and evidence triples remain evaluator-held.",
        "- CE scores are treated as paid evidence and are not visible to summary/graph/full-context routers.",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--hyperlink-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--max-doc-chars", type=int, default=1800)
    parser.add_argument("--no-ce", action="store_true")
    parser.add_argument("--example-rows", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("HF_HOME", str(args.hf_cache.parent))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_cache))

    rows = load_2wiki_rows(args.max_rows, args.hf_cache)
    title_hash = hashlib.sha1("\n".join(sorted({ctx["title"] for row in rows for ctx in row["contexts"]})).encode("utf-8")).hexdigest()[:10]
    cache_path = REPORTS / f"2wiki_title_links_{args.max_rows}_{title_hash}.json"
    link_map = build_link_map(rows, args.hyperlink_corpus, cache_path)
    records = score_views(rows, link_map, args)
    result = summarize(records, args)
    result["hyperlink_cache"] = str(cache_path)
    json_path, md_path = write_outputs(result, records, args)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "n_rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
