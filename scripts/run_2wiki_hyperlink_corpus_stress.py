from __future__ import annotations

import argparse
import hashlib
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
    DEFAULT_CORPUS,
    DEFAULT_HF_CACHE,
    REPORTS,
    TITLE_FIELD_RE,
    bm25_scores,
    deterministic_rng,
    entropy_from_scores,
    f1_at_k,
    load_2wiki_rows,
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


VIEW_ORDER = ["summary", "provided_context", "hyperlink_1hop", "hyperlink_2hop", "full_local_pool"]
VIEW_COST = {
    "summary": 0.0,
    "provided_context": 0.08,
    "hyperlink_1hop": 0.20,
    "hyperlink_2hop": 0.34,
    "full_local_pool": 0.55,
}
CONTROL_ORDER = [
    "title_only",
    "provided_context",
    "real_1hop",
    "shuffled_1hop",
    "degree_random_1hop",
    "real_2hop",
    "random_2hop",
    "full_local_pool",
]
CONTROL_COST = {
    "title_only": VIEW_COST["summary"],
    "provided_context": VIEW_COST["provided_context"],
    "real_1hop": VIEW_COST["hyperlink_1hop"],
    "shuffled_1hop": VIEW_COST["hyperlink_1hop"],
    "degree_random_1hop": VIEW_COST["hyperlink_1hop"],
    "real_2hop": VIEW_COST["hyperlink_2hop"],
    "random_2hop": VIEW_COST["hyperlink_2hop"],
    "full_local_pool": VIEW_COST["full_local_pool"],
}


def json_title_from_line(line: str) -> str | None:
    match = TITLE_FIELD_RE.search(line)
    if not match:
        return None
    try:
        return norm_title(json.loads('"' + match.group(1) + '"'))
    except json.JSONDecodeError:
        return None


def corpus_row_to_doc(obj: dict[str, Any]) -> dict[str, Any]:
    links = set()
    for mention in obj.get("mentions", []) or []:
        ref_url = str(mention.get("ref_url", "")).strip()
        if ref_url:
            links.add(norm_title(ref_url))
    sentences = [str(s) for s in obj.get("sentences", [])]
    return {
        "id": str(obj.get("id", "")),
        "title": str(obj.get("title", "")),
        "norm_title": norm_title(obj.get("title", "")),
        "sentences": sentences,
        "text": " ".join(sentences),
        "links": links,
        "degree": len(links),
    }


def scan_corpus_for_titles(corpus_path: Path, needed: set[str], cache_path: Path, pass_name: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if cache_path.exists():
        payload = json.load(cache_path.open("r", encoding="utf-8"))
        if set(payload.get("needed_titles", [])) == needed:
            docs = {}
            for title, doc in payload["docs"].items():
                doc["links"] = set(doc.get("links", []))
                docs[title] = doc
            return docs, payload.get("profile", {})

    start = time.time()
    docs: dict[str, dict[str, Any]] = {}
    lines = 0
    bytes_seen = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            lines += 1
            bytes_seen += len(line)
            title = json_title_from_line(line)
            if title not in needed:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs[title] = corpus_row_to_doc(obj)
            if len(docs) == len(needed):
                break

    seconds = time.time() - start
    profile = {
        "pass": pass_name,
        "needed": len(needed),
        "found": len(docs),
        "seconds": round(seconds, 3),
        "lines_scanned": lines,
        "bytes_scanned": bytes_seen,
        "scan_mb_per_s": round(bytes_seen / max(seconds, 1e-9) / 1e6, 3),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": str(corpus_path),
            "needed_titles": sorted(needed),
            "profile": profile,
            "docs": {
                title: {
                    **doc,
                    "links": sorted(doc["links"]),
                }
                for title, doc in docs.items()
            },
        },
        cache_path.open("w", encoding="utf-8"),
        indent=2,
    )
    return docs, profile


def title_overlap_score(question: str, title: str) -> float:
    q = token_set(question)
    t = token_set(title)
    if not t:
        return 0.0
    overlap = len(q & t)
    score = overlap / max(1, len(t))
    if norm_title(title) in norm_title(question):
        score += 1.0
    return float(score)


def context_anchors(row: dict[str, Any]) -> list[str]:
    scored = []
    for ctx in row["contexts"]:
        title = ctx["title"]
        scored.append((title_overlap_score(row["question"], title), norm_title(title)))
    anchors = [title for score, title in scored if score >= 0.45]
    if not anchors and scored:
        anchors = [max(scored, key=lambda x: x[0])[1]]
    return anchors


def choose_1hop_titles(row: dict[str, Any], seed_docs: dict[str, dict[str, Any]], cap: int) -> list[str]:
    context_titles = {norm_title(ctx["title"]) for ctx in row["contexts"]}
    anchors = set(context_anchors(row))
    counter: Counter[str] = Counter()
    anchor_counter: Counter[str] = Counter()
    for ctx in row["contexts"]:
        src = norm_title(ctx["title"])
        doc = seed_docs.get(src)
        if not doc:
            continue
        for dst in doc["links"]:
            if dst in context_titles:
                continue
            counter[dst] += 1
            if src in anchors:
                anchor_counter[dst] += 1
    ranked = sorted(
        counter,
        key=lambda t: (
            -anchor_counter[t],
            -counter[t],
            -title_overlap_score(row["question"], t),
            t,
        ),
    )
    return ranked[:cap]


def choose_2hop_titles(row: dict[str, Any], first_titles: list[str], first_docs: dict[str, dict[str, Any]], cap: int) -> list[str]:
    context_titles = {norm_title(ctx["title"]) for ctx in row["contexts"]}
    first_set = set(first_titles)
    counter: Counter[str] = Counter()
    for mid in first_titles:
        doc = first_docs.get(mid)
        if not doc:
            continue
        for dst in doc["links"]:
            if dst in context_titles or dst in first_set:
                continue
            counter[dst] += 1
    ranked = sorted(
        counter,
        key=lambda t: (
            -counter[t],
            -title_overlap_score(row["question"], t),
            t,
        ),
    )
    return ranked[:cap]


def context_doc_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        for ctx in row["contexts"]:
            title = norm_title(ctx["title"])
            docs[title] = {
                "id": f"context::{title}",
                "title": ctx["title"],
                "norm_title": title,
                "sentences": ctx["sentences"],
                "text": " ".join(ctx["sentences"]),
                "links": set(),
                "degree": 0,
            }
    return docs


def build_local_expansion(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    seed_titles = {norm_title(ctx["title"]) for row in rows for ctx in row["contexts"]}
    seed_hash = hashlib.sha1("\n".join(sorted(seed_titles)).encode("utf-8")).hexdigest()[:10]
    seed_docs, seed_profile = scan_corpus_for_titles(
        args.hyperlink_corpus,
        seed_titles,
        REPORTS / f"2wiki_hyper_seed_{args.max_rows}_{seed_hash}.json",
        "seed_context_titles",
    )

    first_by_row: dict[str, list[str]] = {}
    first_needed: set[str] = set()
    for row in rows:
        first = choose_1hop_titles(row, seed_docs, args.max_1hop)
        first_by_row[row["id"]] = first
        first_needed.update(first)

    first_hash = hashlib.sha1("\n".join(sorted(first_needed)).encode("utf-8")).hexdigest()[:10]
    first_docs, first_profile = scan_corpus_for_titles(
        args.hyperlink_corpus,
        first_needed,
        REPORTS / f"2wiki_hyper_1hop_{args.max_rows}_{args.max_1hop}_{first_hash}.json",
        "selected_1hop_titles",
    )

    second_by_row: dict[str, list[str]] = {}
    second_needed: set[str] = set()
    for row in rows:
        second = choose_2hop_titles(row, first_by_row[row["id"]], first_docs, args.max_2hop)
        second_by_row[row["id"]] = second
        second_needed.update(second)

    second_hash = hashlib.sha1("\n".join(sorted(second_needed)).encode("utf-8")).hexdigest()[:10]
    second_docs, second_profile = scan_corpus_for_titles(
        args.hyperlink_corpus,
        second_needed,
        REPORTS / f"2wiki_hyper_2hop_{args.max_rows}_{args.max_1hop}_{args.max_2hop}_{second_hash}.json",
        "selected_2hop_titles",
    )

    docs = context_doc_map(rows)
    docs.update(seed_docs)
    docs.update(first_docs)
    docs.update(second_docs)
    row_specs = []
    for row in rows:
        row_specs.append(
            {
                "id": row["id"],
                "context_titles": [norm_title(ctx["title"]) for ctx in row["contexts"]],
                "anchors": context_anchors(row),
                "first_titles": first_by_row[row["id"]],
                "second_titles": second_by_row[row["id"]],
            }
        )
    profile = {
        "corpus": str(args.hyperlink_corpus),
        "seed_pass": seed_profile,
        "first_hop_pass": first_profile,
        "second_hop_pass": second_profile,
        "selected_1hop_unique": len(first_needed),
        "selected_2hop_unique": len(second_needed),
        "doc_map_size": len(docs),
        "mean_1hop_per_query": float(np.mean([len(x["first_titles"]) for x in row_specs])) if row_specs else 0.0,
        "mean_2hop_per_query": float(np.mean([len(x["second_titles"]) for x in row_specs])) if row_specs else 0.0,
    }
    return profile, docs, row_specs


def graph_scores(
    question: str,
    candidates: list[str],
    context_titles: list[str],
    anchors: list[str],
    first_titles: list[str],
    docs: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_set = set(candidates)
    first_set = set(first_titles)
    anchors = anchors or context_titles[:1]
    one = np.zeros(len(candidates), dtype=np.float32)
    two = np.zeros(len(candidates), dtype=np.float32)
    idx = {title: i for i, title in enumerate(candidates)}

    for src in context_titles:
        doc = docs.get(src)
        if not doc:
            continue
        links = doc["links"]
        for dst in links & candidate_set:
            weight = 1.0 if src in anchors else 0.35
            one[idx[dst]] += weight
        if src in anchors:
            for dst in candidate_set:
                if dst in links:
                    one[idx[dst]] += 0.5

    for mid in first_set:
        doc = docs.get(mid)
        if not doc:
            continue
        links = doc["links"]
        mid_weight = 1.0 if mid in idx and one[idx[mid]] > 0 else 0.5
        for dst in links & candidate_set:
            two[idx[dst]] += 0.6 * mid_weight

    for i, title in enumerate(candidates):
        doc = docs.get(title)
        degree = float(doc["degree"] if doc else 0)
        two[i] += one[i] + 0.01 * min(degree, 50.0) + 0.05 * title_overlap_score(question, title)
    return one, two


def dedup_titles(titles: list[str]) -> list[str]:
    out = []
    seen = set()
    for title in titles:
        nt = norm_title(title)
        if not nt or nt in seen:
            continue
        out.append(nt)
        seen.add(nt)
    return out


def score_records(rows: list[dict[str, Any]], docs: dict[str, dict[str, Any]], row_specs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    spec_by_id = {spec["id"]: spec for spec in row_specs}
    universe = sorted({title for spec in row_specs for title in spec["first_titles"] + spec["second_titles"] if title in docs})
    degree_by_title = {title: docs.get(title, {}).get("degree", 0) for title in universe}
    records = []
    for row in rows:
        spec = spec_by_id[row["id"]]
        context_titles = dedup_titles(spec["context_titles"])
        first_titles = [t for t in dedup_titles(spec["first_titles"]) if t in docs]
        second_titles = [t for t in dedup_titles(spec["second_titles"]) if t in docs]
        one_candidates = dedup_titles(context_titles + first_titles)
        two_candidates = dedup_titles(context_titles + first_titles + second_titles)
        gold_titles = set(row["support_titles"])
        endpoint_gold = set(row["evidence_entities"])

        def texts(titles: list[str], use_title: bool = False) -> list[str]:
            vals = []
            for title in titles:
                doc = docs.get(title, {})
                if use_title:
                    vals.append(str(doc.get("title", title)))
                else:
                    vals.append(str(doc.get("text", doc.get("title", title))))
            return vals

        def display_titles(titles: list[str]) -> list[str]:
            return [str(docs.get(t, {}).get("title", t)) for t in titles]

        summary_scores = bm25_scores(row["question"], texts(context_titles, use_title=True))
        context_scores = bm25_scores(row["question"], texts(context_titles))
        one_title = bm25_scores(row["question"], texts(one_candidates, use_title=True))
        one_text = bm25_scores(row["question"], texts(one_candidates))
        two_title = bm25_scores(row["question"], texts(two_candidates, use_title=True))
        two_text = bm25_scores(row["question"], texts(two_candidates))
        one_graph, two_graph = graph_scores(row["question"], one_candidates, context_titles, spec["anchors"], first_titles, docs)
        _, two_graph_full = graph_scores(row["question"], two_candidates, context_titles, spec["anchors"], first_titles, docs)
        full_scores = bm25_scores(row["question"], texts(two_candidates))
        rng = deterministic_rng(row["id"], args.seed, "hyper-controls")

        # Protocol B scores support-title acquisition.  The external hyperlink
        # corpus is a paid evidence view used to rerank the context candidates;
        # external paragraphs are evidence, not additional gold-labeled targets.
        def expanded_context_texts(depth: int, randomize: bool = False) -> list[str]:
            out = []
            rng_local = deterministic_rng(row["id"], args.seed + depth, f"context-expand-{randomize}")
            random_pool = [t for t in universe if t not in set(context_titles)]
            for ctx_title in context_titles:
                base_doc = docs.get(ctx_title, {})
                linked = []
                if randomize:
                    take = min(args.random_evidence_per_context, len(random_pool))
                    linked = list(rng_local.choice(random_pool, size=take, replace=False)) if take else []
                else:
                    ctx_links = list((base_doc.get("links", set()) or set()) & set(first_titles))
                    ctx_links = sorted(ctx_links, key=lambda t: (-title_overlap_score(row["question"], t), t))
                    linked.extend(ctx_links[: args.evidence_per_context])
                    if depth >= 2:
                        second = []
                        for mid in ctx_links[: args.evidence_per_context]:
                            mid_doc = docs.get(mid, {})
                            second.extend(list((mid_doc.get("links", set()) or set()) & set(second_titles)))
                        second = sorted(set(second), key=lambda t: (-title_overlap_score(row["question"], t), t))
                        linked.extend(second[: args.evidence_per_context])
                text_parts = [str(base_doc.get("text", base_doc.get("title", ctx_title)))]
                text_parts.extend(str(docs.get(t, {}).get("text", docs.get(t, {}).get("title", t))) for t in linked if t in docs)
                out.append(" ".join(text_parts))
            return out

        one_context_scores = bm25_scores(row["question"], expanded_context_texts(depth=1)) + 0.20 * summary_scores
        two_context_scores = bm25_scores(row["question"], expanded_context_texts(depth=2)) + 0.18 * summary_scores
        random_one_context_scores = bm25_scores(row["question"], expanded_context_texts(depth=1, randomize=True)) + 0.20 * summary_scores
        random_two_context_scores = bm25_scores(row["question"], expanded_context_texts(depth=2, randomize=True)) + 0.18 * summary_scores
        shuffled_one_context_scores = rng.permutation(one_context_scores) if len(one_context_scores) else one_context_scores

        shuffled_one_graph = rng.permutation(one_graph) if len(one_graph) else one_graph
        random_one_titles = [t for t in degree_matched_random(first_titles, universe, degree_by_title, set(context_titles), rng, len(first_titles)) if t in docs]
        random_one_candidates = dedup_titles(context_titles + random_one_titles)
        random_one_title = bm25_scores(row["question"], texts(random_one_candidates, use_title=True))
        random_one_text = bm25_scores(row["question"], texts(random_one_candidates))
        random_one_graph = random_matched_scores(one_graph, rng)[: len(random_one_candidates)]
        if len(random_one_graph) < len(random_one_candidates):
            random_one_graph = np.pad(random_one_graph, (0, len(random_one_candidates) - len(random_one_graph)))

        random_two_titles = [t for t in degree_matched_random(second_titles, universe, degree_by_title, set(context_titles + first_titles), rng, len(second_titles)) if t in docs]
        random_two_candidates = dedup_titles(context_titles + random_one_titles + random_two_titles)
        random_two_title = bm25_scores(row["question"], texts(random_two_candidates, use_title=True))
        random_two_text = bm25_scores(row["question"], texts(random_two_candidates))
        random_two_graph = random_matched_scores(two_graph_full, rng)[: len(random_two_candidates)]
        if len(random_two_graph) < len(random_two_candidates):
            random_two_graph = np.pad(random_two_graph, (0, len(random_two_candidates) - len(random_two_graph)))

        view_def = {
            "summary": (context_titles, summary_scores),
            "provided_context": (context_titles, context_scores + 0.15 * summary_scores),
            "hyperlink_1hop": (context_titles, one_context_scores),
            "hyperlink_2hop": (context_titles, two_context_scores),
            "full_local_pool": (context_titles, bm25_scores(row["question"], expanded_context_texts(depth=2)) + 0.10 * context_scores),
        }
        control_def = {
            "title_only": (context_titles, summary_scores),
            "provided_context": (context_titles, context_scores + 0.15 * summary_scores),
            "real_1hop": view_def["hyperlink_1hop"],
            "shuffled_1hop": (context_titles, shuffled_one_context_scores),
            "degree_random_1hop": (context_titles, random_one_context_scores),
            "real_2hop": view_def["hyperlink_2hop"],
            "random_2hop": (context_titles, random_two_context_scores),
            "full_local_pool": view_def["full_local_pool"],
        }

        view_records = {}
        for view, (candidate_titles, scores) in view_def.items():
            ranked_norm = rank_from_scores(candidate_titles, scores)
            ranked = display_titles(ranked_norm)
            view_records[view] = metrics_for_rank(row, view, ranked, scores, gold_titles, endpoint_gold, args.lambda_cost, VIEW_COST[view], args.topk)
        control_records = {}
        for view, (candidate_titles, scores) in control_def.items():
            ranked_norm = rank_from_scores(candidate_titles, scores)
            ranked = display_titles(ranked_norm)
            control_records[view] = metrics_for_rank(row, view, ranked, scores, gold_titles, endpoint_gold, args.lambda_cost, CONTROL_COST[view], args.topk)

        oracle_view = max(VIEW_ORDER, key=lambda v: view_records[v]["utility"])
        records.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "answer": row["answer"],
                "support_titles": sorted(gold_titles),
                "evidences": row["evidences"],
                "n_context": len(context_titles),
                "n_1hop": len(first_titles),
                "n_2hop": len(second_titles),
                "n_local_pool": len(two_candidates),
                "question_len": len(tokens(row["question"])),
                "graph_stats": {
                    "anchor_count": float(len(spec["anchors"])),
                    "onehop_count": float(len(first_titles)),
                    "twohop_count": float(len(second_titles)),
                    "local_pool_count": float(len(two_candidates)),
                    "onehop_found_share": float(len(first_titles) / max(1, len(spec["first_titles"]))),
                    "twohop_found_share": float(len(second_titles) / max(1, len(spec["second_titles"]))),
                    "mean_1hop_degree": float(np.mean([docs[t]["degree"] for t in first_titles])) if first_titles else 0.0,
                    "mean_2hop_degree": float(np.mean([docs[t]["degree"] for t in second_titles])) if second_titles else 0.0,
                    "one_graph_coverage": float(np.mean(one_graph > 0)) if len(one_graph) else 0.0,
                    "two_graph_coverage": float(np.mean(two_graph_full > 0)) if len(two_graph_full) else 0.0,
                },
                "views": view_records,
                "controls": control_records,
                "oracle_view": oracle_view,
                "oracle_utility": view_records[oracle_view]["utility"],
            }
        )
    return records


def degree_matched_random(
    source_titles: list[str],
    universe: list[str],
    degree_by_title: dict[str, int],
    excluded: set[str],
    rng: np.random.Generator,
    count: int,
) -> list[str]:
    if count <= 0 or not universe:
        return []
    candidates = [t for t in universe if t not in excluded]
    if not candidates:
        return []
    source_degrees = [degree_by_title.get(t, 0) for t in source_titles] or [0]
    chosen = []
    unused = set(candidates)
    for deg in source_degrees[:count]:
        if not unused:
            break
        nearest = sorted(unused, key=lambda t: (abs(degree_by_title.get(t, 0) - deg), t))[: min(64, len(unused))]
        pick = str(rng.choice(nearest))
        chosen.append(pick)
        unused.remove(pick)
    while len(chosen) < count and unused:
        pick = str(rng.choice(list(unused)))
        chosen.append(pick)
        unused.remove(pick)
    return chosen


def metrics_for_rank(
    row: dict[str, Any],
    view: str,
    ranked_titles: list[str],
    scores: np.ndarray,
    gold_titles: set[str],
    endpoint_gold: set[str],
    lambda_cost: float,
    cost: float,
    topk: int,
) -> dict[str, Any]:
    ranked_norm = [norm_title(t) for t in ranked_titles]
    ndcg = ndcg_at_k(ranked_titles, gold_titles, topk)
    endpoint_hit = len(set(ranked_norm[:topk]) & endpoint_gold) / max(1, len(endpoint_gold))
    return {
        "view": view,
        "ndcg": float(ndcg),
        "recall": float(recall_at_k(ranked_titles, gold_titles, topk)),
        "support_f1": float(f1_at_k(ranked_titles, gold_titles, topk)),
        "endpoint_recall": float(endpoint_hit),
        "endpoint_f1": float(f1_at_k(ranked_titles, endpoint_gold, topk)),
        "utility": float(ndcg - lambda_cost * cost),
        "cost": cost,
        "score_top": float(np.max(scores)) if len(scores) else 0.0,
        "score_margin": margin(scores),
        "score_entropy": entropy_from_scores(scores),
        "ranked_titles": ranked_titles[:topk],
    }


def fixed_policy(records: list[dict[str, Any]], indices: np.ndarray, name: str, route: str | list[str]) -> dict[str, Any]:
    utils, regrets, ndcgs, recalls, costs, chosen = [], [], [], [], [], []
    for pos, idx in enumerate(indices):
        rec = records[int(idx)]
        view = route[pos] if isinstance(route, list) else route
        vr = rec["views"][view]
        utils.append(vr["utility"])
        regrets.append(rec["oracle_utility"] - vr["utility"])
        ndcgs.append(vr["ndcg"])
        recalls.append(vr["recall"])
        costs.append(vr["cost"])
        chosen.append(view)
    counts = Counter(chosen)
    return {
        "policy": name,
        "utility": float(np.mean(utils)) if utils else 0.0,
        "raw_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "cost": float(np.mean(costs)) if costs else 0.0,
        "regret": float(np.mean(regrets)) if regrets else 0.0,
        "view_share": {v: round(counts[v] / max(1, len(chosen)), 4) for v in VIEW_ORDER if counts[v]},
    }


def make_features(records: list[dict[str, Any]], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    rows, y, keys = [], [], []
    for idx in indices:
        rec = records[int(idx)]
        summary = rec["views"]["summary"]
        gs = rec["graph_stats"]
        base = [
            rec["question_len"],
            rec["n_context"],
            rec["n_1hop"],
            rec["n_2hop"],
            rec["n_local_pool"],
            summary["score_top"],
            summary["score_margin"],
            summary["score_entropy"],
            gs["anchor_count"],
            gs["onehop_count"],
            gs["twohop_count"],
            gs["local_pool_count"],
            gs["onehop_found_share"],
            gs["twohop_found_share"],
            gs["mean_1hop_degree"],
            gs["mean_2hop_degree"],
            gs["one_graph_coverage"],
            gs["two_graph_coverage"],
        ]
        for view_i, view in enumerate(VIEW_ORDER):
            vr = rec["views"][view]
            rows.append(
                base
                + [1.0 if i == view_i else 0.0 for i in range(len(VIEW_ORDER))]
                + [
                    vr["cost"],
                    1.0 if view == "provided_context" else 0.0,
                    1.0 if view == "hyperlink_1hop" else 0.0,
                    1.0 if view == "hyperlink_2hop" else 0.0,
                    1.0 if view == "full_local_pool" else 0.0,
                ]
            )
            y.append(vr["utility"])
            keys.append((int(idx), view))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


def route_from_predictions(keys: list[tuple[int, str]], preds: np.ndarray) -> list[str]:
    by_idx: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (idx, view), pred in zip(keys, preds):
        by_idx[idx].append((view, float(pred)))
    return [max(by_idx[idx], key=lambda x: (x[1], -VIEW_ORDER.index(x[0])))[0] for idx in sorted(by_idx)]


def train_router(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    models = {
        "ridge_hyper": Ridge(alpha=1.0),
        "rf_hyper": RandomForestRegressor(n_estimators=180, min_samples_leaf=4, random_state=seed, n_jobs=-1),
        "et_hyper": ExtraTreesRegressor(n_estimators=240, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "hgb_hyper": HistGradientBoostingRegressor(max_iter=140, learning_rate=0.05, random_state=seed),
    }
    x_train, y_train, _ = make_features(records, train_idx)
    x_dev, _, dev_keys = make_features(records, dev_idx)
    x_test, _, test_keys = make_features(records, test_idx)
    dev_eval = []
    test_routes = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        dev_route = route_from_predictions(dev_keys, model.predict(x_dev))
        dev_summary = fixed_policy(records, np.asarray(sorted(set(i for i, _ in dev_keys))), name, dev_route)
        dev_eval.append(dev_summary)
        test_routes[name] = route_from_predictions(test_keys, model.predict(x_test))
    best_name = max(dev_eval, key=lambda r: r["utility"])["policy"]
    best_dev = max(dev_eval, key=lambda r: r["utility"])
    out = fixed_policy(records, np.asarray(sorted(set(i for i, _ in test_keys))), best_name, test_routes[best_name])
    out["policy"] = f"best_legal_router({best_name})"
    out["dev_utility"] = best_dev["utility"]
    return out


def summary_gate_features(rec: dict[str, Any]) -> dict[str, float]:
    summary = rec["views"]["summary"]
    top_title = summary["ranked_titles"][0] if summary["ranked_titles"] else ""
    return {
        "margin": summary["score_margin"],
        "neg_entropy": -summary["score_entropy"],
        "top_overlap": title_overlap_score(rec["question"], top_title),
        "context_count": float(rec["n_context"]),
        "pool_count": float(rec["n_local_pool"]),
    }


def train_threshold_gate(records: list[dict[str, Any]], train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    features = ["margin", "neg_entropy", "top_overlap", "pool_count"]
    rich_views = ["provided_context", "hyperlink_1hop", "hyperlink_2hop", "full_local_pool"]
    train_vals = {name: np.asarray([summary_gate_features(records[int(i)])[name] for i in train_idx], dtype=np.float64) for name in features}
    candidates = []
    for feature in features:
        vals = train_vals[feature]
        if len(vals) == 0:
            continue
        thresholds = sorted(set(float(x) for x in np.quantile(vals, np.linspace(0.05, 0.95, 19))))
        for threshold in thresholds:
            for direction in ["high_summary", "low_summary"]:
                for rich in rich_views:
                    route = []
                    for i in dev_idx:
                        val = summary_gate_features(records[int(i)])[feature]
                        choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
                        route.append("summary" if choose_summary else rich)
                    dev_eval = fixed_policy(records, dev_idx, "dev_gate", route)
                    candidates.append((dev_eval["utility"], feature, threshold, direction, rich))
    if not candidates:
        return fixed_policy(records, test_idx, "threshold_gate(summary)", "summary")
    dev_utility, feature, threshold, direction, rich = max(candidates, key=lambda x: x[0])
    route = []
    for i in test_idx:
        val = summary_gate_features(records[int(i)])[feature]
        choose_summary = val >= threshold if direction == "high_summary" else val <= threshold
        route.append("summary" if choose_summary else rich)
    out = fixed_policy(records, test_idx, f"threshold_gate({feature},{rich})", route)
    out["gate_feature"] = feature
    out["gate_threshold"] = float(threshold)
    out["gate_direction"] = direction
    out["rich_view"] = rich
    out["dev_utility"] = float(dev_utility)
    return out


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    train_idx, dev_idx, test_idx = split_indices(len(records), args.seed)
    fixed_rows = [fixed_policy(records, test_idx, f"fixed_{view}", view) for view in VIEW_ORDER]
    best_fixed = max(fixed_rows, key=lambda r: r["utility"])
    gate = train_threshold_gate(records, train_idx, dev_idx, test_idx)
    learned = train_router(records, train_idx, dev_idx, test_idx, args.seed)
    best_adaptive = max([gate, learned], key=lambda r: r.get("dev_utility", -1e9))
    best_adaptive = dict(best_adaptive)
    best_adaptive["policy"] = "best_legal_adaptive"
    oracle_route = [records[int(i)]["oracle_view"] for i in test_idx]
    oracle = fixed_policy(records, test_idx, "oracle_route", oracle_route)
    denominator = max(oracle["utility"] - best_fixed["utility"], 1e-12)
    for row in fixed_rows + [gate, learned, best_adaptive, oracle]:
        row["diff_vs_best_fixed"] = row["utility"] - best_fixed["utility"]
        row["gap_closed"] = (row["utility"] - best_fixed["utility"]) / denominator

    repeat = []
    for r in range(args.repeats):
        tr, dv, te = split_indices(len(records), args.seed + 100 + r)
        fixed = [fixed_policy(records, te, f"fixed_{view}", view) for view in VIEW_ORDER]
        bf = max(fixed, key=lambda x: x["utility"])
        gate_r = train_threshold_gate(records, tr, dv, te)
        learned_r = train_router(records, tr, dv, te, args.seed + r + 1)
        adapt_r = max([gate_r, learned_r], key=lambda x: x.get("dev_utility", -1e9))
        repeat.append(adapt_r["utility"] - bf["utility"])
    repeat = np.asarray(repeat, dtype=np.float64)

    control_rows = []
    control_indices = test_idx
    for control in CONTROL_ORDER:
        utils, regrets, ndcgs, recalls, supp_f1, endpoint_f1, costs = [], [], [], [], [], [], []
        for i in control_indices:
            rec = records[int(i)]
            cr = rec["controls"][control]
            utils.append(cr["utility"])
            regrets.append(rec["oracle_utility"] - cr["utility"])
            ndcgs.append(cr["ndcg"])
            recalls.append(cr["recall"])
            supp_f1.append(cr["support_f1"])
            endpoint_f1.append(cr["endpoint_f1"])
            costs.append(cr["cost"])
        control_rows.append(
            {
                "control": control,
                "utility": float(np.mean(utils)),
                "raw_ndcg": float(np.mean(ndcgs)),
                "support_recall": float(np.mean(recalls)),
                "support_f1": float(np.mean(supp_f1)),
                "triple_endpoint_f1": float(np.mean(endpoint_f1)),
                "cost": float(np.mean(costs)),
                "regret": float(np.mean(regrets)),
            }
        )
    best_control = max(control_rows, key=lambda r: r["utility"])
    for row in control_rows:
        row["diff_vs_best_control"] = row["utility"] - best_control["utility"]

    type_rows = []
    for qtype in sorted({records[int(i)]["type"] for i in test_idx}):
        idxs = np.asarray([int(i) for i in test_idx if records[int(i)]["type"] == qtype])
        if len(idxs) == 0:
            continue
        fixed = [fixed_policy(records, idxs, f"fixed_{view}", view) for view in VIEW_ORDER]
        bf = max(fixed, key=lambda r: r["utility"])
        # Report oracle as a type-level ceiling; learned per-type routing is
        # available in the aggregate table and repeated split summary.
        oracle_type = fixed_policy(records, idxs, "oracle_route", [records[int(i)]["oracle_view"] for i in idxs])
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
        "task": "2wiki_hyperlink_corpus_stress",
        "dataset": "voidful/2wikimultihopqa + para_with_hyperlink corpus",
        "n_rows": len(records),
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "lambda_cost": args.lambda_cost,
        "topk": args.topk,
        "view_costs": VIEW_COST,
        "views": {
            "summary": "provided-context title/entity sketch only",
            "provided_context": "10 provided context paragraphs",
            "hyperlink_1hop": "query-local 1-hop paragraph expansion from the 7GB hyperlink corpus",
            "hyperlink_2hop": "capped 2-hop paragraph path expansion from selected 1-hop pages",
            "full_local_pool": "all query-local provided/1-hop/2-hop paragraphs",
        },
        "policy_rows": fixed_rows + [gate, learned, best_adaptive, oracle],
        "best_fixed": best_fixed["policy"],
        "control_rows": control_rows,
        "type_rows": type_rows,
        "oracle_view_share": {k: float(v / max(1, len(test_idx))) for k, v in oracle_share.items()},
        "repeat_summary": {
            "learned_minus_best_fixed_mean": float(np.mean(repeat)) if len(repeat) else 0.0,
            "ci95": [float(np.quantile(repeat, 0.025)), float(np.quantile(repeat, 0.975))] if len(repeat) else [0.0, 0.0],
            "positive_share": float(np.mean(repeat > 0)) if len(repeat) else 0.0,
            "repeats": int(len(repeat)),
        },
    }


def build_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    chosen = None
    for rec in records:
        if rec["views"]["hyperlink_1hop"]["utility"] > rec["views"]["provided_context"]["utility"] + 0.05:
            chosen = rec
            break
    if chosen is None:
        chosen = records[0]
    legal_route = max(["summary", "provided_context", "hyperlink_1hop", "hyperlink_2hop"], key=lambda v: chosen["views"][v]["utility"])
    legal = {
        "query_id": chosen["id"],
        "cell_id": f"2wiki_hyper_{safe_id(chosen['id'])}",
        "tier": "B1_hyperlink",
        "cost_menu": "2wiki-hyper-v1-op",
        "ranked_views": list(dict.fromkeys([legal_route, "provided_context", "hyperlink_1hop", "hyperlink_2hop"])),
        "route": legal_route,
    }
    illegal = {
        "query_id": chosen["id"],
        "cell_id": f"2wiki_hyper_{safe_id(chosen['id'])}",
        "tier": "B1_hyperlink",
        "cost_menu": "2wiki-hyper-v1-op",
        "route": "hyperlink_2hop",
        "support_titles": chosen["support_titles"],
        "answer": chosen["answer"],
    }
    return {
        "visible_fields": {
            "query_id": chosen["id"],
            "question": chosen["question"],
            "declared_views": VIEW_ORDER,
            "visible_titles": chosen["views"]["summary"]["ranked_titles"],
            "released_corpus_view": "title sketch plus declared hyperlink-expansion budget; support labels are hidden",
        },
        "legal_action": {
            "submitted_jsonl": legal,
            "charged_cost": VIEW_COST[legal_route],
            "scored_utility": chosen["views"][legal_route]["utility"],
        },
        "hidden_evaluator_fields": {
            "support_titles": chosen["support_titles"],
            "evidence_triples": chosen["evidences"],
            "answer": chosen["answer"],
        },
        "illegal_variant": {
            "submitted_jsonl": illegal,
            "rejection_reason": "support titles and answers are evaluator-only fields, even when hyperlink expansion is legal.",
        },
        "scope_note": "This is a hyperlink-corpus evidence-acquisition trace, not a generated-answer RAG transcript.",
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], profile: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"2wiki_hyperlink_corpus_stress_{args.max_rows}"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    payload = dict(result)
    payload["corpus_profile"] = profile
    payload["protocol_trace"] = build_trace(records)
    payload["examples"] = [
        {
            "id": rec["id"],
            "question": rec["question"],
            "support_titles": rec["support_titles"],
            "oracle_view": rec["oracle_view"],
            "n_1hop": rec["n_1hop"],
            "n_2hop": rec["n_2hop"],
            "view_ndcg": {v: round(rec["views"][v]["ndcg"], 4) for v in VIEW_ORDER},
            "top_titles": {v: rec["views"][v]["ranked_titles"] for v in VIEW_ORDER},
        }
        for rec in records[: args.example_rows]
    ]
    payload["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    json.dump(payload, json_path.open("w", encoding="utf-8"), indent=2)

    policy_cols = ["policy", "utility", "raw_ndcg", "recall", "cost", "regret", "gap_closed", "diff_vs_best_fixed", "view_share"]
    control_cols = ["control", "utility", "raw_ndcg", "support_recall", "support_f1", "triple_endpoint_f1", "cost", "regret", "diff_vs_best_control"]
    type_cols = ["type", "n", "best_fixed", "fixed_utility", "oracle_utility", "oracle_gap"]
    md = [
        "# 2Wiki Hyperlink-Corpus Stress Audit",
        "",
        "Narrow Protocol B stress audit using the 7GB `para_with_hyperlink.jsonl` corpus. The task is evidence acquisition only: a method sees the question and provided-context title sketch, then may buy provided context, query-local 1-hop hyperlink expansion, capped 2-hop expansion, or the full local paragraph pool. Supporting facts, evidence triples, and answers remain evaluator-held.",
        "",
        f"- Rows: {result['n_rows']} ({result['split_sizes']})",
        f"- Expansion caps: 1-hop={args.max_1hop}, 2-hop={args.max_2hop}",
        f"- Top-k: {result['topk']}; lambda: {result['lambda_cost']}",
        f"- Oracle view share: {result['oracle_view_share']}",
        "",
        "## Corpus expansion/profile",
        "",
        "```json",
        json.dumps(profile, indent=2),
        "```",
        "",
        "## Fixed and adaptive policies",
        "",
        markdown_table(result["policy_rows"], policy_cols),
        "",
        "## Anti-token and path controls",
        "",
        markdown_table(result["control_rows"], control_cols),
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
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--hyperlink-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-1hop", type=int, default=24)
    parser.add_argument("--max-2hop", type=int, default=24)
    parser.add_argument("--evidence-per-context", type=int, default=6)
    parser.add_argument("--random-evidence-per-context", type=int, default=6)
    parser.add_argument("--example-rows", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("HF_HOME", str(args.hf_cache.parent))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_cache))

    rows = load_2wiki_rows(args.max_rows, args.hf_cache)
    profile, docs, row_specs = build_local_expansion(rows, args)
    records = score_records(rows, docs, row_specs, args)
    result = summarize(records, args)
    json_path, md_path = write_outputs(result, records, profile, args)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "n_rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
