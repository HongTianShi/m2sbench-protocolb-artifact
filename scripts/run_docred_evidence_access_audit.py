from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TEXT_ROOT = WORKSPACE / "\u6587\u672c\u6570\u636e\u96c6"
REPORTS = ROOT / "reports"


@dataclass(slots=True)
class DocredEvidenceConfig:
    max_queries: int = 30_000
    topk: int = 5
    pq_dim: int = 64
    lambda_cost: float = 0.08
    cost_summary: float = 0.0
    cost_pq: float = 0.20
    cost_full: float = 0.58
    random_state: int = 20260525
    batch_size: int = 128
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def _find_one(pattern: str, must_contain: str | None = None) -> Path:
    candidates = [p for p in TEXT_ROOT.rglob(pattern) if p.is_file()]
    if must_contain:
        candidates = [p for p in candidates if must_contain.lower() in str(p).lower()]
    if not candidates:
        raise FileNotFoundError(f"Could not find {pattern} under {TEXT_ROOT}")
    return sorted(candidates, key=lambda p: (-p.stat().st_size, len(str(p))))[0]


def _manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return path.name


def _markdown_table(frame: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _load_docred() -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, str]]]:
    train_path = _find_one("train_annotated.json", "DocRED")
    dev_path = _find_one("dev.json", "DocRED")
    rel_path = _find_one("rel_info.json", "DocRED")
    rel_info = json.load(rel_path.open("r", encoding="utf-8"))
    docs: list[dict[str, Any]] = []
    for split, path in [("train", train_path), ("dev", dev_path)]:
        data = json.load(path.open("r", encoding="utf-8"))
        for local_id, item in enumerate(data):
            item = dict(item)
            item["_split"] = split
            item["_doc_id"] = f"{split}:{local_id}"
            docs.append(item)
    manifest = [
        {"split": "train", "source": _manifest_path(train_path)},
        {"split": "dev", "source": _manifest_path(dev_path)},
        {"split": "relations", "source": _manifest_path(rel_path)},
    ]
    return docs, rel_info, manifest


def _entity_name(vertex_set: list[list[dict[str, Any]]], idx: int) -> str:
    if idx < 0 or idx >= len(vertex_set) or not vertex_set[idx]:
        return f"entity_{idx}"
    mentions = vertex_set[idx]
    names = [str(m.get("name", "")).strip() for m in mentions if str(m.get("name", "")).strip()]
    return max(names, key=len) if names else f"entity_{idx}"


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _entity_score(sentence: str, head: str, tail: str, rel_name: str) -> float:
    lower = sentence.lower()
    head_hit = 1.0 if head.lower() in lower else 0.0
    tail_hit = 1.0 if tail.lower() in lower else 0.0
    rel_tokens = _tokens(rel_name)
    sent_tokens = _tokens(sentence)
    rel_overlap = len(rel_tokens & sent_tokens) / max(1, len(rel_tokens))
    return 2.0 * (head_hit + tail_hit) + 0.25 * rel_overlap


def build_queries(config: DocredEvidenceConfig) -> tuple[list[dict[str, Any]], list[str], list[str], list[dict[str, str]]]:
    docs, rel_info, manifest = _load_docred()
    queries: list[dict[str, Any]] = []
    sentence_texts: list[str] = []
    sentence_key_to_id: dict[tuple[str, int], int] = {}

    for doc in docs:
        sents = [" ".join(sent) for sent in doc.get("sents", [])]
        if len(sents) < 2:
            continue
        sent_ids: list[int] = []
        for sid, text in enumerate(sents):
            key = (doc["_doc_id"], sid)
            sentence_key_to_id[key] = len(sentence_texts)
            sentence_texts.append(text)
            sent_ids.append(sentence_key_to_id[key])
        vertex_set = doc.get("vertexSet") or []
        for label in doc.get("labels") or []:
            evidence = sorted({int(e) for e in label.get("evidence", []) if isinstance(e, int) or str(e).isdigit()})
            evidence = [e for e in evidence if 0 <= e < len(sents)]
            if not evidence:
                continue
            h = int(label.get("h", -1))
            t = int(label.get("t", -1))
            rel_id = str(label.get("r", "UNK"))
            head = _entity_name(vertex_set, h)
            tail = _entity_name(vertex_set, t)
            rel_name = rel_info.get(rel_id, rel_id)
            query_text = f"{doc.get('title', '')}: evidence for relation {rel_name} between {head} and {tail}"
            summary_scores = [_entity_score(text, head, tail, rel_name) - 0.001 * sid for sid, text in enumerate(sents)]
            queries.append(
                {
                    "query_id": f"{doc['_doc_id']}:{len(queries)}",
                    "doc_id": doc["_doc_id"],
                    "split": doc["_split"],
                    "title": doc.get("title", ""),
                    "query_text": query_text,
                    "candidate_sentence_ids": sent_ids,
                    "relevant_local": evidence,
                    "relevant_sentence_ids": [sent_ids[e] for e in evidence],
                    "summary_scores": summary_scores,
                    "candidate_count": len(sent_ids),
                    "relation": rel_id,
                }
            )

    rng = random.Random(config.random_state)
    rng.shuffle(queries)
    if config.max_queries and len(queries) > config.max_queries:
        queries = queries[: config.max_queries]
    query_texts = [q["query_text"] for q in queries]
    return queries, query_texts, sentence_texts, manifest


def ndcg_at_k(order: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(order[:k]):
        if int(item) in relevant:
            dcg += 1.0 / math.log2(rank + 2)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def hit_at_k(order: np.ndarray, relevant: set[int], k: int) -> float:
    return float(any(int(item) in relevant for item in order[:k]))


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def encode_texts(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def evaluate(config: DocredEvidenceConfig) -> dict[str, Any]:
    start = time.time()
    queries, query_texts, sentence_texts, manifest = build_queries(config)
    query_emb = encode_texts(config.model_name, query_texts, config.batch_size)
    sent_emb = encode_texts(config.model_name, sentence_texts, config.batch_size)
    pq_dim = min(config.pq_dim, sent_emb.shape[1])
    query_pq = _normalize(query_emb[:, :pq_dim].copy())
    sent_pq = _normalize(sent_emb[:, :pq_dim].copy())

    view_names = ["summary", "pq_dense", "full_dense"]
    costs = {"summary": config.cost_summary, "pq_dense": config.cost_pq, "full_dense": config.cost_full}
    rows: dict[str, dict[str, list[float]]] = {
        name: {"ndcg": [], "utility": [], "regret": [], "hit": [], "cost": []} for name in view_names
    }
    best_counts = {name: 0 for name in view_names}
    pq_dominates_full = 0
    non_full_best = 0
    timings = {name: [] for name in view_names}
    per_query: list[dict[str, float]] = []

    for qi, q in enumerate(queries):
        cand = np.asarray(q["candidate_sentence_ids"], dtype=np.int64)
        relevant_local = {q["candidate_sentence_ids"].index(sid) for sid in q["relevant_sentence_ids"]}

        t0 = time.perf_counter()
        summary_scores = np.asarray(q["summary_scores"], dtype=np.float32)
        summary_order = np.argsort(-summary_scores, kind="mergesort")
        timings["summary"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pq_scores = sent_pq[cand] @ query_pq[qi]
        pq_order = np.argsort(-pq_scores, kind="mergesort")
        timings["pq_dense"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        full_scores = sent_emb[cand] @ query_emb[qi]
        full_order = np.argsort(-full_scores, kind="mergesort")
        timings["full_dense"].append(time.perf_counter() - t0)

        orders = {"summary": summary_order, "pq_dense": pq_order, "full_dense": full_order}
        utilities: dict[str, float] = {}
        ndcgs: dict[str, float] = {}
        hits: dict[str, float] = {}
        for name in view_names:
            ndcg = ndcg_at_k(orders[name], relevant_local, config.topk)
            hit = hit_at_k(orders[name], relevant_local, config.topk)
            utility = ndcg - config.lambda_cost * costs[name]
            ndcgs[name] = ndcg
            hits[name] = hit
            utilities[name] = utility
        best_utility = max(utilities.values())
        best_name = max(view_names, key=lambda n: (utilities[n], -costs[n]))
        best_counts[best_name] += 1
        if best_name != "full_dense":
            non_full_best += 1
        if ndcgs["pq_dense"] >= ndcgs["full_dense"]:
            pq_dominates_full += 1
        for name in view_names:
            rows[name]["ndcg"].append(ndcgs[name])
            rows[name]["utility"].append(utilities[name])
            rows[name]["regret"].append(best_utility - utilities[name])
            rows[name]["hit"].append(hits[name])
            rows[name]["cost"].append(costs[name])
        per_query.append(
            {
                "summary_ndcg": ndcgs["summary"],
                "pq_ndcg": ndcgs["pq_dense"],
                "full_ndcg": ndcgs["full_dense"],
                "summary_utility": utilities["summary"],
                "pq_utility": utilities["pq_dense"],
                "full_utility": utilities["full_dense"],
            }
        )

    def summarize(mask: np.ndarray, scope: str) -> dict[str, Any]:
        if not mask.any():
            return {
                "scope": scope,
                "queries": 0,
                "summary_ndcg": None,
                "pq_ndcg": None,
                "full_ndcg": None,
                "best": "",
                "non_full_best": None,
            }
        view_ndcg = {
            "summary": np.asarray([q["summary_ndcg"] for q in per_query], dtype=np.float64)[mask],
            "pq_dense": np.asarray([q["pq_ndcg"] for q in per_query], dtype=np.float64)[mask],
            "full_dense": np.asarray([q["full_ndcg"] for q in per_query], dtype=np.float64)[mask],
        }
        utilities_arr = {
            "summary": np.asarray([q["summary_utility"] for q in per_query], dtype=np.float64)[mask],
            "pq_dense": np.asarray([q["pq_utility"] for q in per_query], dtype=np.float64)[mask],
            "full_dense": np.asarray([q["full_utility"] for q in per_query], dtype=np.float64)[mask],
        }
        stack = np.vstack([utilities_arr[name] for name in view_names])
        best_idx = np.argmax(stack, axis=0)
        best_name = view_names[int(np.argmax([utilities_arr[name].mean() for name in view_names]))]
        return {
            "scope": scope,
            "queries": int(mask.sum()),
            "summary_ndcg": float(view_ndcg["summary"].mean()),
            "pq_ndcg": float(view_ndcg["pq_dense"].mean()),
            "full_ndcg": float(view_ndcg["full_dense"].mean()),
            "best": best_name,
            "non_full_best": float(np.mean(best_idx != 2)),
        }

    view_rows: list[dict[str, Any]] = []
    for name in view_names:
        view_rows.append(
            {
                "view": name,
                "NDCG@5": float(np.mean(rows[name]["ndcg"])),
                "utility": float(np.mean(rows[name]["utility"])),
                "regret": float(np.mean(rows[name]["regret"])),
                "hit@5": float(np.mean(rows[name]["hit"])),
                "cost": costs[name],
                "mean_us": float(np.mean(timings[name]) * 1e6),
            }
        )

    summary_ndcg_array = np.asarray([q["summary_ndcg"] for q in per_query], dtype=np.float64)
    scope_rows = [
        summarize(np.ones(len(per_query), dtype=bool), "all"),
        summarize(summary_ndcg_array < 0.999, "summary_imperfect"),
        summarize(summary_ndcg_array < 0.500, "summary_hard"),
    ]

    s_ndcg = view_rows[0]["NDCG@5"]
    p_ndcg = view_rows[1]["NDCG@5"]
    f_ndcg = view_rows[2]["NDCG@5"]
    lambda_s_p = (p_ndcg - s_ndcg) / max(1e-12, config.cost_pq - config.cost_summary)
    lambda_p_f = (f_ndcg - p_ndcg) / max(1e-12, config.cost_full - config.cost_pq)
    cell_sizes = [q["candidate_count"] for q in queries]
    result = {
        "audit": "docred_evidence_access",
        "status": "optional_repository_record",
        "purpose": "Exact-qrel evidence-sentence access audit for a RAG-style DocRED setting.",
        "config": asdict(config),
        "sources": manifest,
        "model": config.model_name,
        "queries": len(queries),
        "unique_candidate_sentences": len(sentence_texts),
        "candidate_sentence_median": float(np.median(cell_sizes)),
        "candidate_sentence_p90": float(np.percentile(cell_sizes, 90)),
        "rows": view_rows,
        "scope_rows": scope_rows,
        "best_view_share": {name: best_counts[name] / len(queries) for name in view_names},
        "non_full_best_share": non_full_best / len(queries),
        "pq_dominates_full_precost_share": pq_dominates_full / len(queries),
        "lambda_star": {"summary_to_pq": float(lambda_s_p), "pq_to_full": float(lambda_p_f)},
        "elapsed_seconds": time.time() - start,
        "reading": (
            "DocRED evidence retrieval supplies exact evidence-sentence qrels. Cheap entity/co-mention evidence is often enough, "
            "while dense sentence evidence is a targeted escalation view for the residual hard subset."
        ),
    }
    return result


def write_report(result: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / "docred_evidence_access_audit.json"
    md_path = REPORTS / "docred_evidence_access_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    table = _markdown_table(result["rows"], ["view", "NDCG@5", "utility", "regret", "hit@5", "cost", "mean_us"])
    scope_table = _markdown_table(
        result["scope_rows"], ["scope", "queries", "summary_ndcg", "pq_ndcg", "full_ndcg", "best", "non_full_best"]
    )
    best_share = result["best_view_share"]
    md = f"""# DocRED Evidence Access Audit

This optional audit instantiates the Protocol B access contract as an evidence-sentence ranking task. It complements the IVF-PQ label-surrogate slice by using exact DocRED evidence annotations as qrels.

## Setup

- Dataset: DocRED train/dev documents with relation evidence annotations.
- Query: document title plus a head entity, tail entity, and relation description.
- Candidate set: sentences in the same document.
- Qrels: evaluator-held evidence sentence indices supplied by DocRED.
- Views:
  - summary: entity/co-mention lexical summary with no dense sentence evidence;
  - PQ dense: the first {result['config']['pq_dim']} dimensions of a dense sentence embedding;
  - full dense: full {result['model']} sentence evidence.
- Metric: NDCG@5 before cost and utility after the default access menu, with lambda = {result['config']['lambda_cost']} and costs summary/PQ/full = {result['config']['cost_summary']}/{result['config']['cost_pq']}/{result['config']['cost_full']}.

## Results

Scored {result['queries']:,} relation queries over {result['unique_candidate_sentences']:,} candidate sentences. Median/p90 candidate sentences per query: {result['candidate_sentence_median']:.1f}/{result['candidate_sentence_p90']:.1f}. Runtime: {result['elapsed_seconds']:.1f}s.

{table}

Subset diagnostics:

{scope_table}

Per-query access diagnostics:

- Non-full views are cost-adjusted best on {result['non_full_best_share']:.1%} of queries.
- PQ dense dominates full dense before cost on {result['pq_dominates_full_precost_share']:.1%} of queries.
- Best-view shares: summary {best_share['summary']:.1%}, PQ dense {best_share['pq_dense']:.1%}, full dense {best_share['full_dense']:.1%}.
- Aggregate break-even thresholds: summary to PQ lambda* = {result['lambda_star']['summary_to_pq']:.3f}; PQ to full lambda* = {result['lambda_star']['pq_to_full']:.3f}.

## Reading

This is a RAG-style evidence access boundary case with exact qrels. Cheap entity/co-mention evidence is often sufficient because many DocRED evidence sentences explicitly mention the queried entities. The hard subset records where this summary breaks down and dense sentence evidence becomes a targeted escalation view. The result supports the same access-routing contract without relying on label-derived pseudo-qrels.
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    defaults = DocredEvidenceConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=defaults.max_queries)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--model-name", default=defaults.model_name)
    args = parser.parse_args()
    config = DocredEvidenceConfig(max_queries=args.max_queries, batch_size=args.batch_size, model_name=args.model_name)
    result = evaluate(config)
    write_report(result)


if __name__ == "__main__":
    main()
