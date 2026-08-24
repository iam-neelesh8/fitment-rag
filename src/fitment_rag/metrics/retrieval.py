"""Retrieval metrics scored at the document level.

Chunk granularity varies between configs, so scoring at chunk level would make
chunk-size comparisons meaningless. Everything is reduced to doc_ids first.
"""

from __future__ import annotations

import math
from typing import Iterable


def _to_doc(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def dedupe_docs(chunk_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        did = _to_doc(cid)
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def hit_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved_docs[:k]) & relevant else 0.0


def recall_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved_docs[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved_docs[:k]) & relevant) / k


def mrr(retrieved_docs: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(retrieved_docs):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2) for i, did in enumerate(retrieved_docs[:k]) if did in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def score_query(retrieved_chunk_ids: list[str], relevant_docs: set[str], ks: list[int]) -> dict:
    docs = dedupe_docs(retrieved_chunk_ids)
    out: dict[str, float] = {"mrr": mrr(docs, relevant_docs)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(docs, relevant_docs, k)
        out[f"recall@{k}"] = recall_at_k(docs, relevant_docs, k)
        out[f"precision@{k}"] = precision_at_k(docs, relevant_docs, k)
        out[f"ndcg@{k}"] = ndcg_at_k(docs, relevant_docs, k)
    return out


def aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
