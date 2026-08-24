"""Retrievers: dense, BM25, and reciprocal-rank-fusion hybrid.

BM25 is the honesty check in this benchmark. If a 400MB embedding model can't
beat a keyword baseline on your corpus, that's the finding -- report it.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from .config import RetrievalConfig
from .embedding import Embedder
from .vectorstores.base import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        chunks: list[dict[str, Any]],
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ):
        self.cfg = cfg
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.embedder = embedder
        self.store = store
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._reranker = None

        if cfg.mode in ("bm25", "hybrid"):
            self._build_bm25()
        if cfg.reranker:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(cfg.reranker)

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_ids = [c["chunk_id"] for c in self.chunks]
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _dense(self, queries: list[str], k: int):
        vecs = self.embedder.encode(queries, is_query=True)
        return self.store.search(vecs, k)

    def _bm25_search(self, queries: list[str], k: int):
        ids_out, scores_out = [], []
        for q in queries:
            scores = self._bm25.get_scores(tokenize(q))
            top = np.argsort(scores)[::-1][:k]
            ids_out.append([self._bm25_ids[i] for i in top])
            scores_out.append([float(scores[i]) for i in top])
        return ids_out, scores_out

    @staticmethod
    def _rrf(rankings: list[list[str]], weights: list[float], k: int, c: int = 60):
        fused: dict[str, float] = {}
        for ranking, w in zip(rankings, weights):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + w / (c + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [cid for cid, _ in ordered], [s for _, s in ordered]

    def retrieve(self, queries: list[str]) -> tuple[list[list[str]], list[list[float]], float]:
        cfg = self.cfg
        depth = max(cfg.top_k, cfg.candidate_k if (cfg.reranker or cfg.mode == "hybrid") else cfg.top_k)
        start = time.perf_counter()

        if cfg.mode == "dense":
            ids, scores = self._dense(queries, depth)
        elif cfg.mode == "bm25":
            ids, scores = self._bm25_search(queries, depth)
        else:
            d_ids, _ = self._dense(queries, depth)
            b_ids, _ = self._bm25_search(queries, depth)
            fused = [
                self._rrf([d, b], [cfg.hybrid_alpha, 1.0 - cfg.hybrid_alpha], depth)
                for d, b in zip(d_ids, b_ids)
            ]
            ids = [f[0] for f in fused]
            scores = [f[1] for f in fused]

        if self._reranker is not None:
            ids, scores = self._rerank(queries, ids)

        ids = [row[: cfg.top_k] for row in ids]
        scores = [row[: cfg.top_k] for row in scores]
        elapsed = time.perf_counter() - start
        return ids, scores, elapsed

    def _rerank(self, queries: list[str], ids: list[list[str]]):
        out_ids, out_scores = [], []
        for q, row in zip(queries, ids):
            pairs = [(q, self.by_id[cid]["text"]) for cid in row]
            if not pairs:
                out_ids.append([])
                out_scores.append([])
                continue
            scores = self._reranker.predict(pairs)
            order = np.argsort(scores)[::-1]
            out_ids.append([row[i] for i in order])
            out_scores.append([float(scores[i]) for i in order])
        return out_ids, out_scores

    def contexts(self, chunk_ids: list[str]) -> list[str]:
        return [self.by_id[cid]["text"] for cid in chunk_ids if cid in self.by_id]
