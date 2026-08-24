from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .base import VectorStore


class FaissStore(VectorStore):
    """Exhaustive FAISS backend -- exact nearest neighbours, no approximation.

    Vectors are L2-normalized upstream, so inner product == cosine similarity,
    and IndexFlatIP is therefore exact cosine search. Being exact is the point:
    it makes this the recall ceiling for Phase 2's approximate backends.
    """

    def __init__(self, dim: int, params: dict | None = None, variant: str = "flat"):
        super().__init__(dim, params)
        self.variant = variant
        self.name = f"faiss_{variant}"
        self.index = None
        self.ids: list[str] = []

    def _new_index(self, n_vectors: int):
        import faiss

        return faiss.IndexFlatIP(self.dim)

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        start = time.perf_counter()
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.index = self._new_index(len(ids))
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.ids = list(ids)
        self.build_seconds = time.perf_counter() - start

    def search(self, queries: np.ndarray, top_k: int):
        queries = np.ascontiguousarray(queries, dtype="float32")
        scores, idx = self.index.search(queries, top_k)
        out_ids = [[self.ids[i] for i in row if i != -1] for row in idx]
        out_scores = [[float(s) for s, i in zip(srow, irow) if i != -1]
                      for srow, irow in zip(scores, idx)]
        return out_ids, out_scores

    def save(self, path: Path) -> None:
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    def load(self, path: Path) -> None:
        import faiss

        self.index = faiss.read_index(str(path / "index.faiss"))
        self.ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
