"""Exhaustive vector index over FAISS."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class FaissIndex:
    """Brute-force inner-product search.

    Vectors are L2-normalised upstream, so inner product is cosine similarity
    and this returns exact nearest neighbours. Exactness is the point: it fixes
    the recall ceiling that an approximate index would be measured against.
    """

    name = "faiss_flat"

    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self.ids: list[str] = []
        self.build_seconds = 0.0

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        import faiss

        start = time.perf_counter()
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.ids = list(ids)
        self.build_seconds = time.perf_counter() - start

    def search(self, queries: np.ndarray, top_k: int):
        """Return (ids, scores) per query, highest score first."""
        scores, idx = self.index.search(np.ascontiguousarray(queries, dtype="float32"), top_k)
        ids = [[self.ids[i] for i in row if i != -1] for row in idx]
        out = [[float(s) for s, i in zip(srow, irow) if i != -1]
               for srow, irow in zip(scores, idx)]
        return ids, out

    def save(self, path: Path) -> None:
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    def load(self, path: Path) -> None:
        import faiss

        self.index = faiss.read_index(str(path / "index.faiss"))
        self.ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))

    @staticmethod
    def size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
