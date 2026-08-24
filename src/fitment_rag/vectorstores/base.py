"""Vector store interface.

Every backend implements the same three operations so the Phase 2 VDB comparison
is a config change, not a code change. Backends also report build time and
on-disk size, which are half the story when comparing FAISS vs Chroma vs Qdrant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class VectorStore(ABC):
    name: str = "base"

    def __init__(self, dim: int, params: dict | None = None):
        self.dim = dim
        self.params = params or {}
        self.build_seconds: float = 0.0

    @abstractmethod
    def build(self, vectors: np.ndarray, ids: list[str]) -> None: ...

    @abstractmethod
    def search(self, queries: np.ndarray, top_k: int) -> tuple[list[list[str]], list[list[float]]]:
        """Returns (ids per query, scores per query); higher score = more similar."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    def size_bytes(self, path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
