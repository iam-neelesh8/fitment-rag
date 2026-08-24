from __future__ import annotations

from .base import VectorStore
from .faiss_store import FaissStore


def build_store(backend: str, dim: int, params: dict | None = None) -> VectorStore:
    """Phase 1 ships exact search only.

    `faiss_flat` is exhaustive, so its recall is the ceiling every approximate
    backend would be measured against. Adding HNSW / IVF / Chroma / Qdrant is
    Phase 2a: implement the VectorStore interface and register it here.
    See context/02-plan.md.
    """
    if backend == "faiss_flat":
        return FaissStore(dim, params or {}, variant="flat")
    raise ValueError(
        f"unknown vectorstore backend: {backend!r} (Phase 1 supports 'faiss_flat')"
    )
