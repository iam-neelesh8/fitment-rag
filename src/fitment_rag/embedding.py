"""Sentence-Transformers embedding wrapper with an on-disk vector cache.

The cache is keyed by index_id, so swapping the embedding model in a config
forces a re-embed while re-running the same config is instant.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import DATA_DIR, EmbeddingConfig

EMB_DIR = DATA_DIR / "embeddings"


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model = SentenceTransformer(cfg.model, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], *, is_query: bool, show_progress: bool = False) -> np.ndarray:
        prefix = self.cfg.query_prefix if is_query else self.cfg.doc_prefix
        payload = [prefix + t for t in texts] if prefix else texts
        vecs = self.model.encode(
            payload,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_corpus(self, texts: list[str], index_id: str) -> tuple[np.ndarray, float]:
        """Returns (vectors, seconds_spent). Zero seconds means it came from cache."""
        path = EMB_DIR / f"{index_id}.npy"
        if path.exists():
            return np.load(path), 0.0

        start = time.perf_counter()
        vecs = self.encode(texts, is_query=False, show_progress=True)
        elapsed = time.perf_counter() - start

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs, elapsed
