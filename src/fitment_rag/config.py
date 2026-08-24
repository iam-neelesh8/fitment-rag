"""Typed, YAML-backed run configuration.

Every experiment in this benchmark is fully described by one config file. The
config is hashed into the run id, so a run's directory name is a fingerprint of
the settings that produced it -- that is what makes the leaderboard honest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


class DataConfig(BaseModel):
    source: Literal["amazon_automotive"] = "amazon_automotive"
    hf_repo: str = "McAuley-Lab/Amazon-Reviews-2023"
    # The raw JSONL, streamed directly. `datasets` 4.x dropped script loaders,
    # which this dataset still ships, so load_dataset() no longer works.
    hf_file: str = "raw/meta_categories/meta_Automotive.jsonl"
    # Corpus size ladder: 1k (smoke) -> 10k -> 50k -> 200k.
    n_docs: int = 1000
    # Take every Nth usable row. Spreads the sample over a wider slice of the
    # file at bounded cost; set to 1 for large corpora to avoid over-streaming.
    stride: int = 4
    timeout_s: int = 120
    seed: int = 17


class ChunkConfig(BaseModel):
    strategy: Literal["whole_doc", "fixed", "sentence"] = "fixed"
    chunk_size: int = 512  # characters
    chunk_overlap: int = 64


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
    device: str | None = None  # None -> auto (cuda if available, else cpu)
    query_prefix: str = ""     # e.g. "query: " for e5 models
    doc_prefix: str = ""       # e.g. "passage: " for e5 models


class VectorStoreConfig(BaseModel):
    # Phase 1 only needs exact search. Approximate backends (HNSW, IVF) and
    # alternative stores are a Phase 2 concern -- see context/02-plan.md.
    backend: Literal["faiss_flat"] = "faiss_flat"
    params: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = 5
    # What top_k counts. Metrics are scored per DOCUMENT, so counting chunks
    # here would hand configs with fewer chunks-per-document more distinct
    # candidates for free -- a confound that falls entirely on the chunking
    # comparison. "chunk" reproduces the older, biased behaviour.
    unit: Literal["document", "chunk"] = "document"
    candidate_k: int = 50       # search depth before dedupe / fusion / rerank
    hybrid_alpha: float = 0.5   # 1.0 = pure dense, 0.0 = pure bm25
    reranker: str | None = None  # cross-encoder model id, or None


class EvalConfig(BaseModel):
    eval_set: str = "evalsets/amazon_automotive_1k.jsonl"
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    limit: int | None = None  # cap number of eval queries (debugging)


class RunConfig(BaseModel):
    name: str = "unnamed"
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def fingerprint(self, *, exclude: tuple[str, ...] = ("name",)) -> str:
        payload = self.model_dump(exclude=set(exclude))
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.fingerprint()}"

    @property
    def corpus_id(self) -> str:
        """Identifies a (data, chunking) pair -- chunk files are cached by this."""
        payload = {"data": self.data.model_dump(), "chunking": self.chunking.model_dump()}
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def index_id(self) -> str:
        """Identifies a (corpus, embedding, vectorstore) triple -- indexes cache by this.

        Deliberately excludes retrieval and eval settings, so switching between
        dense / bm25 / hybrid reuses the same embeddings and index.
        """
        payload = {
            "corpus": self.corpus_id,
            "embedding": self.embedding.model_dump(),
            "vectorstore": self.vectorstore.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]
