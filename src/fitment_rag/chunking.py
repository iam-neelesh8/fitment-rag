"""Document -> chunk strategies.

Chunk ids are `{doc_id}::{ordinal}` so a retrieval hit can always be traced back
to its source document, which is what the recall metric is scored against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import DATA_DIR, ChunkConfig

CHUNK_DIR = DATA_DIR / "chunks"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, max(1, len(text)), step) if text[i : i + size].strip()]


def _sentence(text: str, size: int, overlap: int) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for s in sents:
        if length + len(s) > size and buf:
            chunks.append(" ".join(buf))
            # carry the tail of the previous chunk forward as overlap
            keep: list[str] = []
            kept = 0
            for prev in reversed(buf):
                if kept + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                kept += len(prev)
            buf, length = keep, kept
        buf.append(s)
        length += len(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [text]


def chunk_documents(docs: list[dict[str, Any]], cfg: ChunkConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        text = doc["text"]
        if cfg.strategy == "whole_doc":
            pieces = [text]
        elif cfg.strategy == "fixed":
            pieces = _fixed(text, cfg.chunk_size, cfg.chunk_overlap)
        elif cfg.strategy == "sentence":
            pieces = _sentence(text, cfg.chunk_size, cfg.chunk_overlap)
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ValueError(f"unknown chunking strategy: {cfg.strategy}")

        for i, piece in enumerate(pieces):
            out.append(
                {
                    "chunk_id": f"{doc['doc_id']}::{i}",
                    "doc_id": doc["doc_id"],
                    "ordinal": i,
                    "text": piece,
                    "title": doc.get("title", ""),
                    "store": doc.get("store", ""),
                }
            )
    return out


def load_or_build_chunks(
    docs: list[dict[str, Any]], cfg: ChunkConfig, corpus_id: str
) -> list[dict[str, Any]]:
    path = CHUNK_DIR / f"{corpus_id}.jsonl"
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    chunks = chunk_documents(docs, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks
