"""Amazon Reviews 2023, Automotive product metadata.

Research use with citation (McAuley Lab, UCSD); see DATA_LICENSES.md. Nothing is
redistributed -- the corpus is streamed at build time and cached to data/raw/.

Streamed over HTTP rather than through `datasets.load_dataset`, which no longer
works: version 4.x dropped script-based loaders and this dataset still ships
one. The source file is plain JSONL and 5.35 GB, so it is read line by line and
abandoned once enough rows are collected.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

import requests

from ..config import DATA_DIR, DataConfig

RAW_DIR = DATA_DIR / "raw"
HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_clean(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(f"{k}: {_clean(v)}" for k, v in value.items())
    return str(value).strip()


def _doc_text(row: dict[str, Any]) -> str:
    """Flatten one product row into a single retrievable document.

    Field labels give the embedder lexical anchors, so a query mentioning
    "brand" has something to match.
    """
    parts: list[str] = []

    title = _clean(row.get("title"))
    if title:
        parts.append(f"Title: {title}")

    store = _clean(row.get("store"))
    if store:
        parts.append(f"Brand/Store: {store}")

    cats = row.get("categories") or []
    if cats:
        parts.append(f"Categories: {' > '.join(_clean(c) for c in cats)}")

    price = _clean(row.get("price"))
    if price and price.lower() not in ("none", "null", "nan"):
        parts.append(f"Price: {price}")

    features = row.get("features") or []
    if features:
        parts.append("Features: " + " | ".join(_clean(f) for f in features))

    description = row.get("description") or []
    if description:
        parts.append("Description: " + " ".join(_clean(d) for d in description))

    details = row.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {"details": details}
    if isinstance(details, dict) and details:
        kv = " | ".join(f"{k}: {_clean(v)}" for k, v in details.items() if _clean(v))
        if kv:
            parts.append("Details: " + kv)

    return "\n".join(parts)


def _doc_id(row: dict[str, Any], fallback: int) -> str:
    asin = _clean(row.get("parent_asin")) or _clean(row.get("asin"))
    return asin or f"row-{fallback}"


def _to_document(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    text = _doc_text(row)
    if len(text) < 120:  # near-empty listings poison a retrieval eval
        return None
    return {
        "doc_id": _doc_id(row, index),
        "text": text,
        "title": _clean(row.get("title")),
        "store": _clean(row.get("store")),
        "categories": [_clean(c) for c in (row.get("categories") or [])],
        "average_rating": row.get("average_rating"),
        "rating_number": row.get("rating_number"),
        "price": _clean(row.get("price")),
    }


def stream_rows(cfg: DataConfig, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw rows from the Hugging Face file, in file order.

    Stops as soon as the caller stops consuming, so only the needed prefix is
    ever transferred.
    """
    url = HF_BASE.format(repo=cfg.hf_repo, path=cfg.hf_file)
    with requests.get(url, stream=True, timeout=cfg.timeout_s) as resp:
        resp.raise_for_status()
        for i, line in enumerate(resp.iter_lines(decode_unicode=True)):
            if limit is not None and i >= limit:
                return
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _sample(cfg: DataConfig) -> list[dict[str, Any]]:
    """Take every `stride`-th usable row until n_docs are collected.

    A deterministic prefix scan, not a uniform sample: drawing uniformly would
    mean streaming all 5.35 GB for even a 1k corpus. Reproducible, and nested --
    a 10k corpus is a superset of the 1k corpus at the same stride.
    """
    stride = max(1, cfg.stride)
    docs: list[dict[str, Any]] = []
    usable = 0

    for i, row in enumerate(stream_rows(cfg)):
        doc = _to_document(row, i)
        if doc is None:
            continue
        if usable % stride == 0:
            docs.append(doc)
            if len(docs) >= cfg.n_docs:
                break
        usable += 1

    return docs


def load_documents(cfg: DataConfig, *, refresh: bool = False) -> list[dict[str, Any]]:
    """Return the configured corpus, using the on-disk cache when available."""
    cache = RAW_DIR / f"{cfg.source}-n{cfg.n_docs}-s{cfg.stride}.jsonl"
    if cache.exists() and not refresh:
        with open(cache, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    docs = _sample(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


def corpus_checksum(docs: list[dict[str, Any]]) -> str:
    """Hash of (doc_id, text) pairs -- lets a reader prove they rebuilt our corpus."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x["doc_id"]):
        h.update(d["doc_id"].encode())
        h.update(b"\x00")
        h.update(d["text"].encode())
        h.update(b"\x00")
    return h.hexdigest()
