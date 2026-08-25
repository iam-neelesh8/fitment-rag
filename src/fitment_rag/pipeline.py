"""End-to-end Phase 1 run: corpus -> chunks -> embeddings -> index -> retrieve -> score.

Each stage caches to disk keyed by a fingerprint of the settings that produced
it, so re-running a config is instant and changing one knob only invalidates the
stages downstream of it.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import load_or_build_chunks
from .config import DATA_DIR, RESULTS_DIR, RunConfig
from .data.amazon import corpus_checksum, load_documents
from .embedding import Embedder
from .evalset.build import load_eval_set
from .metrics.retrieval import aggregate, score_query
from .retrieval import Retriever
from .vectorstore import FaissIndex

INDEX_DIR = DATA_DIR / "indexes"


@dataclass
class RunArtifacts:
    config: RunConfig
    metrics: dict[str, Any]
    records: list[dict[str, Any]]
    out_dir: Path


def build_index(cfg: RunConfig, verbose: bool = True):
    """Materialize corpus, chunks, embeddings, and the vector index. Returns handles."""
    t0 = time.perf_counter()
    docs = load_documents(cfg.data)
    if verbose:
        print(f"[data]   {len(docs)} documents  (corpus_id={cfg.corpus_id})")

    chunks = load_or_build_chunks(docs, cfg.chunking, cfg.corpus_id)
    if verbose:
        print(f"[chunk]  {len(chunks)} chunks  strategy={cfg.chunking.strategy}")

    embedder = Embedder(cfg.embedding)
    if verbose:
        print(f"[embed]  {cfg.embedding.model}  dim={embedder.dim}  device={embedder.device}")
    vectors, embed_seconds = embedder.embed_corpus([c["text"] for c in chunks], cfg.index_id)

    store = FaissIndex(embedder.dim)
    index_path = INDEX_DIR / cfg.index_id / store.name

    if (index_path / "index.faiss").exists():
        store.load(index_path)
        if verbose:
            print(f"[index]  loaded cached {store.name}")
    else:
        store.build(vectors, [c["chunk_id"] for c in chunks])
        store.save(index_path)
        if verbose:
            print(f"[index]  built {store.name} in {store.build_seconds:.1f}s")

    stats = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embed_dim": embedder.dim,
        "embed_device": embedder.device,
        "embed_seconds": round(embed_seconds, 2),
        "index_build_seconds": round(store.build_seconds, 2),
        "index_size_bytes": store.size_bytes(index_path),
        "corpus_checksum": corpus_checksum(docs),
        "setup_seconds": round(time.perf_counter() - t0, 2),
    }
    return docs, chunks, embedder, store, stats


def run(cfg: RunConfig, *, verbose: bool = True) -> RunArtifacts:
    docs, chunks, embedder, store, stats = build_index(cfg, verbose=verbose)

    eval_path = Path(cfg.eval.eval_set)
    if not eval_path.is_absolute():
        eval_path = Path(__file__).resolve().parents[2] / eval_path
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval set not found: {eval_path}\n"
            f"Build one with: fitment-rag build-evalset --config <your.yaml>"
        )
    queries = load_eval_set(eval_path)
    if cfg.eval.limit:
        queries = queries[: cfg.eval.limit]
    if verbose:
        print(f"[eval]   {len(queries)} queries from {eval_path.name}")

    # An eval set built from a different corpus scores badly but does not crash,
    # which looks like a model result instead of a broken experiment. Check that
    # the answers are actually reachable before spending time retrieving them.
    corpus_docs = {d["doc_id"] for d in docs}
    gold = {d for q in queries for d in q["relevant_doc_ids"]}
    reachable = len(gold & corpus_docs) / len(gold)
    if reachable < 0.99:
        raise ValueError(
            f"only {reachable:.1%} of gold documents exist in this corpus "
            f"({len(gold & corpus_docs)}/{len(gold)}). The eval set was built for a "
            f"different corpus, so hit@1 is capped at {reachable:.3f} no matter how "
            f"good retrieval is. Rebuild it with: "
            f"fitment-rag build-evalset --config <this config> --out <new path>"
        )

    retriever = Retriever(cfg.retrieval, chunks, embedder=embedder, store=store)
    questions = [q["question"] for q in queries]
    retrieved, scores, retrieval_seconds = retriever.retrieve(questions)

    records: list[dict[str, Any]] = []
    per_query_metrics: list[dict[str, float]] = []

    for q, chunk_ids, chunk_scores in zip(queries, retrieved, scores):
        relevant = set(q["relevant_doc_ids"])
        m = score_query(chunk_ids, relevant, cfg.eval.ks)
        per_query_metrics.append(m)

        records.append(
            {
                "query_id": q["query_id"],
                "question": q["question"],
                "ground_truth": q.get("ground_truth"),
                "relevant_doc_ids": q["relevant_doc_ids"],
                "retrieved_chunk_ids": chunk_ids,
                "retrieval_scores": chunk_scores,
                "contexts": retriever.contexts(chunk_ids),
                "metrics": m,
            }
        )

    metrics: dict[str, Any] = {
        "run_id": cfg.run_id,
        "name": cfg.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(queries),
        **stats,
        **{k: round(v, 4) for k, v in aggregate(per_query_metrics).items()},
        "retrieval_seconds_total": round(retrieval_seconds, 3),
        "retrieval_ms_per_query": round(1000 * retrieval_seconds / max(1, len(queries)), 2),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    out_dir = RESULTS_DIR / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(), indent=2, default=str), encoding="utf-8"
    )
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[done]   results -> {out_dir}")
    return RunArtifacts(cfg, metrics, records, out_dir)
