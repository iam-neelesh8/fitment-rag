"""Offline tests: no network, no model downloads, no Ollama.

These cover the logic that is easy to get subtly wrong -- chunk id round-tripping,
metric arithmetic, config fingerprinting -- so a broken sweep fails in seconds
instead of after an hour of embedding.
"""

from __future__ import annotations

import numpy as np
import pytest

from fitment_rag.chunking import chunk_documents
from fitment_rag.config import ChunkConfig, RunConfig
from fitment_rag.metrics.retrieval import dedupe_docs, mrr, ndcg_at_k, recall_at_k, score_query
from fitment_rag.vectorstores.registry import build_store


@pytest.fixture
def docs():
    return [
        {"doc_id": f"D{i}", "title": f"Part {i}", "store": "Acme",
         "text": f"Title: Brake Pad {i}\nBrand: Acme\n" + ("detail text " * 60)}
        for i in range(5)
    ]


# --- chunking -------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["whole_doc", "fixed", "sentence"])
def test_chunk_ids_trace_back_to_documents(docs, strategy):
    chunks = chunk_documents(docs, ChunkConfig(strategy=strategy, chunk_size=200, chunk_overlap=20))
    assert chunks
    for c in chunks:
        assert c["chunk_id"].split("::")[0] == c["doc_id"]
    assert {c["doc_id"] for c in chunks} == {d["doc_id"] for d in docs}


def test_smaller_chunks_produce_more_chunks(docs):
    small = chunk_documents(docs, ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=0))
    large = chunk_documents(docs, ChunkConfig(strategy="fixed", chunk_size=1000, chunk_overlap=0))
    assert len(small) > len(large)


def test_no_empty_chunks(docs):
    for c in chunk_documents(docs, ChunkConfig(strategy="fixed", chunk_size=128, chunk_overlap=16)):
        assert c["text"].strip()


# --- retrieval metrics ----------------------------------------------------

def test_dedupe_preserves_rank_order():
    assert dedupe_docs(["D1::0", "D1::3", "D2::0", "D1::1", "D3::2"]) == ["D1", "D2", "D3"]


def test_recall_and_mrr_on_a_known_ranking():
    ranked = ["D9", "D4", "D1"]
    assert recall_at_k(ranked, {"D1"}, 3) == 1.0
    assert recall_at_k(ranked, {"D1"}, 2) == 0.0
    assert mrr(ranked, {"D4"}) == pytest.approx(0.5)
    assert mrr(ranked, {"D7"}) == 0.0


def test_ndcg_rewards_higher_ranks():
    top = ndcg_at_k(["D1", "D2", "D3"], {"D1"}, 3)
    bottom = ndcg_at_k(["D2", "D3", "D1"], {"D1"}, 3)
    assert top == 1.0 and top > bottom


def test_score_query_is_chunk_aware():
    m = score_query(["D1::0", "D1::1", "D2::0"], {"D1"}, [1, 3])
    assert m["hit@1"] == 1.0          # D1::0 and D1::1 collapse to one doc
    assert m["precision@1"] == 1.0
    assert m["mrr"] == 1.0



# --- vector store ---------------------------------------------------------

def test_faiss_flat_returns_exact_nearest_neighbour():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    store = build_store("faiss_flat", 8)
    store.build(vecs, [f"C{i}" for i in range(50)])

    ids, scores = store.search(vecs[3:4], top_k=3)
    assert ids[0][0] == "C3"                     # a vector is its own nearest neighbour
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("backend", ["pinecone", "faiss_hnsw", "chroma"])
def test_unimplemented_backends_are_rejected(backend):
    """Phase 1 is exact-search only; anything else must fail loudly, not silently."""
    with pytest.raises(ValueError):
        build_store(backend, 8)


# --- config fingerprinting ------------------------------------------------

def test_changing_a_knob_changes_the_run_id():
    a = RunConfig(name="x")
    b = RunConfig(name="x")
    b.embedding.model = "BAAI/bge-small-en-v1.5"
    assert a.run_id != b.run_id


def test_name_alone_does_not_change_the_fingerprint():
    a, b = RunConfig(name="alpha"), RunConfig(name="beta")
    assert a.fingerprint() == b.fingerprint()


def test_index_id_ignores_retrieval_but_not_chunking():
    a, b = RunConfig(), RunConfig()
    b.retrieval.mode = "bm25"
    assert a.index_id == b.index_id      # switching retrieval mode reuses the index
    b.chunking.chunk_size = 999
    assert a.index_id != b.index_id      # chunking must invalidate it
