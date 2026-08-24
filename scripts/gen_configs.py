"""Generate every sweep config from one base + one varied knob.

Phase 1 only: embedders, chunking strategies, retrieval modes. 1,000 documents,
exact search, no generation. Sized for a 15W CPU laptop (Core Ultra 5, 16GB, no
dGPU) so all 14 configs can be swept in an afternoon.

Phases 2 and 3 are described in context/02-plan.md but are not implemented here.

    python scripts/gen_configs.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "data": {"source": "amazon_automotive", "n_docs": 10000, "stride": 4, "seed": 17},
    "chunking": {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 64},
    "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "batch_size": 32},
    "vectorstore": {"backend": "faiss_flat"},
    "retrieval": {"mode": "dense", "top_k": 5, "candidate_k": 50},
    "eval": {"eval_set": "evalsets/amazon_automotive_10k.jsonl", "ks": [1, 3, 5, 10]},
}

# name, HF model id, params, query prefix, doc prefix
EMBEDDERS = [
    ("minilm-l6", "sentence-transformers/all-MiniLM-L6-v2", "22M", "", ""),
    ("bge-small", "BAAI/bge-small-en-v1.5", "33M",
     "Represent this sentence for searching relevant passages: ", ""),
    ("e5-small", "intfloat/e5-small-v2", "33M", "query: ", "passage: "),
    ("gte-small", "thenlper/gte-small", "33M", "", ""),
    ("mpnet-base", "sentence-transformers/all-mpnet-base-v2", "110M", "", ""),  # slow ceiling
]

CHUNKINGS = [
    ("whole", {"strategy": "whole_doc", "chunk_overlap": 0}),
    ("fixed256", {"strategy": "fixed", "chunk_size": 256, "chunk_overlap": 32}),
    ("fixed512", {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 64}),
    ("fixed1024", {"strategy": "fixed", "chunk_size": 1024, "chunk_overlap": 128}),
    ("sent512", {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}),
]

RETRIEVERS = [
    ("bm25", {"mode": "bm25"}),
    ("dense", {"mode": "dense"}),
    ("hybrid", {"mode": "hybrid", "hybrid_alpha": 0.5}),
    ("hybrid-rerank", {"mode": "hybrid", "hybrid_alpha": 0.5,
                       "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"}),
]


def write(name: str, path: Path, **sections) -> None:
    cfg = copy.deepcopy(BASE)
    cfg["name"] = name
    for section, values in sections.items():
        cfg[section] = {**cfg.get(section, {}), **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    n = 0
    for tag, model, _params, qp, dp in EMBEDDERS:
        write(f"emb-{tag}", ROOT / f"configs/phase1/emb/{tag}.yaml",
              embedding={"model": model, "query_prefix": qp, "doc_prefix": dp})
        n += 1

    for tag, chunking in CHUNKINGS:
        write(f"chunk-{tag}", ROOT / f"configs/phase1/chunk/{tag}.yaml", chunking=chunking)
        n += 1

    for tag, retrieval in RETRIEVERS:
        write(f"ret-{tag}", ROOT / f"configs/phase1/retrieval/{tag}.yaml", retrieval=retrieval)
        n += 1


    print(f"wrote {n} Phase 1 configs under configs/phase1/")


if __name__ == "__main__":
    main()
