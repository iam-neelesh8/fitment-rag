# Architecture

## Layout

```
src/fitment_rag/        the library -- notebooks import from here, nothing duplicated
  config.py             typed YAML config; hashes into run_id / corpus_id / index_id
  data/amazon.py        HTTP streaming, row -> document flattening, corpus checksum
  chunking.py           whole_doc / fixed / sentence
  embedding.py          sentence-transformers wrapper with a vector cache
  vectorstores/
    base.py             the VectorStore interface (build / search / save / load)
    faiss_store.py      flat, HNSW, IVF
    chroma_store.py     persistent local Chroma
    registry.py         backend name -> class
  retrieval.py          dense, BM25, hybrid RRF, optional cross-encoder rerank
  generation.py         Ollama client, one prompt, temperature 0
  evalset/build.py      template (deterministic) and LLM question generators
  metrics/
    retrieval.py        hit@k, recall@k, precision@k, mrr, ndcg@k
    answer.py           exact match, contains, token F1, abstain rate
  pipeline.py           the full run, with per-stage caching
  report.py             leaderboard, shared by CLI and notebooks
  cli.py                doctor / build-evalset / run / sweep / compare

notebooks/              the guided walkthrough -- the primary interface
configs/                generated experiment matrix (27 configs + smoke)
evalsets/               COMMITTED: questions and gold documents
results/                one directory per run
context/                this folder
tests/                  17 offline tests, no network or model downloads
```

## The pipeline

```
  raw JSONL line          str
    json.loads()
      v
  raw row                 dict, 14 keys, nested
    _to_document()          |-- _doc_text() flattens everything into one string
      v
  document                dict {doc_id, text, title, store, categories, rating, price}
    cached -> data/raw/*.jsonl
    chunk_documents()
      v
  chunk                   dict {chunk_id, doc_id, ordinal, text, title, store}
    cached -> data/chunks/*.jsonl        chunk_id = "{doc_id}::{ordinal}"
    Embedder.encode()
      v
  vectors                 numpy (n_chunks, dim) float32     <- the only non-dict stage
    cached -> data/embeddings/*.npy
    store.build()
      v
  index                   FAISS/Chroma + parallel list[str] of chunk_ids
    cached -> data/indexes/
      v
  retrieve                top-k chunk_ids -> Retriever.by_id lookup
      v
  generate                Ollama, numbered contexts, "answer only from context"
      v
  score                   retrieval metrics + answer metrics
      v
  results/{run_id}/       metrics.json | config.json | records.jsonl
```

**No DataFrame anywhere in the pipeline.** Pandas appears only in notebooks for display and in
`report.py` for the leaderboard. Reasons: rows are heterogeneous (`details` has 246 distinct keys,
so a DataFrame would be ~99% null), the source is streamed rather than materialised, and the unit
of work is a document, not a column. For a *relational* source, a DataFrame path would be
appropriate — that is a Phase 4 item.

## Caching: three fingerprints

Every stage caches on a hash of exactly the inputs that produce it, so changing one knob
invalidates only what is downstream of it.

| id | hashes | invalidated by |
|---|---|---|
| `corpus_id` | data + chunking | corpus size, stride, chunk strategy/size/overlap |
| `index_id` | corpus_id + embedding + vectorstore | the above, plus embedder or backend |
| `run_id` | the whole config (minus `name`) | anything at all |

Consequences worth knowing:

- Changing the **LLM** does not re-embed anything — `index_id` excludes `generation`.
- Changing **chunk size** invalidates everything downstream, including the index.
- `run_id` is `{name}-{hash}`, so `results/emb-bge-small-a1b2c3d4e5/` is a fingerprint of the
  settings that produced it. **You cannot accidentally compare two runs that differed in a knob
  you forgot about.**
- Two configs that are genuinely identical share a fingerprint. That is why the baseline appears
  in several sweeps — it makes each sweep table self-contained.

## Design decisions embedded in the code

**Chunk ids carry their parent.** `"{doc_id}::{ordinal}"` means every hit traces back to a source
document with a string split. That is what lets metrics be scored at document level, which in turn
is what makes a 256-char config comparable to a whole-document one.

**Vectors are L2-normalised.** Inner product then equals cosine similarity, so FAISS's fast IP
search *is* cosine search. Every backend reports scores where **higher is better** — Chroma
returns cosine *distance*, so `chroma_store.py` flips it.

**Flat FAISS is the ceiling, not a competitor.** It is exhaustive and therefore exact. Approximate
backends are reported as a loss against it.

**Generation failures never kill a run.** A single bad Ollama call is caught and recorded as
`[generation error: ...]`; the sweep continues.

**`records.jsonl` holds everything.** Every retrieved chunk, every score, every generated answer.
Aggregates are for the leaderboard; this file is for the reviewer who wants to audit one query.

## Extension points

- **New vector DB** — implement `VectorStore` (4 methods), register in `registry.py`.
- **New chunking strategy** — add a branch in `chunking.py`, a `Literal` value in `ChunkConfig`.
- **New embedder** — config only, if it is a sentence-transformers model. Set the prefixes.
- **New corpus** — implement a loader returning `[{doc_id, text, ...}]`. The rest is agnostic.
- **New answer metric** — add to `metrics/answer.py::score_answer`; it flows to the leaderboard.
