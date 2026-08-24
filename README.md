# fitment-rag

A reproducible benchmark for **retrieval** over auto-parts product data. It answers one question
that most RAG write-ups skip:

> **Which part of a retrieval pipeline actually determines quality — and what does each part cost?**

Not a RAG demo. A controlled study: every setting is varied one at a time against a fixed set of
questions, and the price of each choice is reported next to its benefit.

Runs entirely on a CPU laptop. No GPU, no API keys, no paid services, no private data.

---

## Early result

From the retrieval comparison — 10,000 products, 39,026 chunks, 200 questions:

| method | hit@1 | recall@5 | MRR | ms/query |
|---|---|---|---|---|
| hybrid + cross-encoder rerank | 0.995 | 1.000 | 0.998 | 1389 |
| **BM25** (keyword only, no neural net) | **0.945** | 0.995 | 0.967 | 78 |
| hybrid (RRF, alpha = 0.5) | 0.740 | 0.920 | 0.811 | 82 |
| dense (all-MiniLM-L6-v2) | 0.545 | 0.705 | 0.612 | 3.4 |

**BM25 — a keyword-scoring formula from 1994 — beat the embedding model by 40 points on hit@1.**

Auto-parts questions are dominated by brand names and model codes ("olifant hydraulic rear"),
which are rare tokens. BM25 weights rare tokens heavily; embedding models have weak
representations for rare proper nouns. Dense retrieval returned products that were topically
right and specifically wrong.

Two caveats stated up front, because they matter more than the headline:

1. **The eval set is built by lexical uniqueness** — question terms are chosen until they appear
   in exactly one document. That is precisely what BM25 optimises for, so the gap is real but
   inflated by the benchmark's own construction.
2. **Hybrid scoring worse than BM25 alone is a tuning artefact**, not a finding. `hybrid_alpha`
   is fixed at 0.5, so weak dense results dilute strong BM25 ones. Nobody tuned it.

---

## Quick start

**Requires Python 3.10+ and [Poetry 2.x](https://python-poetry.org/docs/#installation).**
Poetry 1.x will not work — it ignores the PEP 621 `[project]` table.

```bash
git clone https://github.com/iam-neelesh8/fitment-rag.git
cd fitment-rag

poetry install --extras "nb dev"
poetry run fitment-rag doctor
```

`doctor` should print `ok` for torch, sentence_transformers, faiss, and rank_bm25.

### Run the benchmark

```bash
# 1. Download the corpus and build the questions  (~2 min, streams from HuggingFace)
poetry run fitment-rag build-evalset --config configs/phase1_smoke.yaml -n 200

# 2. One config end to end  (~15 min first time: embeds 39k chunks on CPU)
poetry run fitment-rag run --config configs/phase1_smoke.yaml

# 3. Compare retrieval methods  (fast, reuses the cached embeddings)
poetry run fitment-rag sweep --configs "configs/phase1/retrieval/*.yaml"

# 4. The leaderboard
poetry run fitment-rag compare --sort recall@5
```

Every stage caches, so re-running a config is instant and changing one setting only recomputes
what depends on it.

Other sweeps:

```bash
poetry run fitment-rag sweep --configs "configs/phase1/emb/*.yaml"     # ~2 hours
poetry run fitment-rag sweep --configs "configs/phase1/chunk/*.yaml"   # ~1 hour
```

### Notebooks

```bash
poetry run python -m ipykernel install --user --name fitment-rag \
    --display-name "Python (fitment-rag)"
poetry run jupyter lab notebooks/
```

Select the **Python (fitment-rag)** kernel.

| notebook | what it does |
|---|---|
| `00_understand_the_data.ipynb` | **Start here.** The real schema, measured: field fill rates, the `details` long tail, document lengths, and how a raw row becomes retrievable text |
| `01_pipeline_walkthrough.ipynb` | Builds the pipeline one stage at a time, stopping to show what changed |
| `02_retrieval_experiments.ipynb` | Runs the sweeps and produces the leaderboard |

---

## What gets compared

Every config comes from one shared base in `scripts/gen_configs.py`, so **the only difference
between two runs is the setting named in the filename**. Change two things at once and the result
tells you nothing.

```
                        base config
                             |
        +--------------------+--------------------+
        |                    |                    |
   [1a] EMBEDDER        [1b] CHUNKING       [1c] RETRIEVAL
   5 models             5 strategies        4 modes
        |                    |                    |
        +--------------------+--------------------+
                             |
                        leaderboard()
```

**14 configs — added, not multiplied.** A full 5x5x4 grid would be 100 runs and days of CPU. The
trade-off: this design finds which setting matters most, but cannot detect interactions between
settings.

<details>
<summary><b>1a — Embedding models (5)</b></summary>

| tag | model | params | dim |
|---|---|---|---|
| minilm-l6 | `all-MiniLM-L6-v2` | 22M | 384 |
| bge-small | `BAAI/bge-small-en-v1.5` | 33M | 384 |
| e5-small | `intfloat/e5-small-v2` | 33M | 384 |
| gte-small | `thenlper/gte-small` | 33M | 384 |
| mpnet-base | `all-mpnet-base-v2` | 110M | 768 |

bge and e5 were trained with instruction prefixes (`query:` / `passage:`). They are set in the
configs — omitting them silently costs several points of recall, and is one of the most common
ways a public embedding comparison ends up quietly rigged.
</details>

<details>
<summary><b>1b — Chunking (5)</b></summary>

Sizes are in **characters**. Measured on the real corpus:

| strategy | chunks per doc |
|---|---|
| `whole_doc` | 1.00 |
| `fixed 256 / 32` | 7.36 |
| `fixed 512 / 64` | 3.92 (default) |
| `fixed 1024 / 128` | 2.21 |
| `sentence 512 / 64` | 3.73 |

Median document is 1,438 characters, so 94% of documents split at 512. `fixed 256` costs 7.4x the
embedding time of `whole_doc` for the same corpus.
</details>

<details>
<summary><b>1c — Retrieval mode (4)</b></summary>

| mode | how it finds documents |
|---|---|
| `bm25` | word overlap, weighted so rare words count more. No neural network. |
| `dense` | embed the question, find nearest vectors by cosine similarity |
| `hybrid` | both, merged by reciprocal rank fusion |
| `hybrid + rerank` | as above, then a cross-encoder re-scores the top 50 |

BM25 is the honesty check. If the neural stack cannot beat a keyword baseline, that is the single
most useful thing this benchmark can report.
</details>

---

## How it works

```
  Amazon Automotive          [1] CORPUS       stream JSONL, cache locally
  product metadata     ->    load_documents()
  (HuggingFace, 5.35 GB)          |
                                  v
                             [2] CHUNK        ids are {doc_id}::{ordinal}
                             chunk_documents()
                                  |
                                  v
                             [3] EMBED        L2-normalised -> IP == cosine
                             Embedder
                                  |
                                  v
                             [4] INDEX        FAISS flat — exhaustive, exact
                             VectorStore
                                  |
     question  ------------> [5] RETRIEVE     dense / bm25 / hybrid
                             Retriever
                                  |
                                  v
                             [6] SCORE        hit@k, recall@k, MRR, nDCG
                             metrics/          scored at DOCUMENT level
                                  |
                                  v
                          results/{run_id}/
                            metrics.json  config.json  records.jsonl
```

`run_id` is a hash of the config, so a results directory is a fingerprint of the settings that
produced it. You cannot accidentally compare two runs that differed in a setting you forgot about.

Metrics are scored at **document** level, not chunk level. Chunk counts vary 7x across chunking
strategies, so chunk-level scoring would make the chunking comparison meaningless.

---

## Reproducibility

> **The corpus is regenerated, never redistributed. The eval set ships in the repo.**

The Amazon Reviews 2023 dataset is licensed for research with citation, not for re-hosting. So
`data/` is gitignored and rebuilt by a script. What *does* ship is `evalsets/` — the questions,
the answers, and which document each answer came from. That is what a skeptical reader actually
needs in order to check the numbers.

Three things make a result auditable:

- **`corpus_checksum`** in every `metrics.json` — a hash of the documents used. Run the same
  config, compare the checksum, and you know you started from identical inputs.
- **`records.jsonl`** — every retrieved chunk and every score, per query. Audit one question
  instead of trusting an average.
- **`config.json`** — the exact settings, stored next to the numbers they produced.

### How the questions are built

Answers are **extracted, not generated**, from metadata fields that are almost always present
(`store`, `price`, `average_rating`, `categories`). No LLM, no hallucination risk, and the correct
document is known by construction.

Three rules keep them honest, each with a regression test in `tests/test_evalset.py`:

1. **No verbatim title.** Questions use a minimal set of descriptive terms.
2. **No answer leakage.** Terms appearing in the answer are stripped, so "Who makes the Bosch
   brake pad?" cannot give away the answer "Bosch".
3. **Unique by construction.** Terms are added until exactly one document matches, so the correct
   label is never arbitrary.

Rule 3 exists because the first version violated it. Questions quoted the full product title, so
each question contained the first line of its own target document — string lookup, not retrieval.
Every config scored a perfect 1.0, BM25 included, and nothing could be distinguished from anything
else. The commit history has the fix.

---

## Layout

```
src/fitment_rag/      the library — notebooks import from here, nothing is duplicated
  config.py           typed YAML config; hashes into run_id / corpus_id / index_id
  data/amazon.py      HTTP streaming, row -> document flattening, checksum
  chunking.py         whole_doc / fixed / sentence
  embedding.py        sentence-transformers wrapper with a vector cache
  vectorstores/       FAISS flat + a pluggable VectorStore base class
  retrieval.py        dense, BM25, hybrid RRF, optional cross-encoder rerank
  evalset/build.py    deterministic question generator
  metrics/retrieval   hit@k, recall@k, precision@k, MRR, nDCG
  pipeline.py         the full run, with per-stage caching
  report.py           the leaderboard, shared by CLI and notebooks
notebooks/            the guided walkthrough — start at 00
configs/              generated experiment matrix (14 + smoke)
evalsets/             committed: the questions and their answers
context/              project brief, plan, architecture, decision log
tests/                33 offline tests — no network, no model downloads
```

Run the tests with `poetry run pytest -q`. They take under two seconds and need no network.

---

## Scope

This repo implements **retrieval only**. There is no answer-generation step and no LLM anywhere in
the code — an LLM cannot rescue a chunk that was never retrieved, so retrieval gets settled first.

Approximate indexes, the scale ladder to 200k documents, and small-LLM comparison are planned in
detail in [`context/02-plan.md`](context/02-plan.md) but are not implemented here.

### Known limitations

Stated because they make the reported numbers more credible, not less:

- One corpus, one seed, 10,000 documents
- Questions are template-generated, and their lexical construction favours BM25
- Exact search only, so no approximate-index comparison
- `hybrid_alpha` is untuned at 0.5
- Settings are compared one at a time, so interactions between them are invisible
- Everything measured on a single CPU laptop

---

## Data and licensing

See [`DATA_LICENSES.md`](DATA_LICENSES.md). The corpus is Amazon Reviews 2023 (McAuley Lab, UCSD),
used for research with citation and downloaded rather than redistributed. No scraped or
proprietary catalogs are used anywhere in this project.

## License

MIT
