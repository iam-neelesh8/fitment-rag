# The plan

Staged so each phase produces a **reportable result before the next one starts**. A benchmark
that only pays off at the end is a benchmark that never ships.

```
 PHASE 0  understand           PHASE 1  small + fast        PHASE 2  scale
 the data                      1k docs, CPU minutes         1k -> 200k docs
 -----------------             --------------------         ------------------
 schema, fill rates      -->   1a  5 embedders        -->   2a  4 vector DBs
 details long tail             1b  5 chunk strategies       2b  scale ladder
 doc length distribution       1c  4 retrieval modes        (overnight run)
        |                             |                            |
        v                             v                            v
  is chunking even         which knob matters?          where does exact
  going to matter?         does dense beat BM25?        search stop scaling?

                          PHASE 3  generation              PHASE 4  depth
                          retrieval frozen                 optional
                          ------------------               ------------------
                          5 LLMs, 1-3B                     LLM-written evals
                          quality vs latency               multiple seeds
                          abstain / bluff rates            RAGAS on a GPU
                                  |                        ACES/PIES fitment
                                  v                        Qdrant backend
                          does the LLM matter,
                          or is retrieval the wall?
```

The organising principle: **a RAG system is a chain, and a chain is only measurable one link at a
time.** Change two things and you learn nothing.

---

## Phase 0 — Understand the data ✅ done

Notebook `00_understand_the_data.ipynb`. Findings that drive every later decision are in
[`03-data.md`](03-data.md). Headline: median document is 1,438 characters, so **94% of documents
split at a 512-char chunk size** — Phase 1b is a live experiment, not a formality.

---

## Phase 1 — Which link matters?

**Corpus:** 1,000 documents. **Runtime:** minutes per config. **Generation:** OFF.

Generation is off on purpose. An LLM cannot rescue a chunk that was never retrieved, so measuring
it here adds hours of CPU time and noise for no signal.

### 1a. Embedding model — *does model choice actually move retrieval?*

| tag | model | params | dim | note |
|---|---|---|---|---|
| `minilm-l6` | all-MiniLM-L6-v2 | 22M | 384 | the fast default |
| `bge-small` | BAAI/bge-small-en-v1.5 | 33M | 384 | needs a query prefix |
| `e5-small` | intfloat/e5-small-v2 | 33M | 384 | needs `query:`/`passage:` prefixes |
| `gte-small` | thenlper/gte-small | 33M | 384 | no prefix |
| `mpnet-base` | all-mpnet-base-v2 | 110M | 768 | the slow ceiling |

**The prefixes are not cosmetic.** bge and e5 were trained with asymmetric instructions; omitting
them silently costs several points of recall. They are set in the configs — this is one of the
most common ways a public embedding comparison ends up quietly rigged.

**Hypothesis:** on short product listings with heavy keyword overlap, differences will be *small*.
That is publishable, because the field's default assumption is that a bigger embedder always wins.

**Decide on quality-per-second, not raw rank.** The Phase 1a winner has to be affordable to run
200× at Phase 2 scale. If mpnet-base buys two points of recall for 5× the embedding time, it loses.

### 1b. Chunking — *does the split matter more than the model?*

Measured chunk multipliers on 1,000 real documents:

| strategy | chunks/doc | 1k | 10k | 50k | 200k |
|---|---|---|---|---|---|
| `whole_doc` | 1.00 | 1,000 | 10,000 | 50,000 | 200,000 |
| `fixed 256/32` | 7.36 | 7,362 | 73,620 | 368,100 | **1,472,400** |
| `fixed 512/64` | 3.92 | 3,925 | 39,250 | 196,250 | 785,000 |
| `fixed 1024/128` | 2.21 | 2,214 | 22,140 | 110,700 | 442,800 |
| `sentence 512/64` | 3.73 | 3,733 | 37,330 | 186,650 | 746,600 |

`fixed 256/32` is **7.4× the embedding cost of whole_doc** for the same corpus. Run it last.

The failure mode to watch: a tail chunk holding a part number but not the title, so it cannot
identify itself. That is what `chunk_overlap` mitigates and what `whole_doc` avoids entirely.

### 1c. Retrieval mode — *does the neural stack beat a 1994 keyword algorithm?*

`bm25` / `dense` / `hybrid` (reciprocal rank fusion) / `hybrid + cross-encoder rerank`.

**This is the credibility experiment.** BM25 has no neural network at all, and auto parts are full
of alphanumeric part numbers — exactly where lexical matching is strongest and dense embeddings
are weakest. If BM25 wins or comes close, report it loudly. A benchmark that never states its
baseline is not a benchmark.

The reranker reorders the top-50; it cannot find anything new. Expect it to help `mrr` more than
`recall@5`, and to be the slowest option by a wide margin.

### Exit criteria

You can state, with a table behind each:

1. **The baseline.** What did BM25 get?
2. **The biggest lever.** Which single knob moved `recall@5` most, and by how much?
3. **The surprise.** What contradicted the prediction you wrote before running?
4. **The cost.** Embed time, index size, and ms/query for the winning config.

---

## Phase 2 — What breaks at scale?

### 2a. Vector database — *what does approximate search cost you?*

FAISS flat / FAISS HNSW / FAISS IVF / Chroma.

Flat is exhaustive, therefore **exact**, therefore the recall ceiling. Every approximate index is
reported as a **loss against exact**, never as a standalone score:

```
recall_loss = recall@5(approximate) - recall@5(faiss_flat)
```

"Chroma got 0.71" says nothing. "Chroma matched exact recall while building 3× slower" is a finding.

At 1k documents all four will look identical with sub-millisecond search. **That is the correct
result** — approximate indexes solve a problem you do not have yet. Re-run after the scale ladder
and the table separates. Reporting both ends is the point.

### 2b. Scale — *how does quality decay as the corpus grows?*

| docs | chunks @512/64 | embed time (MiniLM, this CPU) | vectors in RAM |
|---|---|---|---|
| 1k | 3,925 | under a minute | 6 MB |
| 10k | 39,250 | 2–5 min | 60 MB |
| 50k | 196,250 | 15–25 min | 301 MB |
| 200k | 785,000 | 1–2 hours | **1.2 GB** |

Recall will drop — the same question now competes against 200× more distractors. The interesting
quantity is the **shape** of the decay: a cliff between 10k and 50k means chunking is not
discriminative enough; a slow linear slide is normal and reportable.

200k is the overnight run. Start it before bed. Past ~500k on a 15W CPU, rent a GPU box — and say
so in the write-up rather than pretending the ladder ended naturally.

---

## Phase 3 — Does the LLM matter?

Retrieval frozen at the Phase 1 winner. Five models, quantized, over Ollama.

| model | size (Q4) | speed on this CPU | role |
|---|---|---|---|
| `llama3.2:1b` | 1.3 GB | fastest | the floor — does size hurt? |
| `qwen2.5:1.5b-instruct` | 1.0 GB | fast | **the default** |
| `gemma2:2b` | 1.6 GB | moderate | different training family |
| `llama3.2:3b` | 2.0 GB | slow | quality step up |
| `qwen2.5:3b-instruct` | 1.9 GB | slow | quality ceiling here |

**7B+ deliberately excluded.** ~2–3 tokens/sec on this chip makes a 50-question eval take over an
hour. The study covers the 1–3B CPU class, which is the class that matters without a GPU budget.

### The headline is not "which model won"

```
recall@5         fraction of questions where the right doc WAS in the prompt   (the ceiling)
answer_contains  fraction where the model actually produced the right answer
-----------------------------------------------------------------------------
generation_gap   how much the LLM threw away
```

- **Wide gap** → the LLM is losing information it was handed. Model choice is the lever.
- **Narrow gap** → the model already extracts nearly everything available, and further hours spent
  on model selection are stolen from chunking and retrieval, where the remaining error lives.

Most RAG projects never compute this, and consequently optimise the wrong half of the system.

### Failure modes to read manually, not just aggregate

- **Bluffing** — confident answer when the passages contained none. The expensive one.
- **Copying** — echoing a whole passage. Scores well on `answer_contains`, badly on `answer_f1`,
  which is exactly why both are reported.
- **Over-abstaining** — "I don't know" when the answer was in passage 2. Common in 1B models.
- **Format drift** — preamble and restatement. Costs `answer_em`; means the prompt needs tightening.

---

## Phase 4 — Optional depth

Roughly in order of value per hour:

1. **LLM-generated eval questions.** `--generator llm` writes harder, more natural questions.
   Keep both sets and report separately; the template set stays the headline because it carries
   zero hallucination risk.
2. **Multiple seeds.** Every current number is a single sample. Three seeds and a standard
   deviation turns "0.71 beats 0.68" into a claim that survives scrutiny.
3. **RAGAS** (`--extras judge`) for faithfulness and answer relevancy. Worth doing once a GPU hour
   is available — a 1.5B judge scoring a 1.5B generator is not a measurement.
4. **A real fitment layer.** ACES/PIES sample data turns this from a generic product benchmark
   into a domain-credible auto-parts one. This is the single highest-value upgrade for the
   ecommerce career goal. See `DATA_LICENSES.md`.
5. **Approximate indexes.** `vectorstore.py` holds one exact FAISS index; HNSW, IVF or an
   external store would slot in beside it.
6. **Normalised SQL view of the catalog.** Products + EAV attributes + categories as real tables,
   then denormalise back to documents with a `GROUP_CONCAT`. Enables metadata filtering
   ("ceramic brake pads under $50") and matches the shape real catalog data arrives in.

---

## Metrics, and why these

**Retrieval — scored at document level**, so a 256-char config stays comparable to whole-document.

| metric | question |
|---|---|
| `hit@k` | was the right document anywhere in the top k? |
| `recall@k` | what fraction of relevant documents were found? |
| `mrr` | how high did the first correct document rank? |
| `ndcg@k` | rank-weighted quality of the whole list |

**Answers — deterministic**, from `metrics/answer.py`.

| metric | question |
|---|---|
| `answer_contains` | did the ground-truth string survive into the answer? |
| `answer_f1` | token overlap — catches passage-copying that `contains` rewards |
| `answer_em` | exact match after normalisation |
| `abstain_rate` | how often did it say "I don't know"? |

`abstain_rate` is not decoration. The prompt instructs the model to abstain when the context lacks
the answer, so a near-zero abstain rate alongside mediocre accuracy means the model is **bluffing** —
precisely the failure you want visible in a results table.

---

## Where the eval questions come from

A benchmark is only as trustworthy as its ground truth, so this is deliberately boring.

**Template generator (the default).** Questions are built from structured fields that are almost
always present — `store`, `price`, `average_rating`, `categories`. The ground-truth answer is
**extracted, not generated**, so hallucination risk is zero and the gold document is known by
construction:

> "Which brand or store sells the product X?" → answer = the `store` field → gold doc = that row

Deliberately *not* built from `details`: 246 distinct keys across 500 sampled products, most
appearing on a handful of rows. Ground truth must be reliable before it can be a benchmark.

**LLM generator (optional).** A local model reads a listing and writes a natural question. Richer
phrasing, but every item is tagged `generator: llm` so a reader can filter them out.

Both live in `evalsets/*.jsonl`, **committed to git**. That is the deal that makes this
reproducible without redistributing licensed data: the corpus is regenerated by a script, while
the questions, answers, and gold documents ship where anyone can inspect them.

---

## What "done" looks like

A README whose every claim points at a row in `results/leaderboard.csv`, with a corpus checksum
proving the reader started from the same documents.

Stated limitations are part of the deliverable, not an apology: one corpus, template-generated
questions, a single seed, one machine, 1–3B models on CPU. Naming them makes the numbers you *do*
report more credible, not less.
