# Decision log

Every non-obvious choice, and the reasoning. Written so future-you does not re-litigate a settled
question or, worse, silently undo a decision that had a reason behind it.

---

### Use only openly licensed data; regenerate the corpus rather than ship it

**Decision.** The corpus is Amazon Reviews 2023 (research use + citation). `data/` is gitignored
and rebuilt by a script. No retailer or manufacturer catalog PDFs anywhere in the project.

**Why.** Reproducibility does not require hosting source files — it requires that someone else can
arrive at the same numbers. A corpus a reader cannot legally regenerate makes the benchmark
*less* reproducible, not more. `corpus_checksum()` closes the gap: a reader verifies they rebuilt
the same documents before comparing any metric.

**Consequence.** Zero legal caveats. The whole thing runs from one command.

---

### Stream the raw JSONL over HTTP instead of `datasets.load_dataset`

**Decision.** `data/amazon.py` reads `raw/meta_categories/meta_Automotive.jsonl` with `requests`.
`datasets` was removed as a dependency entirely.

**Why.** `datasets` 4.x removed script-based loaders and this dataset still ships one, so
`load_dataset` raises `RuntimeError: Dataset scripts are no longer supported`. Pinning
`datasets<4` was the alternative; streaming is simpler, faster (500 rows in ~2s), removes a heavy
dependency, and never risks pulling the full 5.35 GB.

**Consequence.** Sampling is a deterministic prefix scan rather than a uniform draw. Stated in
`03-data.md` and in the loader docstring.

---

### Sample by prefix + stride, not by hash over the full file

**Decision.** Take every `stride`-th usable row from the start; `stride=4` up to 10k docs,
`stride=1` above.

**Why.** Uniform sampling means streaming all 2M rows (~5.35 GB) even for a 1k corpus — roughly 35
minutes for the smoke test. Stride spreads the sample across a wider window at bounded cost.

**Trade-off accepted.** The sample is a prefix, so it is not uniform over the full catalog. It is
exactly reproducible, and the scale ladder is nested (10k ⊃ 1k), which is convenient. Named as a
limitation rather than hidden.

---

### Dicts and JSONL end to end; no DataFrame in the pipeline

**Decision.** Documents and chunks are `list[dict]`, cached as JSONL. Pandas only in notebooks and
`report.py`.

**Why.** `details` holds 246 distinct keys across 500 products — as columns that is ~99% null. The
source is streamed, so nothing can be materialised up front. And the unit of work is a document,
not a column: there is no groupby, join, or column-wise math anywhere.

**Not the reason:** speed. Embedding is 2–3 orders of magnitude slower than any row-iteration
method, so how rows are walked is never the bottleneck.

**Where this flips.** For a SQL-backed catalog — rectangular, homogeneous — `pd.read_sql` then
row→text is the right shape, and normalising `details` into an EAV table is the right model. That
is a Phase 4 item, and it matches how ACES/PIES actually structure data.

---

### Score retrieval at document level, not chunk level

**Decision.** `chunk_id` is `"{doc_id}::{ordinal}"`; metrics dedupe to `doc_id` before scoring.

**Why.** Chunk counts vary 7× across chunking strategies (1.00 to 7.36 chunks/doc). Chunk-level
recall would make Phase 1b meaningless — a strategy producing more chunks would score differently
for purely mechanical reasons.

---

### Deterministic answer metrics instead of RAGAS as the headline

**Decision.** `metrics/answer.py`: exact match, contains, token F1, abstain rate. RAGAS available
via the `judge` extra but not the default.

**Why.** RAGAS scores every answer with another LLM call. On a 15W CPU that is hours per sweep,
and the judge would be a 1.5B model scoring a 1.5B model. **That is not a measurement.** The
deterministic metrics cost nothing, are perfectly reproducible, and answer the question that
matters: did the model use the retrieved context?

**Revisit when** a GPU hour is available.

---

### Both `answer_contains` and `answer_f1`, not just one

**Why.** `contains` alone rewards a model that echoes an entire passage — the ground truth is in
there somewhere. `token_f1` punishes that padding. Together they distinguish *answering* from
*copying*.

---

### `abstain_rate` is a first-class metric

**Why.** The prompt instructs the model to reply "I don't know" when the context lacks the answer.
A near-zero abstain rate alongside mediocre accuracy means the model is **bluffing** — the most
expensive RAG failure mode, and invisible unless measured.

---

### Generation OFF for all of Phase 1

**Why.** An LLM cannot rescue a chunk that was never retrieved. Running it during retrieval
experiments adds hours of CPU time and noise for no signal.

---

### BM25 is in the comparison, not assumed beaten

**Why.** This is the credibility experiment. BM25 has no neural network; auto parts are full of
alphanumeric part numbers where lexical matching is strongest. A benchmark that never reports its
baseline is not a benchmark — and if BM25 wins, that is the most useful finding available.

---

### Model sizes chosen for a 15W CPU with no GPU

**Decision.** Embedders 22M–110M. LLMs 1–3B quantized. 7B+ excluded.

**Why.** `mistral:7b` runs at ~2–3 tokens/sec here, so a 50-question eval takes over an hour, and
the experiment matrix has 27 configs. The 1–3B CPU class is also precisely the class that matters
for anyone deploying RAG without a GPU budget — so the constraint produces a *more* relevant study,
not a compromised one.

**State it as scope, not apology.**

---

### Instruction prefixes set per embedding model

**Decision.** `query_prefix` / `doc_prefix` in config; bge gets its retrieval instruction, e5 gets
`query:` / `passage:`.

**Why.** Those models were trained with asymmetric instructions. Omitting them costs several points
of recall silently — one of the most common ways a public embedding comparison ends up rigged.

---

### Config hashes into the run id

**Decision.** `run_id = "{name}-{sha256(config)[:10]}"`.

**Why.** A results directory becomes a fingerprint of the settings that produced it. You cannot
accidentally compare two runs that differed in a knob you forgot you changed.

---

### Every config generated from one base by a script

**Decision.** `scripts/gen_configs.py` writes all 27 configs from a single `BASE` dict.

**Why.** Hand-edited YAML is where silent inconsistencies enter a benchmark. With a generator, the
only difference between two configs is the knob named in the filename — which is the entire premise
of one-variable-at-a-time comparison.

---

### Notebooks as the primary interface; CLI for sweeps

**Why.** The goal is understanding, not just numbers. Notebooks import from `src/` rather than
redefining anything, so what you learn in a notebook is what runs in a sweep. The CLI exists for
long unattended sweeps and CI.

---

### Poetry venv at `C:\venvs`, not in-project

**Decision.** Committed `poetry.toml` sets `virtualenvs.path = "C:\\venvs"`.

**Why.** The project path is deep, and `torch` ships a pathologically nested license directory. An
in-project `.venv` overran the Windows 260-character limit and `pip install` failed outright. A
short venv root fixes it without needing admin rights to enable long paths.

---

### `notebooks/*.ipynb` committed **with** outputs

**Why.** The outputs are the evidence. A reader should be able to see the measured numbers without
running anything, and diffs show when a result actually changed.

**Cost accepted.** Noisier git diffs.
