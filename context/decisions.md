# Decisions

Why things are the way they are. The README covers what the project does and how to run it; this
covers the choices behind it, so they do not get re-litigated or silently undone.

---

## Data

### Use only openly licensed data, and regenerate rather than redistribute

The corpus is Amazon Reviews 2023, licensed for research with citation but not for re-hosting. So
`data/` is gitignored and rebuilt by a script, while the eval set is committed.

Reproducibility does not require hosting source files — it requires that someone else can arrive
at the same numbers. A corpus a reader cannot legally regenerate would make the benchmark *less*
reproducible, not more. `corpus_checksum()` closes the gap: compare hashes before comparing scores.

### Stream the raw JSONL instead of using `datasets.load_dataset`

`datasets` 4.x removed script-based loaders and this dataset still ships one, so `load_dataset`
raises. The source file is plain JSONL, so it is streamed line by line with `requests` and
abandoned once enough rows are collected. `datasets` was dropped as a dependency entirely.

### Sample by prefix and stride, not uniformly

Drawing uniformly from 2M rows means streaming the whole 5.35 GB file for even a 1k corpus —
about 35 minutes for a smoke test. The loader takes every `stride`-th usable row from the start and
stops.

Trade-off accepted: the sample is a prefix, not a uniform draw. It is exactly reproducible, and the
corpus sizes are nested, which is convenient. This is a real limitation and is named in the README.

### Dicts and JSONL end to end, no DataFrame in the pipeline

`details` holds 246 distinct keys across 500 sampled products — as columns that is ~99% null. The
source is streamed, so nothing can be materialised up front. Every operation is per-document; there
is no groupby, join, or column-wise arithmetic anywhere.

Not a performance argument. Embedding is orders of magnitude slower than any row-iteration method,
so how rows are walked is never the bottleneck. For a SQL-backed catalog — rectangular and
homogeneous — `pd.read_sql` then row-to-text would be the right shape.

---

## Measurement

### Score per document, not per chunk

`chunk_id` is `{doc_id}::{ordinal}`, and metrics deduplicate to `doc_id` before scoring. Chunk
counts vary 7× across strategies, so chunk-level recall would make the chunking comparison
meaningless.

### `top_k` counts documents, not chunks

This was wrong initially and it invalidated the entire chunking comparison. With `top_k` counting
chunks, a strategy producing 7.4 chunks per document fitted fewer distinct documents into a 5-chunk
window than one producing 1.0 — so it got fewer chances to be right for reasons unrelated to
retrieval quality.

Measured before the fix, distinct documents inside the top 5 chunks:

| strategy | distinct docs |
|---|---|
| whole_doc | 5.00 |
| fixed 1024 | 4.44 |
| sentence 512 | 4.04 |

Fixing it reversed the ranking. `whole_doc` went from best to second-worst; `fixed256` went from
worst to second-best.

### 2,000 questions, not 200

At n=200 the 95% interval on hit@1 is ±5 pp and power to detect a 3 pp difference is 0.19. At
n=2,000 the interval is ±1.6 pp and power is 0.93.

This matters: the chunking effect looked like nothing at n=200 (p=0.31) and is real at n=2,000
(p=0.0007). Same effect, different resolution.

### Confidence intervals and paired tests on every claim

Wilson intervals, because scores reach 0.98 where the normal approximation runs past 1.0. Paired
exact McNemar, because every config answers the same questions — ignoring that pairing discards
most of the statistical power.

This exists because the first analysis reported a 2.5 pp chunking "trend" and invented a mechanism
for it. Every one of those comparisons had p > 0.3.

### Held-out dev/test split

`e5-small` was chosen as the base *because* it won, so reporting its score on the same questions
inflates it. Decisions are made on `dev`, headline numbers come from `test`. Split assignment
hashes `query_id`, so it is deterministic and cannot be nudged afterwards by picking a seed.

The measured effect turned out small — mean dev−test gap 0.6 pp — but that is a result, not an
assumption.

### `recall@k` is not reported

Every query has exactly one relevant document, so `recall@k` equals `hit@k` exactly. Printing both
made two metrics look like four.

---

## Eval set

### Answers are extracted, never generated

Questions come from metadata fields that are almost always present, with the answer read directly
from the field. No model writes ground truth, so there is no hallucination risk and the correct
document is known by construction.

Deliberately not built from `details` — 246 keys, most appearing on a handful of products. Ground
truth has to be reliable before it can be a benchmark.

### Three rules, each with a regression test

1. **No verbatim titles.** The first version quoted the full product title, so every question
   contained its target document's first line. That is string lookup, not retrieval: every config
   scored a perfect 1.000, BM25 included, and nothing could be distinguished from anything.
2. **No answer leakage.** Terms appearing in the answer are stripped from the question.
3. **Unique by construction.** Terms are added until exactly one document matches.

Part-number tokens are excluded from question terms. They are perfect lexical keys and recreate the
same trivial-match problem.

Known bias: rule 3 validates uniqueness by lexical intersection, which is what BM25 optimises for.
BM25's margin is real but somewhat inflated by this. Named in the README.

---

## Scope

### Retrieval only, no generation

An LLM cannot recover a document that was never retrieved. Measuring generation before retrieval is
settled adds hours of compute and noise for no signal. Generation code was removed rather than left
unused.

### Exact search only

At ~39k vectors, brute-force search is sub-millisecond. Approximate indexes solve a problem that
does not exist at this scale, and exact recall is the ceiling any later comparison would need.

### Models sized for a CPU laptop

Embedders are 22M–110M parameters. Anything larger cannot be swept 14 times on a 15 W chip. The
claims are scoped to that class rather than presented as general.

### Instruction prefixes set per model

bge and e5 were trained with asymmetric instructions (`query:` / `passage:`). Omitting them
silently costs several points of recall, and is one of the most common ways a public embedding
comparison ends up quietly rigged.

---

## Engineering

### Config hashes into the run id

`run_id = {name}-{sha256(config)[:10]}`. A results directory names the exact settings that produced
it, so two runs differing in any setting cannot be confused.

### Every config generated from one base

`scripts/gen_configs.py` writes all 14 from a single `BASE` dict. Hand-edited YAML is where silent
inconsistencies enter a benchmark; with a generator, the only difference between two configs is the
setting named in the filename.

### One vector store module, no abstract base class

An interface and a registry for a single implementation is ceremony. Adding a second backend is a
class with four methods next to the first one.

### Poetry venv outside the project

The project path is deep and `torch` ships a heavily nested license directory, which overran the
Windows 260-character limit. The committed `poetry.toml` puts the venv at `C:\venvs`.

### Notebooks committed with outputs

The outputs are the evidence. A reader should see the measured numbers without running anything,
and diffs show when a result actually changed. Cost accepted: noisier diffs.

---

## Errors caught along the way

Four, each of which would have produced a confident and wrong result:

1. **Saturated eval.** Questions quoted the document title verbatim; every config scored 1.000.
2. **Bad default embedder.** BM25 was benchmarked against MiniLM, the worst model tested. The gap
   was reported as 40 pp; against a competent embedder it is 6.9 pp.
3. **`top_k` confound.** Counting chunks instead of documents reversed the chunking ranking.
4. **Underpowered eval.** n=200 turned a real 2 pp chunking effect into a null result.

Worth recording because the same failure modes recur, and because the third and fourth were only
found by checking work that already looked finished.
