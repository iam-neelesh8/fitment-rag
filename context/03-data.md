# The data

Everything here was **measured** by `notebooks/00_understand_the_data.ipynb`, not quoted from
documentation. Reproduce it by re-running that notebook.

## Source

| | |
|---|---|
| Dataset | Amazon Reviews 2023 — Automotive product metadata |
| Maintainer | McAuley Lab, UC San Diego |
| File | `raw/meta_categories/meta_Automotive.jsonl` |
| Size | **5.35 GB**, ~2 million products |
| Terms | Research use with citation. Not licensed for re-hosting. |
| Access | Streamed line-by-line over HTTP; **never downloaded whole** |

Full licensing detail and citation in `DATA_LICENSES.md`.

## Raw schema — 14 fields

```
main_category    str        "Automotive" on every row -- zero discriminative signal
title            str        the strongest identifying text
average_rating   float      always present, reliable eval target
rating_number    int        always present
features         list[str]  bullet points; short factual claims
description      list[str]  long marketing prose
price            float|None often null
images           list[dict] DROPPED -- not a multimodal benchmark
videos           list[dict] DROPPED
store            str        the practical "brand" field
categories       list[str]  hierarchical path
details          dict       free-form key/value -- the rich, messy one
parent_asin      str        our doc_id
bought_together  None       almost always null
```

## Measured statistics (1,000 streamed rows)

| statistic | value |
|---|---|
| rows → documents | 1,000 → 1,000 (**0 dropped** by the 120-char minimum) |
| document length, median | **1,438 characters** |
| document length, mean | 1,533 |
| document length, max | 6,741 |
| distinct `details` keys (500 rows) | **246** |
| chunks per document @ 512/64 | **3.92** |
| documents fitting in one 512-char chunk | **6.1%** |

### Why the 1,438-char median matters

It is comfortably above the 512-char default chunk size, so **94% of documents split**. Chunking
experiments will produce real signal rather than a flat null result. Had the median come in under
512, Phase 1b would have been a formality.

### Why 246 `details` keys matters

A handful of keys are common (Manufacturer, Brand, Item Weight); hundreds appear on a few products
each. That long tail is the core retrieval difficulty — the answer to "what is the thread pitch"
lives in a key present on ~2% of products.

It is also why **eval questions are not built from `details`**. Ground truth must be reliable
before it can be a benchmark, so questions come from the always-present fields instead.

## What this data can and cannot support

**Can:**

- Attribute lookup — brand, price, rating, category
- Feature questions where features/description state the fact
- Semantic product search — "ceramic brake pads for daily driving"

**Cannot:**

- **True fitment.** There is no vehicle table and no structured year/make/model anywhere. Listings
  say "fits 2015–2018 Civic" in *prose*; there is no queryable fitment relation.
- Part interchange — which part numbers substitute for which
- Technical specs — torque values, service intervals, procedures
- Anything authoritative — these are seller-written listings, marketing copy and errors included

**Consequence for the write-up:** this is a *product-listing retrieval benchmark in the auto-parts
domain*, not a fitment-resolution system. Claiming otherwise is the kind of overstatement a
technical interviewer catches in one question. ACES/PIES sample data is the upgrade path that
would make the stronger claim true.

## Row → document

`data/amazon.py::_doc_text` flattens each row into one labelled string:

```
Title: Monroe SC2951 Magnum Steering Damper
Brand/Store: Monroe
Categories: Automotive > Replacement Parts > Shocks, Struts & Suspension > Stabilizers
Features: Full Displaced Valving: ... | All Weather Fluid: ... | Sintered Iron Piston: ...
Description: The Monro-Magnum steering stabilizer stops vibration before it gets to the driver...
Details: Manufacturer: Tenneco | Brand: Monroe | Item Weight: 4.8 pounds | Item model number: SC2951
```

Three things make it more than naive concatenation:

1. **Separators encode structure.** `categories` joins with `" > "` (a hierarchy), `features` with
   `" | "` (independent claims), `description` with `" "` (continuous prose), `details` as
   `k: v` pairs joined by `" | "`.
2. **Empty fields vanish entirely.** No `Price: None` lines — that would be noise in the embedding
   and a false match for price queries.
3. **Field labels are deliberate.** `Title:`, `Brand/Store:` give the embedder lexical anchors, so
   a query containing "brand" has something to match.

**What is dropped is unrecoverable.** `images`, `videos`, `bought_together`, and `main_category`
never enter the text, so no embedding model can ever surface them. That is the accuracy ceiling,
fixed before a single vector is computed.

Note that `title`, `store`, `price` survive **twice** — inside `text` (retrievable) and as sibling
keys on the document dict (structured). The structured copies are what the eval-set builder reads
to construct questions with known-correct answers.

## Sampling: a deterministic prefix scan

`n_docs=1000` does **not** draw uniformly from all 2M rows — that would mean streaming the entire
5.35 GB file for even a small corpus. Instead the loader takes every `stride`-th usable row from
the start of the file and stops.

- Fully reproducible: same file + same stride ⇒ same corpus, verifiable via `corpus_checksum()`
- Nested: a 10k corpus is a superset of the 1k corpus at the same stride
- `stride=4` for corpora ≤10k; `stride=1` above that, to avoid over-streaming

This is a real limitation and belongs in the write-up: the sample is a prefix, not a uniform draw.

## Data on disk (all gitignored)

```
data/raw/          cached documents, JSONL
data/chunks/       chunked corpus, JSONL, keyed by corpus_id
data/embeddings/   vectors, .npy, keyed by index_id
data/indexes/      FAISS / Chroma indexes, keyed by index_id
```

Nothing under `data/` is ever committed. `evalsets/` and `results/*/metrics.json` are.
