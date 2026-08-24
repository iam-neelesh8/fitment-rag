# Project brief

## What this is

`fitment-rag` is a reproducible benchmark for retrieval-augmented generation over auto-parts
product data. It answers a specific question that most RAG write-ups skip:

> **Which part of a RAG pipeline actually determines quality — and what does each part cost?**

Not "here is a RAG demo." A controlled study where every knob is varied one at a time against a
fixed eval set, with the price of each choice reported alongside its benefit.

## Why this project, for this person

Three goals, in priority order:

1. **Go deep on RAG architecture.** Not framework-level familiarity — mechanism-level. What
   chunking does to recall, why an approximate index loses accuracy, where a small LLM drops
   information it was handed.
2. **Build something recruiters can verify.** Ecommerce is the target domain, and auto parts is
   the current one, so the benchmark sits exactly at that intersection. The artifact is designed
   to survive a technical interviewer opening the repo and asking "how do you know?"
3. **Practice research habits.** Predictions written before runs, baselines reported even when
   they win, limitations stated rather than hidden, and every number traceable to a config hash.

## What success looks like

A stranger clones the repo, runs three commands, and lands on the same numbers:

```bash
poetry install --extras "nb dev"
poetry run fitment-rag build-evalset --config configs/phase1_smoke.yaml
poetry run fitment-rag sweep --configs "configs/phase1/**/*.yaml"
```

They can then check the working, not just the conclusion:

- `results/*/metrics.json` carries a **corpus checksum** — proof they started from the same documents
- `results/*/config.json` is the exact settings, and `run_id` is a hash of them
- `results/*/records.jsonl` holds **every retrieved chunk and generated answer**, so any single
  query can be audited rather than trusting an average
- `evalsets/*.jsonl` holds the questions and gold documents, committed to git

## Deliberate non-goals

- **Not a production system.** No API, no serving layer, no latency SLO.
- **Not a fitment resolver.** The corpus has no structured vehicle table (see `03-data.md`). This
  is product-listing retrieval in the auto-parts domain, and the write-up must say so.
- **Not a leaderboard chase.** A result showing that BM25 matches dense retrieval, or that
  embedding model choice barely matters, is a *finding* and gets reported as one.
- **Not GPU-scale.** The study covers small embedders and 1–3B LLMs on CPU. That is a stated
  scope, not an apology — it is the regime that matters for anyone without a GPU budget.

## Audience

Written for two readers at once:

- **A technical interviewer** who will open the repo, look for a baseline, and check whether the
  claims are falsifiable.
- **Future you**, six months from now, who will not remember why `stride=4` exists or why RAGAS
  isn't the headline metric. Hence `05-decisions.md`.
