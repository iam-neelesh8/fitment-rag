# Project context

Everything about *why* this project is the way it is. The repo root holds what a stranger needs
(`README.md`, `DATA_LICENSES.md`); this folder holds what a contributor — or you in three months —
needs to make good decisions without re-deriving them.

| file | what it answers |
|---|---|
| [`01-project-brief.md`](01-project-brief.md) | What is this, who is it for, what does success look like |
| [`02-plan.md`](02-plan.md) | The full phased experiment plan, with budgets and exit criteria |
| [`03-data.md`](03-data.md) | The corpus: schema, measured statistics, what it can't do |
| [`04-architecture.md`](04-architecture.md) | Code layout, the pipeline, the caching model |
| [`05-decisions.md`](05-decisions.md) | Decision log — every non-obvious choice and its reasoning |
| [`06-environment.md`](06-environment.md) | Hardware, Poetry setup, and the traps already hit |

Phase diagrams (one per phase, draw.io) are generated into `data/diagrams/`, which is
gitignored — run `poetry run python data/diagrams/gen_diagrams.py` to produce them.

## The one-paragraph version

An open, reproducible RAG benchmark over auto-parts product listings. It compares embedding
models, chunking strategies, retrieval modes, vector databases, and small local LLMs — on a corpus
anyone can regenerate with one command, scored against an eval set committed to the repo. The
entire thing runs on a CPU laptop with no GPU, no API keys, and no private data.

## The rule everything else follows

> **The corpus is regenerated, never redistributed. The eval set ships in the repo.**

Every design decision in `05-decisions.md` traces back to that sentence.
