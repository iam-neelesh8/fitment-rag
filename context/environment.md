# Environment

## Machine used for all reported results

| | |
|---|---|
| CPU | Intel Core Ultra 5 125U, 15 W, no discrete GPU |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | 3.13.7 |
| Poetry | 2.1.3 |
| torch | 2.13.0+cpu |

Every runtime in the README is from this machine. Embedding is the bottleneck throughout.

## Setup

```bash
poetry install --extras "nb dev"
poetry run fitment-rag doctor
```

Optional extras:

| extra | contents | needed for |
|---|---|---|
| `nb` | jupyterlab, ipykernel, matplotlib, ipywidgets | the notebooks |
| `dev` | pytest, ruff | tests and linting |

## Measured throughput

Embedding, e5-small on 10,000 documents:

| chunking | chunks | time | chunks/sec |
|---|---|---|---|
| whole_doc | 10,000 | 18.0 min | 9.3 |
| fixed 1024 | 21,971 | 25.2 min | 14.5 |
| sentence 512 | 37,277 | 22.3 min | 27.8 |
| fixed 512 | 39,026 | 31.6 min | 20.6 |

**chunks/sec varies 3×, so it is a useless predictor.** Throughput is roughly constant in *tokens*
— 2,600–3,700 tokens/sec — because a 1,024-character chunk carries four times the tokens of a
256-character one.

Consequence worth knowing: total embedding time barely changes across chunking strategies. Slicing
the same corpus differently does not change how many tokens the model reads. Chunking's real cost
is index size and search latency, which scale with chunk *count*.

Retrieval, 2,000 queries:

| mode | ms/query |
|---|---|
| dense | 3 |
| BM25 | 81 |
| hybrid | 80 |
| hybrid + cross-encoder rerank | 1,373 |

## Traps already hit

### Windows long-path limit breaks the torch install

`OSError: [WinError 3] The system cannot find the path specified` pointing at a deeply nested
`torch-*.dist-info/licenses/...` directory.

The project path is already deep and torch's license tree pushes past 260 characters. The committed
`poetry.toml` puts the venv at `C:\venvs` to avoid it. Alternatively, enable long paths (needs
admin):

```
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### `datasets` cannot load this dataset

`RuntimeError: Dataset scripts are no longer supported`. Version 4.x dropped script-based loaders
and Amazon-Reviews-2023 still ships one. The loader streams the raw JSONL instead, and `datasets`
is not a dependency. If a notebook raises `ModuleNotFoundError: No module named 'datasets'`, it is
running a stale cell — reload it from disk.

### Jupyter keeps a stale copy after files change on disk

*File → Reload Notebook from Disk*, then *Kernel → Restart*. Do not save first; that overwrites the
on-disk version with the stale one.

### Timings are contended

Long sweeps on a thermally limited laptop do not produce comparable timings. `gte-small` measured
27 ms/query against 4 ms for identically sized models, which is contention rather than a model
property. Re-measure serially on an idle machine before quoting any timing.

### HuggingFace symlink warning

Harmless on Windows; caching works, it just uses more disk. Silence with
`HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## Verifying a working setup

```bash
poetry run pytest -q                  # 44 tests, no network, under a second
poetry run fitment-rag doctor         # dependency check
```
