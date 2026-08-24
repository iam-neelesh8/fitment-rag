# Environment

## Development machine

| | |
|---|---|
| CPU | Intel Core Ultra 5 125U @ 1.30 GHz (15W mobile) |
| RAM | 16 GB (15.4 GB usable) |
| GPU | Intel integrated graphics — **no CUDA** |
| Disk | ~767 GB free |
| OS | Windows 11 Home |
| Python | 3.13.7 |
| Poetry | 2.1.3 |
| Ollama | 0.32.0 |

**Everything in this project is sized for this machine.** `torch` installs as `2.13.0+cpu`.
See `05-decisions.md` for how the hardware shaped model selection.

## Setup

```bash
# Poetry 2.x reads the PEP 621 [project] table natively. Poetry 1.x does NOT --
# it would ignore it and expect a [tool.poetry] block. Check with: poetry --version
poetry install --extras "nb dev"

# register the Jupyter kernel (once)
poetry run python -m ipykernel install --user --name fitment-rag --display-name "Python (fitment-rag)"

poetry run fitment-rag doctor          # verify the environment
poetry run jupyter lab notebooks/
```

Optional extras:

| extra | pulls | for |
|---|---|---|
| `nb` | jupyterlab, ipykernel, matplotlib | the notebooks |
| `dev` | pytest, ruff | tests and linting |
| `vdb` | chromadb, qdrant-client | Phase 2a vector DB comparison |
| `judge` | ragas, langchain-ollama | Phase 4 LLM-judged metrics |

## Ollama models

Needed only for Phase 3 (and the optional LLM eval-set generator). Roughly 8 GB total.

```bash
ollama serve
ollama pull qwen2.5:1.5b-instruct     # the default
ollama pull llama3.2:1b
ollama pull gemma2:2b
ollama pull llama3.2:3b
ollama pull qwen2.5:3b-instruct
```

Start with the first two if you want results the same evening. If Ollama is down, runs still
proceed — generation is skipped and retrieval metrics remain valid.

---

## Traps already hit — do not rediscover these

### Windows long-path limit breaks `torch` install

**Symptom.** `OSError: [WinError 3] The system cannot find the path specified:
'...\.venv\Lib\site-packages\torch-2.13.0.dist-info\licenses\third_party\kineto\libkineto\
third_party\dynolog\third_party\DCGM\testing\python3\libs_3rdparty'`

**Cause.** The project path is already deep; torch's nested license tree pushes past 260 chars.

**Fix in place.** `poetry.toml` (committed) sets `virtualenvs.path = "C:\\venvs"`. The venv lives
at `C:\venvs\fitment-rag-<hash>-py3.13`.

**Alternative** (needs admin): enable long paths system-wide —
`reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f`

### `datasets` 4.x cannot load this dataset

**Symptom.** `RuntimeError: Dataset scripts are no longer supported, but found
Amazon-Reviews-2023.py`

**Fix in place.** `datasets` was removed as a dependency; the loader streams the raw JSONL with
`requests`. If you see `ModuleNotFoundError: No module named 'datasets'` in a notebook, you are
running a stale cell — reload the notebook from disk.

### Jupyter holds a stale copy after files change on disk

**Symptom.** A notebook cell runs code you already deleted.

**Fix.** *File → Reload Notebook from Disk*, then *Kernel → Restart*. **Do not save first** — that
overwrites the on-disk version with the stale one.

### HuggingFace symlink warning on Windows

Harmless. Caching works, it just uses more disk. Silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

---

## Runtime budgets on this machine

| operation | time |
|---|---|
| stream 500 rows | ~2 s |
| embed 1k docs (3.9k chunks, MiniLM) | under a minute |
| embed 10k docs (39k chunks) | 2–5 min |
| embed 50k docs (196k chunks) | 15–25 min |
| embed 200k docs (785k chunks) | 1–2 hours — the overnight run |
| generate one answer (1.5B, Q4) | seconds |
| generate one answer (3B, Q4) | slower; a 50-question eval is ~15 min |
| full test suite | 0.4 s |

Beyond ~500k documents, rent a GPU box. Say so in the write-up rather than pretending the ladder
ended naturally.

## Verifying a working setup

```bash
poetry run pytest tests/ -q            # 17 tests, offline, ~0.4s
poetry run fitment-rag doctor          # deps + Ollama status
poetry run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/00_understand_the_data.ipynb
```
