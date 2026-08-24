"""Turn results/*/metrics.json into one comparison table.

Shared by the CLI and the notebooks so a leaderboard never has two definitions.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import RESULTS_DIR


def load_runs() -> list[dict]:
    runs = []
    for f in sorted(RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["_config"] = json.loads((f.parent / "config.json").read_text(encoding="utf-8"))
        runs.append(m)
    return runs


def leaderboard(sort: str = "recall@5") -> pd.DataFrame:
    rows = []
    for m in load_runs():
        c = m["_config"]
        rows.append({
            "run": m.get("name"),
            "embedder": c["embedding"]["model"].split("/")[-1],
            "chunking": f"{c['chunking']['strategy']}/{c['chunking']['chunk_size']}",
            "retrieval": c["retrieval"]["mode"],
            "rerank": "yes" if c["retrieval"].get("reranker") else "-",
            "docs": m.get("n_docs"),
            "chunks": m.get("n_chunks"),
            "hit@1": m.get("hit@1"),
            "recall@5": m.get("recall@5"),
            "mrr": m.get("mrr"),
            "ndcg@5": m.get("ndcg@5"),
            "ms/query": m.get("retrieval_ms_per_query"),
            "embed_s": m.get("embed_seconds"),
            "idx_MB": round((m.get("index_size_bytes") or 0) / 1e6, 1),
        })
    df = pd.DataFrame(rows)
    if not df.empty and sort in df.columns:
        df = df.sort_values(sort, ascending=False, na_position="last").reset_index(drop=True)
    return df
