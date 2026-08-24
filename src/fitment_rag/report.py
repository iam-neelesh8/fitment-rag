"""Turn results/*/metrics.json into one comparison table.

Shared by the CLI and the notebooks so a leaderboard never has two definitions.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import RESULTS_DIR
from .stats import wilson_interval


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
        n = m.get("n_queries") or 0
        hit1 = m.get("hit@1")
        # NOTE: recall@k is deliberately absent. Every query has exactly one
        # relevant document, so recall@k == hit@k to the last decimal. Printing
        # both makes two metrics look like four.
        lo, hi = wilson_interval(round((hit1 or 0) * n), n)
        rows.append({
            "run": m.get("name"),
            "embedder": c["embedding"]["model"].split("/")[-1],
            "chunking": f"{c['chunking']['strategy']}/{c['chunking']['chunk_size']}",
            "retrieval": c["retrieval"]["mode"],
            "rerank": "yes" if c["retrieval"].get("reranker") else "-",
            "unit": c["retrieval"].get("unit", "chunk"),
            "chunks": m.get("n_chunks"),
            "n": n,
            "hit@1": hit1,
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "+/-pp": round((hi - lo) / 2 * 100, 1),
            "hit@5": m.get("hit@5"),
            "mrr": m.get("mrr"),
            "ndcg@5": m.get("ndcg@5"),
            "ms/query": m.get("retrieval_ms_per_query"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and sort in df.columns:
        df = df.sort_values(sort, ascending=False, na_position="last").reset_index(drop=True)
    return df
