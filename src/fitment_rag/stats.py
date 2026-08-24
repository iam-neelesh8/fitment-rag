"""Confidence intervals and significance tests for retrieval metrics.

Wilson intervals rather than the normal approximation, since scores reach 0.98
where the normal approximation runs past 1.0. Paired McNemar rather than a
two-sample test, since every config answers the same questions and ignoring
that pairing throws away most of the statistical power.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import RESULTS_DIR


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_wins: int, b_wins: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(a_wins, b_wins) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def load_per_query(run_id: str, metric: str = "hit@1") -> dict[str, float]:
    """Per-query scores for one run, keyed by query_id."""
    path = RESULTS_DIR / run_id / "records.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        return {r["query_id"]: r["metrics"][metric] for r in map(json.loads, fh)}


def _run_dirs() -> dict[str, str]:
    """name -> run_id, for every run on disk."""
    out = {}
    for f in sorted(RESULTS_DIR.glob("*/metrics.json")):
        out[json.loads(f.read_text(encoding="utf-8"))["name"]] = f.parent.name
    return out


def compare_runs(name_a: str, name_b: str, metric: str = "hit@1") -> dict:
    """Paired comparison of two runs. Returns the numbers a claim needs."""
    runs = _run_dirs()
    missing = [n for n in (name_a, name_b) if n not in runs]
    if missing:
        raise KeyError(f"no results for: {missing}. Available: {sorted(runs)}")

    a, b = load_per_query(runs[name_a], metric), load_per_query(runs[name_b], metric)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(f"{name_a} and {name_b} share no query ids -- different eval sets?")

    a_wins = sum(1 for q in shared if a[q] > b[q])
    b_wins = sum(1 for q in shared if b[q] > a[q])
    p = mcnemar_exact(a_wins, b_wins)
    mean_a = sum(a[q] for q in shared) / len(shared)
    mean_b = sum(b[q] for q in shared) / len(shared)

    return {
        "a": name_a, "b": name_b, "metric": metric, "n": len(shared),
        "score_a": round(mean_a, 4), "score_b": round(mean_b, 4),
        "difference": round(mean_a - mean_b, 4),
        "a_wins": a_wins, "b_wins": b_wins, "discordant": a_wins + b_wins,
        "p_value": round(p, 4), "significant": p < 0.05,
    }


def significance_table(names: list[str], metric: str = "hit@1") -> "object":
    """Every pairwise comparison among `names`, sorted by p-value.

    k configs give k(k-1)/2 tests, so at alpha 0.05 about one in twenty looks
    significant by chance. Read `bonferroni_significant` when scanning the whole
    matrix for a winner.
    """
    import pandas as pd

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            try:
                rows.append(compare_runs(a, b, metric))
            except (KeyError, ValueError):
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n_tests = len(df)
    df["bonferroni_significant"] = df["p_value"] < (0.05 / n_tests)
    df["n_tests"] = n_tests
    return df.sort_values("p_value").reset_index(drop=True)


def power_for_difference(baseline: float, delta: float, n: int,
                         discordance: float = 0.15) -> float:
    """Approximate power to detect `delta` at sample size `n`.

    Normal approximation to the paired sign test, for planning sample sizes
    rather than for reporting.
    """
    if delta <= 0 or n <= 0:
        return 0.0
    n_disc = max(1.0, n * discordance)
    p_shift = 0.5 + delta / (2 * discordance)
    p_shift = min(0.999, p_shift)
    se = math.sqrt(0.25 / n_disc)
    z = (p_shift - 0.5) / se - 1.96
    return max(0.0, min(1.0, 0.5 * (1 + math.erf(z / math.sqrt(2)))))


# Choosing a config on the eval set and then reporting its score on that same
# set inflates the winner. Decisions are made on `dev`, headline numbers come
# from `test`. Assignment hashes query_id, so it is deterministic and cannot be
# nudged afterwards by picking a seed.

def split_of(query_id: str) -> str:
    """Deterministic 50/50 assignment of a query to 'dev' or 'test'."""
    h = hashlib.sha256(f"split:{query_id}".encode()).digest()
    return "dev" if h[0] % 2 == 0 else "test"


def scores_by_split(run_name: str, metric: str = "hit@1") -> dict:
    """Score one run separately on dev and test, with intervals."""
    runs = _run_dirs()
    if run_name not in runs:
        raise KeyError(f"no results for {run_name!r}")

    per_query = load_per_query(runs[run_name], metric)
    out: dict = {"run": run_name, "metric": metric}
    for split in ("dev", "test"):
        vals = [v for q, v in per_query.items() if split_of(q) == split]
        hits = int(sum(vals))
        lo, hi = wilson_interval(hits, len(vals))
        out[split] = {"n": len(vals), "score": round(hits / len(vals), 4) if vals else None,
                      "ci": (round(lo, 3), round(hi, 3))}
    d, t = out["dev"]["score"], out["test"]["score"]
    # A large gap means the ranking was partly fitted to dev.
    out["dev_minus_test"] = round(d - t, 4) if d is not None and t is not None else None
    return out


def compare_runs_split(name_a: str, name_b: str, split: str = "test",
                       metric: str = "hit@1") -> dict:
    """Paired comparison restricted to one split. Report the `test` result."""
    runs = _run_dirs()
    a = load_per_query(runs[name_a], metric)
    b = load_per_query(runs[name_b], metric)
    shared = [q for q in sorted(set(a) & set(b)) if split_of(q) == split]
    if not shared:
        raise ValueError(f"no shared queries in split {split!r}")

    a_wins = sum(1 for q in shared if a[q] > b[q])
    b_wins = sum(1 for q in shared if b[q] > a[q])
    p = mcnemar_exact(a_wins, b_wins)
    mean_a = sum(a[q] for q in shared) / len(shared)
    mean_b = sum(b[q] for q in shared) / len(shared)
    return {"a": name_a, "b": name_b, "split": split, "n": len(shared),
            "score_a": round(mean_a, 4), "score_b": round(mean_b, 4),
            "difference": round(mean_a - mean_b, 4),
            "a_wins": a_wins, "b_wins": b_wins,
            "p_value": round(p, 4), "significant": p < 0.05}
