"""Uncertainty and significance for retrieval metrics.

This module exists because the first pass at this benchmark reported a 2.5pp
"trend" across chunking strategies and invented a mechanism to explain it. At
n=200 the 95% interval on hit@1 is about +/-5pp and every one of those
comparisons had p > 0.3. The differences were sampling noise.

So uncertainty is no longer optional here. `leaderboard()` carries a CI on every
score, and `compare_runs()` runs the paired test that says whether a gap is real.

Two choices worth knowing:

* **Wilson intervals, not normal approximation.** hit@1 is a proportion, and
  scores here run to 0.985 where the normal approximation gives intervals that
  cross 1.0.

* **Paired exact McNemar, not a two-sample test.** Every config answers the same
  questions, so the comparison should only look at the queries where the two
  disagree. Ignoring the pairing throws away most of the power -- e5 vs MiniLM
  is 65 discordant to 6, which an unpaired test would badly understate.
"""

from __future__ import annotations

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

    NOTE ON MULTIPLE COMPARISONS: k configs give k(k-1)/2 tests, so at alpha
    0.05 roughly one in twenty looks significant by chance alone. The
    `bonferroni_significant` column applies the conservative correction. Report
    that column, not `significant`, when scanning a whole matrix for a winner.
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
    """Rough power to detect `delta` at sample size `n`, for planning only.

    Uses a normal approximation to the paired sign test. Good enough to answer
    "is n=200 enough for a 3pp difference?" -- it is not (about 0.2). It is not
    a substitute for reporting the actual p-value.
    """
    if delta <= 0 or n <= 0:
        return 0.0
    n_disc = max(1.0, n * discordance)
    p_shift = 0.5 + delta / (2 * discordance)
    p_shift = min(0.999, p_shift)
    se = math.sqrt(0.25 / n_disc)
    z = (p_shift - 0.5) / se - 1.96
    return max(0.0, min(1.0, 0.5 * (1 + math.erf(z / math.sqrt(2)))))
