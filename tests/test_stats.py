"""Tests for uncertainty and significance.

These exist because the first analysis of this benchmark reported a 2.5pp
chunking "trend" that was sampling noise, and invented a mechanism for it.
The maths that would have caught it is now tested.
"""

from __future__ import annotations

import pytest

from fitment_rag.retrieval import Retriever
from fitment_rag.stats import mcnemar_exact, power_for_difference, wilson_interval


# --- Wilson intervals -----------------------------------------------------

def test_interval_brackets_the_estimate():
    lo, hi = wilson_interval(173, 200)      # hit@1 = 0.865
    assert lo < 0.865 < hi


def test_interval_never_leaves_zero_one():
    """The normal approximation goes above 1.0 near the boundary; Wilson must not."""
    for k, n in [(200, 200), (0, 200), (197, 200), (1, 200)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_interval_narrows_as_n_grows():
    """The reason n=200 could not answer the chunking question."""
    w200 = wilson_interval(170, 200)
    w2000 = wilson_interval(1700, 2000)
    assert (w2000[1] - w2000[0]) < (w200[1] - w200[0]) / 2


def test_empty_sample_is_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --- McNemar --------------------------------------------------------------

def test_the_chunking_result_is_not_significant():
    """whole_doc vs fixed512: 12 wins to 7. This is the number that was oversold."""
    assert mcnemar_exact(12, 7) > 0.05


def test_the_embedder_result_is_significant():
    """e5-small vs MiniLM: 65 wins to 6."""
    assert mcnemar_exact(65, 6) < 0.001


def test_symmetric_in_its_arguments():
    assert mcnemar_exact(12, 7) == mcnemar_exact(7, 12)


def test_no_disagreement_means_no_evidence():
    assert mcnemar_exact(0, 0) == 1.0


def test_pairing_beats_raw_difference():
    """Same 5-point gap, different reliability -- this is why pairing matters."""
    lopsided = mcnemar_exact(30, 25)   # both retrievers volatile
    clean = mcnemar_exact(5, 0)        # one strictly dominates
    assert clean < lopsided


# --- power ----------------------------------------------------------------

def test_n200_cannot_resolve_three_points():
    assert power_for_difference(0.85, 0.03, n=200) < 0.5


def test_n200_easily_resolves_thirty_points():
    assert power_for_difference(0.85, 0.30, n=200) > 0.9


# --- the top_k confound ---------------------------------------------------

def test_dedupe_returns_distinct_documents():
    """Five chunks from two documents must not count as five candidates."""
    ids = ["A::0", "A::1", "A::2", "B::0", "B::1", "C::0", "D::3"]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    got, got_s = Retriever._best_chunk_per_document(ids, scores, 5)
    assert [c.split("::")[0] for c in got] == ["A", "B", "C", "D"]
    assert got_s == [0.9, 0.6, 0.4, 0.3]      # best-ranked chunk per document


def test_dedupe_keeps_rank_order_and_respects_k():
    ids = [f"D{i}::0" for i in range(20)]
    got, _ = Retriever._best_chunk_per_document(ids, list(range(20, 0, -1)), 5)
    assert got == ["D0::0", "D1::0", "D2::0", "D3::0", "D4::0"]


def test_whole_doc_and_split_doc_yield_the_same_candidate_count():
    """The confound itself: before the fix these differed (5.00 vs 4.04)."""
    whole = ["A::0", "B::0", "C::0", "D::0", "E::0"]
    split = ["A::0", "A::1", "B::0", "B::1", "C::0", "C::1", "D::0", "E::0"]
    a, _ = Retriever._best_chunk_per_document(whole, [1.0] * len(whole), 5)
    b, _ = Retriever._best_chunk_per_document(split, [1.0] * len(split), 5)
    assert len({c.split("::")[0] for c in a}) == len({c.split("::")[0] for c in b}) == 5
