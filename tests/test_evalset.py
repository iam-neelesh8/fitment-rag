"""Tests for the eval-set generator.

These exist because the first version of the generator produced a benchmark that
could not measure anything: it quoted the product title verbatim, so every
question contained its own gold document's first line and every config scored a
perfect 1.0 -- BM25 included. That failure was invisible to the unit tests at
the time, so each rule that prevents it now has a test.
"""

from __future__ import annotations

import pytest

from fitment_rag.evalset.build import (
    _candidate_terms,
    _looks_like_part_number,
    _minimal_unique_phrase,
    build_token_index,
    build_eval_set,
    tokens,
)


@pytest.fixture
def corpus():
    def doc(i, title, store, price):
        return {
            "doc_id": f"D{i}",
            "title": title,
            "store": store,
            "price": price,
            "average_rating": 4.0 + i / 10,
            "categories": ["Automotive", "Brakes"],
            "text": f"Title: {title}\nBrand/Store: {store}\nPrice: {price}\n"
                    + ("filler text about the product " * 8),
        }

    return [
        doc(0, "Bosch QuietCast Ceramic Brake Pad BC905", "Bosch", 42.99),
        doc(1, "Wagner ThermoQuiet Ceramic Brake Pad QC465", "Wagner", 38.50),
        doc(2, "Denso Iridium Spark Plug SK20R11", "Denso", 12.25),
        doc(3, "Monroe Magnum Steering Damper SC2951", "Monroe", 55.00),
    ]


# --- part-number detection ------------------------------------------------

@pytest.mark.parametrize("tok", ["sc2951", "fl121", "b00j9i1bd4", "qc465", "12345"])
def test_part_numbers_are_detected(tok):
    assert _looks_like_part_number(tok)


@pytest.mark.parametrize("tok", ["brake", "ceramic", "bosch", "pad", "abc"])
def test_words_are_not_part_numbers(tok):
    assert not _looks_like_part_number(tok)


# --- rule 1: no answer leakage --------------------------------------------

def test_answer_terms_are_stripped_from_the_question():
    """'Who makes the Bosch brake pad?' would hand over the answer for free."""
    terms = _candidate_terms("Bosch QuietCast Ceramic Brake Pad BC905", "Bosch")
    assert "bosch" not in terms
    assert "ceramic" in terms


def test_stopwords_and_part_numbers_are_dropped():
    terms = _candidate_terms("New Universal Ceramic Brake Pad SC2951", "")
    assert "new" not in terms and "universal" not in terms   # generic filler
    assert "sc2951" not in terms                              # too easy a key
    assert "ceramic" in terms and "brake" in terms


# --- rule 2: unique by construction ---------------------------------------

def test_phrase_uniquely_identifies_its_gold_document(corpus):
    index = build_token_index(corpus)
    for i, doc in enumerate(corpus):
        terms = _candidate_terms(doc["title"], doc["store"])
        phrase = _minimal_unique_phrase(terms, index, i)
        if phrase is None:
            continue
        matches = set(index[phrase[0]])
        for t in phrase[1:]:
            matches &= index[t]
        assert matches == {i}, f"{phrase} matched {matches}, expected {{{i}}}"


def test_ambiguous_terms_are_rejected(corpus):
    """'ceramic brake pad' matches docs 0 and 1, so it must not be accepted."""
    index = build_token_index(corpus)
    assert _minimal_unique_phrase(["ceramic", "brake", "pad"], index, 0) is None


# --- rule 3: no verbatim title --------------------------------------------

def test_questions_do_not_reproduce_the_title(corpus, tmp_path):
    items = build_eval_set(corpus, tmp_path / "e.jsonl", n_questions=20, per_doc=4)
    assert items, "generator produced nothing"

    by_id = {d["doc_id"]: d for d in corpus}
    for item in items:
        gold = by_id[item["relevant_doc_ids"][0]]
        q, title = set(tokens(item["question"])), set(tokens(gold["title"]))
        assert not title.issubset(q), f"question repeats the whole title: {item['question']}"


def test_no_answer_appears_in_its_own_question(corpus, tmp_path):
    items = build_eval_set(corpus, tmp_path / "e.jsonl", n_questions=20, per_doc=4)
    for item in items:
        assert item["ground_truth"].lower() not in item["question"].lower()


def test_generation_is_deterministic(corpus, tmp_path):
    a = build_eval_set(corpus, tmp_path / "a.jsonl", n_questions=20, per_doc=4, seed=5)
    b = build_eval_set(corpus, tmp_path / "b.jsonl", n_questions=20, per_doc=4, seed=5)
    assert [x["question"] for x in a] == [x["question"] for x in b]
