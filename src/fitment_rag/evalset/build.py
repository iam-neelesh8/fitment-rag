"""Build the eval set: questions + gold document + ground-truth answer.

This file is the credibility core of the project. The corpus is downloaded, not
redistributed -- but the eval set is *our* artifact and it ships in the repo, so
anyone can inspect exactly what the numbers were scored against.

Ground truth is **extracted, not invented**: answers come from structured
metadata fields that are almost always present (`store`, `price`,
`average_rating`, `categories`). Hallucination risk is zero and the gold
document is known by construction.

Three rules keep the questions honest, and the first version of this file got
the third one wrong:

1. NO ANSWER LEAKAGE. Terms that appear in the answer are stripped from the
   question. Otherwise "Who makes the Caltric ignition coil?" hands over the
   answer "Caltric" for free.

2. UNIQUE BY CONSTRUCTION. Each question keeps adding title terms until exactly
   one document in the corpus contains all of them. Without this, a question
   could legitimately describe several products and the "gold" document would be
   arbitrary -- silently punishing a retriever that found an equally valid match.

3. NO VERBATIM TITLE. The original version quoted the full product title, so the
   question contained the document's own first line word for word. That is
   string lookup, not retrieval: every config scored a perfect 1.0, BM25
   included, and the benchmark could not distinguish anything. Questions now use
   a MINIMAL set of descriptive title terms -- enough to be unambiguous, not
   enough to be a copy.

Part-number-like tokens are also excluded from question terms. They are perfect
lexical keys, so including them recreates the same trivial-match problem.

LLM-written questions are a Phase 4 item; see context/02-plan.md.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

_WORD = re.compile(r"[a-z0-9]+")

# Generic retail/auto filler that identifies nothing on its own.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "with", "to", "in", "on", "by",
    "is", "are", "was", "be", "this", "that", "it", "its", "as", "at", "from",
    "new", "pack", "set", "piece", "pieces", "pcs", "kit", "pair", "compatible",
    "fit", "fits", "fitment", "replacement", "premium", "quality", "universal",
    "car", "auto", "automotive", "vehicle", "truck", "oem", "aftermarket",
    "inch", "inches", "size", "black", "silver", "chrome", "red", "blue",
}

MIN_TERMS = 3   # never fewer -- a 1-2 word question is usually ambiguous
MAX_TERMS = 7   # give up rather than rebuild the whole title

QUESTION_TEMPLATES = [
    ("store", "Who makes the {phrase}?"),
    ("price", "How much does the {phrase} cost?"),
    ("average_rating", "What rating do customers give the {phrase}?"),
    ("categories", "What kind of product is the {phrase}?"),
]


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _looks_like_part_number(tok: str) -> bool:
    """SC2951, FL121, B00J9I1BD4 -- perfect lexical keys, too easy to match on."""
    if len(tok) < 4:
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_alpha = any(c.isalpha() for c in tok)
    return has_digit and (has_alpha or len(tok) >= 5)


def _candidate_terms(title: str, answer: str) -> list[str]:
    """Descriptive title words, minus stopwords, part numbers, and the answer."""
    banned = set(tokens(answer))
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens(title):
        if (t in STOPWORDS or t in banned or t in seen
                or len(t) < 3 or _looks_like_part_number(t)):
            continue
        seen.add(t)
        out.append(t)
    return out


def build_token_index(docs: list[dict[str, Any]]) -> dict[str, set[int]]:
    """token -> set of document positions. Used only for the uniqueness check."""
    index: dict[str, set[int]] = defaultdict(set)
    for i, doc in enumerate(docs):
        for t in set(tokens(doc["text"])):
            index[t].add(i)
    return index


def _minimal_unique_phrase(
    terms: list[str], index: dict[str, set[int]], gold: int
) -> list[str] | None:
    """Shortest prefix of `terms` matching the gold document and nothing else."""
    if len(terms) < MIN_TERMS:
        return None
    for k in range(MIN_TERMS, min(MAX_TERMS, len(terms)) + 1):
        chosen = terms[:k]
        matches = set(index.get(chosen[0], set()))
        for t in chosen[1:]:
            matches &= index.get(t, set())
            if not matches:
                break
        if matches == {gold}:
            return chosen
    return None


def _items(
    docs: list[dict[str, Any]], order: list[int], per_doc: int, rng: random.Random
) -> Iterator[dict]:
    index = build_token_index(docs)

    for i in order:
        doc = docs[i]
        fields = QUESTION_TEMPLATES[:]
        rng.shuffle(fields)

        made = 0
        for field, template in fields:
            if made >= per_doc:
                break

            value = doc.get(field)
            if isinstance(value, list):
                value = value[-1] if value else ""
            value = str(value or "").strip()
            if not value or value.lower() in {"none", "nan", "null", ""}:
                continue

            terms = _candidate_terms(doc.get("title", ""), value)
            phrase = _minimal_unique_phrase(terms, index, i)
            if phrase is None:
                continue

            yield {
                "question": template.format(phrase=" ".join(phrase)),
                "ground_truth": value,
                "relevant_doc_ids": [doc["doc_id"]],
                "source_field": field,
                "n_terms": len(phrase),
                "generator": "template",
            }
            made += 1


def build_eval_set(
    docs: list[dict[str, Any]],
    out_path: Path,
    *,
    n_questions: int = 200,
    per_doc: int = 1,
    generator: str = "template",
    seed: int = 17,
) -> list[dict]:
    if generator != "template":
        raise ValueError(
            f"unknown generator: {generator!r}. Phase 1 ships the deterministic "
            "template generator only; LLM-written questions are Phase 4."
        )

    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)

    items: list[dict] = []
    for item in _items(docs, order, per_doc, rng):
        item["query_id"] = f"q{len(items):05d}"
        items.append(item)
        if len(items) >= n_questions:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return items


def load_eval_set(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
