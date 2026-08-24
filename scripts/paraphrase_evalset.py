"""Build a paraphrased eval set to measure how much question wording favours BM25.

WHY THIS EXISTS
---------------
The template eval set picks question terms until they appear in exactly one
document -- uniqueness by lexical intersection. That is precisely what BM25
optimises for, so BM25's margin over dense retrieval is inflated by the
benchmark's own construction.

Rather than confess that in a footnote, measure it. This script rewrites each
question to carry the same meaning with different words, keeping the gold
document and the ground-truth answer unchanged. Scoring every config against
both sets turns "my eval is biased" into "eval wording was worth N points",
which is a result rather than an apology.

HOW TO READ THE OUTPUT
----------------------
Compare each retriever's drop between v1 and v2, not its absolute score:

  * BM25 falls much more than dense  ->  the original margin was wording, and
    the paraphrased set is the fairer comparison.
  * both fall by a similar amount    ->  the paraphrases lost identifying
    detail and became ambiguous. That is a broken eval, not a fair one --
    check `overlap_after` and re-run with a stricter prompt.
  * neither falls much               ->  the bias was small and the original
    result stands.

The LLM is an AUTHORING tool, used once. It is not part of the eval loop: the
output is a plain JSONL file committed to the repo, human-readable and
inspectable. No model runs at scoring time.

USAGE
-----
    ollama serve
    ollama pull qwen2.5:1.5b-instruct
    poetry run python scripts/paraphrase_evalset.py

Roughly 200 short generations; a few minutes on CPU. Run it when nothing else
is using the machine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitment_rag.config import DataConfig  # noqa: E402
from fitment_rag.data.amazon import load_documents  # noqa: E402
from fitment_rag.evalset.build import STOPWORDS, tokens  # noqa: E402

HOST = "http://localhost:11434"
MODEL = "qwen2.5:1.5b-instruct"

PROMPT = """Rewrite this product search question so it means the same thing but uses \
different words.

Question: {question}

Rules:
- Keep the meaning identical. A shopper must be able to tell it is the same product.
- Replace the descriptive words with synonyms or plain-English equivalents.
- Keep brand names and numbers exactly as they are.
- Keep it one short question.

Reply with ONLY the rewritten question, nothing else."""


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 2}


def overlap(question: str, doc_text: str) -> float:
    """Fraction of the question's content words that appear in the document.

    This is the number the whole exercise turns on -- it is a direct proxy for
    how much of BM25's job is already done by the question's wording.
    """
    q = content_tokens(question)
    if not q:
        return 0.0
    return len(q & set(tokens(doc_text))) / len(q)


def ollama_available() -> bool:
    try:
        return requests.get(f"{HOST}/api/tags", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def installed_models() -> list[str]:
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def paraphrase(question: str, model: str) -> str | None:
    payload = {
        "model": model,
        "prompt": PROMPT.format(question=question),
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 60},
    }
    try:
        r = requests.post(f"{HOST}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
    except requests.RequestException:
        return None

    text = r.json().get("response", "").strip()
    text = text.split("\n")[0].strip().strip('"').strip()
    text = re.sub(r"^(rewritten question|question)\s*:\s*", "", text, flags=re.I)
    if len(text) < 12 or len(text) > 220:
        return None
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="evalsets/amazon_automotive_10k.jsonl")
    ap.add_argument("--outfile", default="evalsets/amazon_automotive_10k_paraphrased.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if not ollama_available():
        print(f"Ollama not reachable at {HOST}. Start it with: ollama serve")
        return 1
    if args.model not in installed_models():
        print(f"Model not pulled. Run:  ollama pull {args.model}")
        print("installed:", installed_models() or "none")
        return 1

    src = ROOT / args.infile
    items = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} questions from {src.name}")
    print("loading corpus for the overlap measurement...")
    docs = {d["doc_id"]: d["text"] for d in load_documents(DataConfig(n_docs=10000, stride=4))}

    out, unchanged = [], 0
    before, after = [], []

    for i, item in enumerate(items, 1):
        gold = docs.get(item["relevant_doc_ids"][0], "")
        o_before = overlap(item["question"], gold)

        new_q = paraphrase(item["question"], args.model)
        if new_q is None:
            new_q, unchanged = item["question"], unchanged + 1

        o_after = overlap(new_q, gold)
        before.append(o_before)
        after.append(o_after)

        out.append({**item,
                    "question": new_q,
                    "original_question": item["question"],
                    "overlap_before": round(o_before, 3),
                    "overlap_after": round(o_after, 3),
                    "generator": "template+paraphrase"})

        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}   mean overlap {sum(before)/len(before):.3f} "
                  f"-> {sum(after)/len(after):.3f}")

    dst = ROOT / args.outfile
    with open(dst, "w", encoding="utf-8") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    mb, ma = sum(before) / len(before), sum(after) / len(after)
    print(f"\nwrote {len(out)} questions -> {dst}")
    print(f"lexical overlap with gold document: {mb:.3f} -> {ma:.3f}  "
          f"({(ma - mb) / mb * 100:+.1f}%)")
    if unchanged:
        print(f"{unchanged} kept verbatim (paraphrase failed or was rejected)")

    print("\n--- sample ---")
    for item in out[:5]:
        print(f"  before: {item['original_question']}")
        print(f"  after : {item['question']}")
        print(f"          overlap {item['overlap_before']} -> {item['overlap_after']}\n")

    if ma >= mb:
        print("WARNING: overlap did not drop. The paraphrases are not lexically different "
              "enough to test anything -- tighten the prompt before drawing conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
