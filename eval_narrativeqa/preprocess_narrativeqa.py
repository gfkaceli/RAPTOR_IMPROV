"""
preprocess_narrativeqa.py — Download and preprocess NarrativeQA for RAPTOR evaluation.

NarrativeQA (Kocisky et al., 2018, TACL) contains questions over full books
(Project Gutenberg) and movie scripts. Documents average tens of thousands of
tokens — roughly an order of magnitude longer than QuALITY passages — which is
exactly the regime where recursive hierarchical indexing should build deep,
genuinely multi-layer trees. Answers are short free-form references (typically
under ten words), with TWO reference answers per question written by different
annotators; scoring takes the max over references.

Two properties matter for the thesis and are recorded here:
  1. Question provenance: annotators wrote questions from Wikipedia plot
     SUMMARIES, not the full stories. Questions therefore concentrate on
     plot-salient content — arguably the best case for summary nodes, and a
     caveat to state when interpreting a hierarchy win.
  2. Compute: full-story tree builds are expensive (60k+ tokens -> hundreds of
     leaf chunks). The run script supports tree caching; this preprocessor
     supports SEEDED RANDOM STORY SAMPLING so a subset run is reproducible and
     unbiased (never select "shortest stories" — that biases toward scripts).

Data source (HF hub, parquet-converted so datasets v4 works):
    load_dataset("deepmind/narrativeqa", split=...)
    One row per QA pair: {document:{id,kind,url,file_size,word_count,start,end,
    summary:{text,title,...}, text}, question:{text,...}, answers:[{text,..} x2]}
    Rows repeat the same document across its ~30 questions -> group by doc id.

Usage:
    python -m eval_narrativeqa.preprocess_narrativeqa --split validation --max-articles 5
    python -m eval_narrativeqa.preprocess_narrativeqa --split test --max-articles 40 \
        --max-doc-words 100000 --sample-seed 224

Output:
    data/narrativeqa/<split>.json — list of stories with cleaned full text and
    grouped questions (each with its list of reference answers).
"""

from __future__ import annotations

import argparse
import html as htmllib
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG.*", re.I | re.S)
HTML_TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)


def clean_story_text(text: str, kind: str, start: str = "", end: str = "") -> str:
    """
    Strip non-story boilerplate from a raw NarrativeQA document.

    Order of operations:
      1. Use the dataset's own start/end content markers when both are present
         and sane — these delimit the actual story inside the raw file.
      2. Gutenberg fallback: cut the license header/footer by their *** markers.
      3. HTML: unescape entities and strip tags (movie scripts are scraped HTML).
      4. Collapse repeated blank lines / whitespace.
    """
    if not text:
        return ""

    # 1. Dataset-provided content markers
    if start:
        i = text.find(start)
        if i >= 0:
            text = text[i:]
    if end:
        j = text.rfind(end)
        if j >= 0:
            text = text[: j + len(end)]

    # 2. Gutenberg header/footer fallback (harmless if markers already applied)
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]

    # 3. HTML (mostly movie scripts, but cheap to run always)
    if "<" in text:
        text = SCRIPT_STYLE.sub(" ", text)
        text = HTML_TAG.sub(" ", text)
        text = htmllib.unescape(text)

    # 4. Whitespace normalization (preserve paragraph breaks lightly)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------

def load_rows(split: str):
    """
    Load the QA-pair rows from HF. The repo was converted to parquet on main,
    so the standard path works under datasets v4; a direct parquet read is kept
    as a fallback for older/odd environments.
    """
    from datasets import load_dataset
    try:
        return load_dataset("deepmind/narrativeqa", split=split)
    except Exception as exc:
        print(f"  standard load failed ({exc}); trying direct parquet fallback")
        return load_dataset(
            "parquet",
            data_files=f"hf://datasets/deepmind/narrativeqa/data/{split}-*.parquet",
            split="train",
        )


def group_rows_into_stories(rows) -> List[Dict]:
    """
    Group one-row-per-question records into one record per story.
    The document text is taken from the first row seen for each id.
    """
    stories: Dict[str, Dict] = {}
    order: List[str] = []

    for row in rows:
        doc = row["document"]
        did = doc["id"]
        if did not in stories:
            summary = doc.get("summary") or {}
            cleaned = clean_story_text(
                doc.get("text") or "",
                kind=doc.get("kind", ""),
                start=doc.get("start") or "",
                end=doc.get("end") or "",
            )
            stories[did] = {
                "story_id": did,
                "kind": doc.get("kind", ""),
                "title": (summary.get("title") or "").strip(),
                "full_text": cleaned,
                "n_words": len(cleaned.split()),
                "n_words_raw": doc.get("word_count", 0),
                "questions": [],
            }
            order.append(did)

        q_text = (row["question"]["text"] or "").strip()
        answers = [
            (a.get("text") or "").strip()
            for a in (row.get("answers") or [])
            if (a.get("text") or "").strip()
        ]
        if not q_text or not answers:
            continue
        s = stories[did]
        s["questions"].append({
            "question_id": f"{did}_{len(s['questions'])}",
            "question": q_text,
            "answers": answers,   # typically two references; scoring maxes over them
        })

    return [stories[d] for d in order if stories[d]["questions"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Preprocess NarrativeQA for RAPTOR evaluation.")
    p.add_argument("--split", default="validation",
                   help="train / validation / test (validation: ~115 stories)")
    p.add_argument("--max-articles", type=int, default=None,
                   help="Number of stories to keep (seeded random sample).")
    p.add_argument("--sample-seed", type=int, default=224,
                   help="Seed for the random story sample (default 224).")
    p.add_argument("--min-doc-words", type=int, default=1000,
                   help="Skip stories with fewer cleaned words than this.")
    p.add_argument("--max-doc-words", type=int, default=None,
                   help="Skip stories longer than this many cleaned words "
                        "(compute cap; document the value in the thesis).")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print(f"Loading NarrativeQA [{args.split}] from HF hub...")
    rows = load_rows(args.split)
    print(f"  Loaded {len(rows)} QA rows.")

    stories = group_rows_into_stories(rows)
    print(f"  Grouped into {len(stories)} stories.")

    # Length filters BEFORE sampling, so the sample frame is well-defined.
    kept = []
    skipped_short = skipped_long = 0
    for s in stories:
        if s["n_words"] < args.min_doc_words:
            skipped_short += 1
            continue
        if args.max_doc_words and s["n_words"] > args.max_doc_words:
            skipped_long += 1
            continue
        kept.append(s)
    print(f"  Filters: -{skipped_short} too short, -{skipped_long} over word cap "
          f"-> {len(kept)} eligible stories")

    # Seeded RANDOM sample — not first-N, not shortest-N — for an unbiased,
    # reproducible subset. Record the seed in the thesis.
    if args.max_articles and args.max_articles < len(kept):
        rng = random.Random(args.sample_seed)
        kept = rng.sample(kept, args.max_articles)
        kept.sort(key=lambda s: s["story_id"])   # stable output order
        print(f"  Random sample of {len(kept)} stories (seed={args.sample_seed})")

    n_q = sum(len(s["questions"]) for s in kept)
    n_books = sum(1 for s in kept if s["kind"] == "gutenberg")
    n_movies = len(kept) - n_books
    words = [s["n_words"] for s in kept]
    if kept:
        print(f"  Final: {len(kept)} stories ({n_books} books, {n_movies} scripts), "
              f"{n_q} questions")
        print(f"  Cleaned length (words): min={min(words):,}, max={max(words):,}, "
              f"mean={sum(words)//len(words):,}")
        print(f"  Avg questions/story: {n_q/len(kept):.1f}")

    out_path = args.output or os.path.join("data", "narrativeqa", f"{args.split}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=1)
    print(f"\nSaved to {out_path}")

    if not kept:
        print("WARNING: no stories survived the filters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
