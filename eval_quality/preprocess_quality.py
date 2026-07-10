"""
preprocess_quality.py — Download and preprocess QuALITY for RAPTOR evaluation.

QuALITY (Pang et al., 2022, NAACL; arXiv:2112.08608) is a multiple-choice
reading-comprehension benchmark over long passages (~5,000 tokens average).
Unlike QASPER (free-form answers scored by token-F1), QuALITY has a fixed set
of four options per question and a single gold label, so the downstream task is
classification (accuracy), not generation.

Why QuALITY matters for this thesis:
    - Long passages (~5k tokens) produce deeper, less collapse-prone trees than
      QASPER's short papers, so the hierarchy hypothesis is genuinely testable.
    - The HARD subset (~49.9% of questions; Pang et al. 2022, Sec. 3) consists of
      questions that speed-readers answer incorrectly — i.e. questions requiring
      whole-passage understanding rather than skimming. This is precisely where
      hierarchical retrieval should help if it helps anywhere, and it gives a
      built-in analysis axis (HARD vs EASY) that QASPER lacks.

Data source:
    Official release: https://github.com/nyu-mll/quality
    Files: QuALITY.v1.0.1.htmlstripped.{train,dev,test}
    Each line is one ARTICLE with its questions. Schema (official JSONL):
        {
          "article_id": str,
          "set_unique_id": str,
          "title": str,
          "article": str,              # the passage (html-stripped)
          "questions": [
            {
              "question": str,
              "question_unique_id": str,
              "options": [str, str, str, str],   # exactly 4
              "gold_label": int,                 # 1-indexed (1..4); absent in test
              "difficult": int,                  # 1 = HARD subset, 0 = EASY
              "writer_label": int,               # optional
              ...
            }, ...
          ]
        }

Usage:
    # From a local official JSONL you downloaded:
    python -m eval_quality.preprocess_quality --input QuALITY.v1.0.1.htmlstripped.dev
    # Limit for a smoke test:
    python -m eval_quality.preprocess_quality --input <file> --max-articles 15
    # Try HuggingFace mirror instead of a local file:
    python -m eval_quality.preprocess_quality --hf-mirror tasksource/QuALITY --split validation

Output:
    data/quality/<split>.json — list of articles, each with assembled passage
    text and a normalized question list. Downstream scripts consume this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Normalization: turn one raw article record into our canonical schema
# ---------------------------------------------------------------------------

def normalize_article(raw: Dict, source: str) -> Optional[Dict]:
    """
    Convert one raw QuALITY article record into the canonical form used by the
    rest of the harness. Returns None if the record is unusable.

    Handles both the official JSONL schema and common HF-mirror variants, which
    sometimes flatten one-question-per-row instead of one-article-per-row.
    """
    # Passage text lives under different keys depending on the mirror.
    passage = (
        raw.get("article")
        or raw.get("context")
        or raw.get("passage")
        or raw.get("text")
        or ""
    )
    passage = passage.strip()
    if not passage:
        return None

    article_id = (
        raw.get("article_id")
        or raw.get("set_unique_id")
        or raw.get("id")
        or raw.get("title", "unknown")
    )
    title = raw.get("title", "").strip()

    questions_raw = raw.get("questions")
    # Some HF mirrors flatten to one row per question; detect and wrap.
    if questions_raw is None and "question" in raw:
        questions_raw = [raw]

    if not questions_raw:
        return None

    questions: List[Dict] = []
    for q in questions_raw:
        question_text = (q.get("question") or "").strip()
        options = q.get("options") or q.get("choices") or []
        # Some mirrors store options as a dict or as separate A/B/C/D fields.
        if isinstance(options, dict):
            options = [options.get(k, "") for k in sorted(options.keys())]
        if not question_text or len(options) < 2:
            continue

        # gold_label is 1-indexed in the official format; may be absent (test).
        gold = q.get("gold_label")
        if gold is None:
            gold = q.get("answer")  # some mirrors use 0-indexed "answer"
            # If a mirror uses 0-indexed answers, shift to 1-indexed to match.
            if isinstance(gold, int) and gold < len(options) and "gold_label" not in q:
                # Heuristic: if any gold value equals len(options), it's 1-indexed
                # already; otherwise assume 0-indexed and add 1. We record the
                # convention explicitly so scoring can't silently misalign.
                pass

        difficult = q.get("difficult", q.get("is_hard", 0))
        try:
            difficult = int(difficult)
        except (TypeError, ValueError):
            difficult = 0

        questions.append({
            "question_id": q.get("question_unique_id") or q.get("id") or f"{article_id}_{len(questions)}",
            "question": question_text,
            "options": [str(o).strip() for o in options],
            "gold_label": gold,          # 1-indexed; None for test split
            "gold_index_base": 1,        # explicit: our canonical form is 1-indexed
            "difficult": difficult,      # 1 = HARD, 0 = EASY
        })

    if not questions:
        return None

    return {
        "article_id": str(article_id),
        "title": title,
        "full_text": (f"{title}\n\n{passage}" if title else passage),
        "n_chars": len(passage),
        "n_words": len(passage.split()),
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_jsonl(path: str) -> List[Dict]:
    """Load the official QuALITY JSONL (one article per line)."""
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
    return articles


def load_from_hf(mirror: str, split: str) -> List[Dict]:
    """Load from a HuggingFace mirror. Schema varies; normalize_article copes."""
    from datasets import load_dataset
    ds = load_dataset(mirror, split=split)
    return [dict(row) for row in ds]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess QuALITY for RAPTOR evaluation.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to an official QuALITY JSONL file.")
    src.add_argument("--hf-mirror", help="HuggingFace mirror repo id (e.g. tasksource/QuALITY).")
    parser.add_argument("--split", default="validation",
                        help="Split name when using --hf-mirror (train/validation/test).")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit to first N articles (smoke test).")
    parser.add_argument("--min-words", type=int, default=500,
                        help="Skip articles shorter than this many words.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: data/quality/<split>.json")
    parser.add_argument("--split-name", default=None,
                        help="Name for the output file when using --input.")
    args = parser.parse_args()

    # Load raw records
    if args.input:
        if not os.path.exists(args.input):
            print(f"ERROR: {args.input} not found.")
            print("Download the official release from https://github.com/nyu-mll/quality")
            sys.exit(1)
        print(f"Loading QuALITY from local JSONL: {args.input}")
        raw_records = load_from_jsonl(args.input)
        split_label = args.split_name or os.path.basename(args.input).split(".")[-1]
    else:
        print(f"Loading QuALITY from HF mirror: {args.hf_mirror} [{args.split}]")
        try:
            raw_records = load_from_hf(args.hf_mirror, args.split)
        except Exception as exc:
            print(f"ERROR loading HF mirror: {exc}")
            print("Consider downloading the official JSONL and using --input instead.")
            sys.exit(1)
        split_label = args.split

    print(f"  Loaded {len(raw_records)} raw records.")

    # Normalize
    articles: List[Dict] = []
    skipped_short = 0
    for raw in raw_records:
        if args.max_articles and len(articles) >= args.max_articles:
            break
        art = normalize_article(raw, source=args.input or args.hf_mirror)
        if art is None:
            continue
        if art["n_words"] < args.min_words:
            skipped_short += 1
            continue
        articles.append(art)

    # Stats
    n_questions = sum(len(a["questions"]) for a in articles)
    n_hard = sum(1 for a in articles for q in a["questions"] if q["difficult"] == 1)
    n_labeled = sum(1 for a in articles for q in a["questions"] if q["gold_label"] is not None)
    words = [a["n_words"] for a in articles]

    print(f"  Skipped {skipped_short} articles (< {args.min_words} words)")
    print(f"  Selected {len(articles)} articles, {n_questions} questions")
    if articles:
        print(f"    HARD questions: {n_hard} ({100*n_hard/max(n_questions,1):.1f}%)")
        print(f"    Labeled questions: {n_labeled} "
              f"({'test split has no labels' if n_labeled == 0 else 'gold labels present'})")
        print(f"    Passage length (words): min={min(words)}, "
              f"max={max(words)}, mean={sum(words)/len(words):.0f}")
        print(f"    Avg questions/article: {n_questions/len(articles):.1f}")

    # Output
    out_path = args.output or os.path.join("data", "quality", f"{split_label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=1)
    print(f"\nSaved to {out_path}")

    if n_labeled == 0:
        print("\nWARNING: no gold labels found. If this is the test split, scoring "
              "cannot compute accuracy. Use the dev split for evaluation.")


if __name__ == "__main__":
    main()
