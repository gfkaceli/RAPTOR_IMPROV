"""
preprocess_qasper.py — Download and preprocess QASPER for RAPTOR evaluation.

Loads the QASPER validation set from HuggingFace (`allenai/qasper`), groups
each paper's full text into a single document, extracts all questions and
references per paper, and saves the result as a JSON file that downstream
scripts (`run_qasper_eval.py`) consume.

Following the Laitenberger pattern:
    data_source/<dataset>/preprocess_<dataset>.py → preprocessed JSON

Why preprocess?
    Loading the full HuggingFace dataset every run is slow and pulls a lot of
    unused fields. The preprocessing collapses each paper into:
        - paper_id, title, abstract
        - full_text (concatenated sections — what we feed to RAPTOR)
        - questions: list of {question_id, question, answers (raw)}
    Downstream scripts can then iterate per-paper without re-parsing.

Usage:
    python -m eval_qasper.preprocess_qasper
    python -m eval_qasper.preprocess_qasper --split validation --max-papers 20
    python -m eval_qasper.preprocess_qasper --split test --output data/qasper/test.json

Requirements:
    pip install datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def assemble_full_text(paper: Dict) -> str:
    """
    Concatenate a QASPER paper's title, abstract, and full text into a single
    string for indexing by RAPTOR.

    QASPER full_text schema (from HuggingFace):
        full_text = {
            "section_name": ["Introduction", "Methods", ...],
            "paragraphs": [["section1_p1", "section1_p2", ...], ...]
        }
    """
    parts: List[str] = []
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()

    if title:
        parts.append(title)
    if abstract:
        parts.append("Abstract")
        parts.append(abstract)

    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", []) or []
    paragraphs_lists = full_text.get("paragraphs", []) or []

    for sname, plist in zip(section_names, paragraphs_lists):
        sname = (sname or "").strip()
        if sname:
            parts.append(sname)
        # plist is a list of paragraph strings for this section
        for p in (plist or []):
            p = (p or "").strip()
            if p:
                parts.append(p)

    return "\n\n".join(parts).strip()


def extract_questions(paper: Dict) -> List[Dict]:
    """
    Extract all questions for one paper, preserving annotator references.

    QASPER qas schema:
        qas = {
            "question": [...],
            "question_id": [...],
            "answers": [
                {"annotation_id": [...], "answer": [<answer_dict>, ...]},
                ...
            ],
            ...
        }
    """
    qas = paper.get("qas", {})
    questions = qas.get("question", []) or []
    question_ids = qas.get("question_id", []) or []
    answers_lists = qas.get("answers", []) or []

    out: List[Dict] = []
    for i, q in enumerate(questions):
        qid = question_ids[i] if i < len(question_ids) else f"q{i}"
        ans_block = answers_lists[i] if i < len(answers_lists) else {}
        # ans_block is {"annotation_id": [...], "answer": [<dict>, <dict>, ...]}
        annotator_answers = ans_block.get("answer", []) or []
        out.append({
            "question_id": qid,
            "question": q,
            "answers": annotator_answers,
        })
    return out


# ---------------------------------------------------------------------------
# QASPER loader — handles both old (script-based) and new (parquet) datasets
# library versions.
# ---------------------------------------------------------------------------

def _load_qasper(split: str):
    """
    Load the QASPER dataset, falling back to direct parquet loading if the
    datasets library has removed script support (>=4.0).

    The QASPER repo has a `qasper.py` loading script that newer versions of
    `datasets` refuse to run. The dataset also ships parquet files under
    `qasper/<split>-*.parquet` which we can load directly.
    """
    from datasets import load_dataset

    # Strategy 1: try the standard loader (works on datasets < 4.0)
    try:
        return load_dataset("allenai/qasper", split=split)
    except Exception as exc1:
        msg = str(exc1)
        if "Dataset scripts are no longer supported" not in msg \
                and "trust_remote_code" not in msg \
                and "no longer supported" not in msg:
            raise  # different error, don't suppress

        print(f"  Old script loader rejected — falling back to parquet loader.")

    # Strategy 2: direct parquet load via hf:// URI
    # The QASPER repo layout is: qasper/<split>-00000-of-00001.parquet
    # We use glob to be robust to multiple parquet shards.
    parquet_uri = f"hf://datasets/allenai/qasper/qasper/{split}-*.parquet"
    try:
        return load_dataset("parquet", data_files=parquet_uri, split="train")
    except Exception as exc2:
        print(f"  Direct parquet load failed: {exc2}")

    # Strategy 3: try the auto-converted parquet branch (refs/convert/parquet)
    # HuggingFace mirrors every dataset as parquet at this revision.
    parquet_uri_ref = (
        f"hf://datasets/allenai/qasper@refs/convert/parquet/qasper/{split}/*.parquet"
    )
    try:
        return load_dataset("parquet", data_files=parquet_uri_ref, split="train")
    except Exception as exc3:
        print(f"  Auto-converted parquet load failed: {exc3}")
        raise RuntimeError(
            "Could not load QASPER through any strategy. Either downgrade to "
            "`pip install datasets<4.0` or check that the parquet files are "
            "available at https://huggingface.co/datasets/allenai/qasper"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess QASPER for RAPTOR evaluation.")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"],
                        help="Which QASPER split to preprocess.")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Limit to first N papers (for fast iteration).")
    parser.add_argument("--min-questions", type=int, default=1,
                        help="Skip papers with fewer than N questions.")
    parser.add_argument("--min-text-chars", type=int, default=2000,
                        help="Skip papers with shorter concatenated text.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Default: data/qasper/<split>.json")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not found. Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading QASPER {args.split} split from HuggingFace...")
    ds = _load_qasper(args.split)
    print(f"  Loaded {len(ds)} papers.")

    processed: List[Dict] = []
    skipped_short = 0
    skipped_no_q = 0

    for i, paper in enumerate(ds):
        if args.max_papers and len(processed) >= args.max_papers:
            break

        full_text = assemble_full_text(paper)
        if len(full_text) < args.min_text_chars:
            skipped_short += 1
            continue

        questions = extract_questions(paper)
        if len(questions) < args.min_questions:
            skipped_no_q += 1
            continue

        processed.append({
            "paper_id": paper.get("id", f"paper_{i}"),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "full_text": full_text,
            "n_chars": len(full_text),
            "n_words": len(full_text.split()),
            "questions": questions,
        })

    print(f"  Skipped {skipped_short} papers (text < {args.min_text_chars} chars)")
    print(f"  Skipped {skipped_no_q} papers (< {args.min_questions} questions)")
    print(f"  Selected {len(processed)} papers")

    if processed:
        n_questions = sum(len(p["questions"]) for p in processed)
        n_words = sum(p["n_words"] for p in processed)
        print(f"  Total: {n_questions} questions, {n_words:,} words")
        print(f"  Avg per paper: {n_questions / len(processed):.1f} questions, "
              f"{n_words / len(processed):,.0f} words")

    # Output
    out_path = args.output or os.path.join("data", "qasper", f"{args.split}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=1)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()