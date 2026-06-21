"""
score_qasper.py — Compute QASPER Answer F1 from predictions.

Loads predictions JSONL files from `run_qasper_eval.py`, applies the official
QASPER Answer F1 metric (max-F1 over annotator references), and outputs:
    - results.json: per-method aggregate metrics
    - results.csv: tabular summary for the thesis writeup
    - detailed_<method>.csv: per-question scores

Usage:
    # Score all methods in a run
    python -m eval_qasper.score_qasper experiments/qasper/<timestamp>

    # Score a specific predictions file
    python -m eval_qasper.score_qasper --file predictions_leiden.jsonl

    # Compare multiple runs
    python -m eval_qasper.score_qasper run1/ run2/ run3/

Output (one row per method):
    method, n, f1, f1_extractive, f1_abstractive, f1_yes_no, f1_unanswerable,
    n_extractive, n_abstractive, n_yes_no, n_unanswerable
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List

from .qasper_metric import score_prediction, aggregate_scores


def score_jsonl(path: str) -> Dict:
    """Score one predictions JSONL file."""
    per_question: List[Dict] = []
    detailed_rows: List[Dict] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = score_prediction(rec["predicted"], rec["answers"])
            per_question.append(scores)
            detailed_rows.append({
                "method": rec.get("method", ""),
                "paper_id": rec["paper_id"],
                "question_id": rec["question_id"],
                "question": rec["question"][:120],
                "predicted": rec["predicted"][:200],
                "f1": round(scores["f1"], 4),
                "rouge_l": round(scores.get("rouge_l", 0.0), 4),
                "bleu": round(scores.get("bleu", 0.0), 4),
                "answer_type": scores["answer_type"],
                "n_references": scores["n_references"],
                "tree_layers": rec.get("tree_layers", 0),
                "tree_nodes": rec.get("tree_nodes", 0),
                "build_time_sec": rec.get("build_time_sec", 0),
            })

    summary = aggregate_scores(per_question)
    summary["file"] = os.path.basename(path)
    return {"summary": summary, "detail": detailed_rows}


def write_detail_csv(rows: List[Dict], out_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(summaries: List[Dict], out_path: str):
    """One row per method. Columns are union of all summary keys."""
    if not summaries:
        return
    all_keys = []
    seen = set()
    for s in summaries:
        for k in s:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    # Prefer a sensible column order
    preferred = ["method", "file", "n",
                 "f1", "rouge_l", "bleu",
                 "f1_extractive", "rouge_l_extractive", "bleu_extractive", "n_extractive",
                 "f1_abstractive", "rouge_l_abstractive", "bleu_abstractive", "n_abstractive",
                 "f1_yes_no", "rouge_l_yes_no", "bleu_yes_no", "n_yes_no",
                 "f1_unanswerable", "rouge_l_unanswerable", "bleu_unanswerable", "n_unanswerable",
                 "f1_none", "n_none"]
    ordered = [k for k in preferred if k in seen]
    extras = [k for k in all_keys if k not in seen.intersection(preferred)]
    fieldnames = ordered + [k for k in extras if k not in ordered]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in fieldnames}
            w.writerows([row])


def score_run_directory(run_dir: str) -> List[Dict]:
    """Find all predictions_*.jsonl files in a run directory and score each."""
    pattern = os.path.join(run_dir, "predictions_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No predictions_*.jsonl files found in {run_dir}")
        return []

    summaries = []
    for jsonl_path in files:
        # Extract method from filename: predictions_<method>.jsonl
        method = os.path.basename(jsonl_path).replace("predictions_", "").replace(".jsonl", "")
        print(f"\n  Scoring {method}...")

        result = score_jsonl(jsonl_path)
        result["summary"]["method"] = method
        summaries.append(result["summary"])

        # Write per-method detail CSV next to the predictions file
        detail_path = os.path.join(run_dir, f"detailed_{method}.csv")
        write_detail_csv(result["detail"], detail_path)
        s = result["summary"]
        print(f"    F1={s['f1']:.4f}  ROUGE-L={s.get('rouge_l', 0):.4f}  "
              f"BLEU={s.get('bleu', 0):.4f}  (n={s['n']})")
        # Per-type breakdown
        for atype in ["extractive", "abstractive", "yes_no", "unanswerable"]:
            key = f"f1_{atype}"
            if key in s:
                print(f"      {atype:14s}: F1={s[key]:.4f}  "
                      f"ROUGE-L={s.get(f'rouge_l_{atype}', 0):.4f}  "
                      f"BLEU={s.get(f'bleu_{atype}', 0):.4f}  "
                      f"(n={s[f'n_{atype}']})")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Score QASPER predictions and aggregate F1 metrics."
    )
    parser.add_argument("paths", nargs="*",
                        help="Run directories or individual JSONL files to score.")
    parser.add_argument("--file", default=None,
                        help="Score a single predictions JSONL file.")
    args = parser.parse_args()

    targets = list(args.paths)
    if args.file:
        targets.append(args.file)

    if not targets:
        print("ERROR: provide one or more run directories or --file <jsonl>")
        sys.exit(1)

    all_summaries: List[Dict] = []
    for target in targets:
        if os.path.isdir(target):
            print(f"\n=== {target} ===")
            run_summaries = score_run_directory(target)
            all_summaries.extend(run_summaries)

            # Save aggregate summary in this run directory
            if run_summaries:
                summary_json = os.path.join(target, "results.json")
                with open(summary_json, "w") as f:
                    json.dump(run_summaries, f, indent=2)

                summary_csv = os.path.join(target, "results.csv")
                write_summary_csv(run_summaries, summary_csv)

                print(f"\n  Saved: {summary_json}")
                print(f"  Saved: {summary_csv}")

        elif os.path.isfile(target) and target.endswith(".jsonl"):
            print(f"\n=== {target} ===")
            method = os.path.basename(target).replace("predictions_", "").replace(".jsonl", "")
            result = score_jsonl(target)
            result["summary"]["method"] = method
            all_summaries.append(result["summary"])

            print(f"  F1={result['summary']['f1']:.4f}  "
                  f"ROUGE-L={result['summary'].get('rouge_l', 0):.4f}  "
                  f"BLEU={result['summary'].get('bleu', 0):.4f}  "
                  f"(n={result['summary']['n']})")
            for atype in ["extractive", "abstractive", "yes_no", "unanswerable"]:
                key = f"f1_{atype}"
                if key in result["summary"]:
                    print(f"    {atype}: F1={result['summary'][key]:.4f} "
                          f"ROUGE-L={result['summary'].get(f'rouge_l_{atype}', 0):.4f} "
                          f"BLEU={result['summary'].get(f'bleu_{atype}', 0):.4f} "
                          f"(n={result['summary'][f'n_{atype}']})")
        else:
            print(f"  Skipping unknown target: {target}")

    # If we scored multiple targets, print a combined comparison
    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("  COMBINED COMPARISON")
        print("=" * 70)
        # Sort by F1 desc
        all_summaries.sort(key=lambda s: -s.get("f1", 0))
        print(f"  {'method':<20s} {'n':>5s} {'F1':>8s} {'ROUGE-L':>8s} {'BLEU':>8s}")
        for s in all_summaries:
            print(f"  {s.get('method', ''):<20s} "
                  f"{s.get('n', 0):>5d} {s.get('f1', 0):>8.4f} "
                  f"{s.get('rouge_l', 0):>8.4f} "
                  f"{s.get('bleu', 0):>8.4f}")


if __name__ == "__main__":
    main()