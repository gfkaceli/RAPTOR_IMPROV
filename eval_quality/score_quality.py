"""
score_quality.py — Score QuALITY predictions.

Produces the three analysis cuts identified as most important for the thesis:
  1. HARD vs EASY accuracy breakdown (the primary claim: does hierarchy help
     more on questions requiring whole-passage understanding?)
  2. Retrieved-node layer distribution (the mechanism: when a method wins, is it
     because it retrieved summary nodes rather than leaves?)
  3. Overall accuracy with bootstrap 95% CIs (so differences can be judged
     against noise rather than reported as bare point estimates)

Usage:
    python -m eval_quality.score_quality experiments/quality/<timestamp>
    python -m eval_quality.score_quality --file predictions_leiden.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Dict, List

from .quality_metric import score_prediction, aggregate_scores, bootstrap_ci


def score_jsonl(path: str) -> Dict:
    per_question: List[Dict] = []
    detail_rows: List[Dict] = []
    layer_counter = Counter()
    layer_counter_hard = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = score_prediction(rec["predicted"], rec["gold_label"], rec["options"])
            row = {
                "correct": s["correct"],
                "parsed": s["parsed"],
                "difficult": rec.get("difficult", 0),
            }
            per_question.append(row)

            # Layer distribution over retrieved nodes
            for li in rec.get("retrieved_layers", []):
                layer_counter[li] += 1
                if rec.get("difficult", 0) == 1:
                    layer_counter_hard[li] += 1

            detail_rows.append({
                "method": rec.get("method", ""),
                "article_id": rec["article_id"],
                "question_id": rec["question_id"],
                "difficult": rec.get("difficult", 0),
                "gold_label": rec["gold_label"],
                "predicted_index": s["predicted_index"],
                "correct": s["correct"],
                "parsed": int(s["parsed"]),
                "predicted_raw": rec["predicted"][:80],
                "tree_layers": rec.get("tree_layers", 0),
            })

    summary = aggregate_scores(per_question)
    summary["file"] = os.path.basename(path)

    # Bootstrap CIs
    summary["ci_overall"] = bootstrap_ci(per_question, subset=None)
    summary["ci_hard"] = bootstrap_ci(per_question, subset="hard")
    summary["ci_easy"] = bootstrap_ci(per_question, subset="easy")

    # Layer distribution as fractions
    total_layers = sum(layer_counter.values())
    if total_layers:
        summary["layer_dist"] = {
            str(k): round(v / total_layers, 4) for k, v in sorted(layer_counter.items())
        }
    total_hard_layers = sum(layer_counter_hard.values())
    if total_hard_layers:
        summary["layer_dist_hard"] = {
            str(k): round(v / total_hard_layers, 4)
            for k, v in sorted(layer_counter_hard.items())
        }

    return {"summary": summary, "detail": detail_rows}


def write_detail_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(method, s):
    print(f"\n  {method}: accuracy={s['accuracy']:.4f} "
          f"[{s['ci_overall']['lo']:.3f}, {s['ci_overall']['hi']:.3f}]  "
          f"(n={s['n']}, parse_rate={s['parse_rate']:.3f})")
    if "accuracy_hard" in s:
        print(f"    HARD: {s['accuracy_hard']:.4f} "
              f"[{s['ci_hard']['lo']:.3f}, {s['ci_hard']['hi']:.3f}] (n={s['n_hard']})")
    if "accuracy_easy" in s:
        print(f"    EASY: {s['accuracy_easy']:.4f} "
              f"[{s['ci_easy']['lo']:.3f}, {s['ci_easy']['hi']:.3f}] (n={s['n_easy']})")
    if "layer_dist" in s:
        dist = ", ".join(f"L{k}={v:.2f}" for k, v in s["layer_dist"].items())
        print(f"    Retrieved layer dist (all):  {dist}")
    if "layer_dist_hard" in s:
        dist = ", ".join(f"L{k}={v:.2f}" for k, v in s["layer_dist_hard"].items())
        print(f"    Retrieved layer dist (HARD): {dist}")


def score_run_directory(run_dir: str) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(run_dir, "predictions_*.jsonl")))
    if not files:
        print(f"  No predictions_*.jsonl in {run_dir}")
        return []

    summaries = []
    for path in files:
        method = os.path.basename(path).replace("predictions_", "").replace(".jsonl", "")
        result = score_jsonl(path)
        result["summary"]["method"] = method
        summaries.append(result["summary"])
        write_detail_csv(result["detail"], os.path.join(run_dir, f"detailed_{method}.csv"))
        print_summary(method, result["summary"])

    # Save aggregate
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    _write_results_csv(summaries, os.path.join(run_dir, "results.csv"))
    return summaries


def _write_results_csv(summaries, path):
    if not summaries:
        return
    cols = ["method", "n", "accuracy", "accuracy_hard", "accuracy_easy",
            "parse_rate", "n_hard", "n_easy"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["ci_lo", "ci_hi", "ci_hard_lo", "ci_hard_hi"])
        for s in summaries:
            w.writerow(
                [s.get(c, "") for c in cols] +
                [s["ci_overall"]["lo"], s["ci_overall"]["hi"],
                 s.get("ci_hard", {}).get("lo", ""), s.get("ci_hard", {}).get("hi", "")]
            )


def main():
    p = argparse.ArgumentParser(description="Score QuALITY predictions.")
    p.add_argument("paths", nargs="*")
    p.add_argument("--file", default=None)
    args = p.parse_args()

    targets = list(args.paths) + ([args.file] if args.file else [])
    if not targets:
        print("ERROR: provide a run directory or --file <jsonl>")
        return

    all_summaries = []
    for t in targets:
        if os.path.isdir(t):
            print(f"\n=== {t} ===")
            all_summaries.extend(score_run_directory(t))
        elif os.path.isfile(t):
            method = os.path.basename(t).replace("predictions_", "").replace(".jsonl", "")
            r = score_jsonl(t)
            r["summary"]["method"] = method
            all_summaries.append(r["summary"])
            print_summary(method, r["summary"])

    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("  COMBINED COMPARISON (sorted by overall accuracy)")
        print("=" * 70)
        all_summaries.sort(key=lambda s: -s.get("accuracy", 0))
        print(f"  {'method':<16s} {'overall':>18s} {'HARD':>18s} {'EASY':>10s}")
        for s in all_summaries:
            ov = f"{s['accuracy']:.3f}[{s['ci_overall']['lo']:.2f},{s['ci_overall']['hi']:.2f}]"
            hd = (f"{s.get('accuracy_hard',0):.3f}"
                  f"[{s.get('ci_hard',{}).get('lo',0):.2f},{s.get('ci_hard',{}).get('hi',0):.2f}]")
            ez = f"{s.get('accuracy_easy',0):.3f}"
            print(f"  {s['method']:<16s} {ov:>18s} {hd:>18s} {ez:>10s}")

        print("\n  Interpretation guide:")
        print("  - If HARD-subset CIs for a tree method sit above flat's HARD CI,")
        print("    that is evidence hierarchy helps on whole-passage questions.")
        print("  - Overlapping CIs mean the difference is within noise at this n.")


if __name__ == "__main__":
    main()
