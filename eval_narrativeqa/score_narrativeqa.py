"""
score_narrativeqa.py — Score NarrativeQA predictions.

Reports, per method:
  - Token F1, ROUGE-L (headline), and BLEU, each max-over-references
  - Bootstrap 95% CIs for F1 and ROUGE-L (2000 resamples, seed 224)
  - A books-vs-movie-scripts breakdown (the two document kinds differ in
    length and structure, so a method can win on one and not the other)
  - Retrieved-layer distribution when the runner captured it

Usage:
    python -m eval_narrativeqa.score_narrativeqa experiments/narrativeqa/<timestamp>
    python -m eval_narrativeqa.score_narrativeqa --file predictions_leiden.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Dict, List

from .narrativeqa_metric import score_prediction, aggregate_scores, bootstrap_ci


def score_jsonl(path: str) -> Dict:
    per_question: List[Dict] = []
    detail_rows: List[Dict] = []
    layer_counter = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = score_prediction(rec["predicted"], rec.get("answers", []))
            s["kind"] = rec.get("kind", "")
            per_question.append(s)

            for li in rec.get("retrieved_layers", []):
                layer_counter[li] += 1

            detail_rows.append({
                "method": rec.get("method", ""),
                "story_id": rec.get("story_id", ""),
                "kind": rec.get("kind", ""),
                "question_id": rec.get("question_id", ""),
                "f1": round(s["f1"], 4),
                "rouge_l": round(s["rouge_l"], 4),
                "bleu": round(s["bleu"], 4),
                "predicted": (rec.get("predicted") or "")[:80],
                "gold_1": (rec.get("answers") or [""])[0][:60],
                "tree_layers": rec.get("tree_layers", 0),
                "tree_cached": rec.get("tree_cached", False),
            })

    summary = aggregate_scores(per_question)
    summary["file"] = os.path.basename(path)

    # CIs on the two metrics that matter most
    summary["ci_f1"] = bootstrap_ci([q["f1"] for q in per_question])
    summary["ci_rouge_l"] = bootstrap_ci([q["rouge_l"] for q in per_question])

    # Books vs scripts breakdown
    for kind, label in (("gutenberg", "books"), ("movie", "scripts")):
        subset = [q for q in per_question if q["kind"] == kind]
        if subset:
            summary[f"rouge_l_{label}"] = round(
                sum(q["rouge_l"] for q in subset) / len(subset), 4)
            summary[f"f1_{label}"] = round(
                sum(q["f1"] for q in subset) / len(subset), 4)
            summary[f"n_{label}"] = len(subset)

    total_layers = sum(layer_counter.values())
    if total_layers:
        summary["layer_dist"] = {
            str(k): round(v / total_layers, 4) for k, v in sorted(layer_counter.items())
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
    print(f"\n  {method}: ROUGE-L={s['rouge_l']:.4f} "
          f"[{s['ci_rouge_l']['lo']:.3f}, {s['ci_rouge_l']['hi']:.3f}]  "
          f"F1={s['f1']:.4f} [{s['ci_f1']['lo']:.3f}, {s['ci_f1']['hi']:.3f}]  "
          f"BLEU={s['bleu']:.4f}  (n={s['n']})")
    if "rouge_l_books" in s or "rouge_l_scripts" in s:
        parts = []
        if "rouge_l_books" in s:
            parts.append(f"books ROUGE-L={s['rouge_l_books']:.3f} (n={s['n_books']})")
        if "rouge_l_scripts" in s:
            parts.append(f"scripts ROUGE-L={s['rouge_l_scripts']:.3f} (n={s['n_scripts']})")
        print(f"    By kind: {'; '.join(parts)}")
    if "layer_dist" in s:
        dist = ", ".join(f"L{k}={v:.2f}" for k, v in s["layer_dist"].items())
        print(f"    Retrieved layer dist: {dist}")


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

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    _write_results_csv(summaries, os.path.join(run_dir, "results.csv"))
    return summaries


def _write_results_csv(summaries, path):
    if not summaries:
        return
    cols = ["method", "n", "rouge_l", "f1", "bleu",
            "rouge_l_books", "rouge_l_scripts", "n_books", "n_scripts"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["rouge_ci_lo", "rouge_ci_hi", "f1_ci_lo", "f1_ci_hi"])
        for s in summaries:
            w.writerow(
                [s.get(c, "") for c in cols] +
                [s["ci_rouge_l"]["lo"], s["ci_rouge_l"]["hi"],
                 s["ci_f1"]["lo"], s["ci_f1"]["hi"]]
            )


def main():
    p = argparse.ArgumentParser(description="Score NarrativeQA predictions.")
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
        print("\n" + "=" * 74)
        print("  COMBINED COMPARISON (sorted by ROUGE-L, the NarrativeQA headline)")
        print("=" * 74)
        all_summaries.sort(key=lambda s: -s.get("rouge_l", 0))
        print(f"  {'method':<16s} {'ROUGE-L':>20s} {'F1':>20s} {'BLEU':>8s}")
        for s in all_summaries:
            rl = f"{s['rouge_l']:.3f}[{s['ci_rouge_l']['lo']:.2f},{s['ci_rouge_l']['hi']:.2f}]"
            f1 = f"{s['f1']:.3f}[{s['ci_f1']['lo']:.2f},{s['ci_f1']['hi']:.2f}]"
            print(f"  {s['method']:<16s} {rl:>20s} {f1:>20s} {s['bleu']:>8.3f}")
        print("\n  Interpretation guide:")
        print("  - The preregistered hypothesis: tree methods' ROUGE-L CIs should sit")
        print("    above flat's here MORE clearly than on QASPER, because NarrativeQA")
        print("    questions target plot-level content over much longer documents.")
        print("  - Overlapping CIs = within noise at this n; do not read a ranking.")


if __name__ == "__main__":
    main()
