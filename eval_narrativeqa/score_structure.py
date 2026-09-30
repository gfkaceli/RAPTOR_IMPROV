"""
score_structure.py — Score the structure-aware retrieval experiment.

Reads predictions_<method>__<policy>.jsonl files written by run_structure_eval.py
and reports, per (method, policy):

  - F1, ROUGE-L, BLEU (max over the two references, as in score_narrativeqa)
  - 95% intervals from a STORY-LEVEL bootstrap. Questions are clustered by
    story (about 30 per story share one tree), so resampling questions would
    understate uncertainty.
  - Paired comparison against the collapsed policy on the same tree:
    mean difference, story-level paired-bootstrap 95% interval, approximate
    two-sided p-value, and Holm-adjusted p-values across all comparisons.
  - Context tokens actually used (budget-fairness check), share of retrieved
    nodes above the leaves, operator mix, adaptive route mix, and the
    fraction of answers that changed relative to collapsed retrieval.
  - Oracle headroom: the score if each question used its best FIXED policy.
    Caveat: a per-question maximum over several noisy policies is biased
    upward even when policies are equally good, so the headroom is an upper
    bound, not an achievable gain.

Usage:
    python -m eval_narrativeqa.score_structure experiments/narrativeqa_structure/<ts>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .narrativeqa_metric import score_prediction

FIXED = ("collapsed", "leaf_only", "summary_only", "drill_down", "roll_up", "sibling")
ORDER = FIXED + ("adaptive",)
METRICS = ("f1", "rouge_l")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _by_story(values: Dict[str, float], story_of: Dict[str, str]) -> Dict[str, List[float]]:
    out = defaultdict(list)
    for qid, v in values.items():
        out[story_of[qid]].append(v)
    return out


def cluster_bootstrap(by_story: Dict[str, List[float]], n_boot=2000, seed=224) -> Dict:
    """Resample whole stories with replacement; statistic = mean over the
    questions of the resampled stories."""
    stories = list(by_story)
    sums = np.array([sum(by_story[s]) for s in stories])
    cnts = np.array([len(by_story[s]) for s in stories])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stories), size=(n_boot, len(stories)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    mean = sums.sum() / cnts.sum()
    return {"mean": float(mean), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "boot": boot}


def paired_test(diff_by_story: Dict[str, List[float]], n_boot=2000, seed=224) -> Dict:
    r = cluster_bootstrap(diff_by_story, n_boot, seed)
    boot = r.pop("boot")
    # Approximate two-sided bootstrap p-value (resolution 1/n_boot)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    r["p"] = float(min(1.0, max(p, 1.0 / n_boot)))
    return r


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> Dict:
    """Returns {(method, policy): {qid: record-with-scores}}."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "predictions_*__*.jsonl"))):
        stem = os.path.basename(path)[len("predictions_"):-len(".jsonl")]
        method, policy = stem.split("__", 1)
        recs = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r.update(score_prediction(r.get("predicted", ""), r.get("answers", [])))
                recs[r["question_id"]] = r   # last write wins on resumed runs
        groups[(method, policy)] = recs
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score structure-aware retrieval runs.")
    ap.add_argument("run_dir")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    groups = load_run(args.run_dir)
    if not groups:
        print(f"No predictions_<method>__<policy>.jsonl files in {args.run_dir}")
        return

    methods = sorted({m for m, _ in groups})
    rows, paired_rows, oracle_rows = [], [], []

    for method in methods:
        pols = [p for p in ORDER if (method, p) in groups]
        base = groups.get((method, "collapsed"), {})
        for pol in pols:
            recs = groups[(method, pol)]
            if not recs:
                continue
            story_of = {q: r["story_id"] for q, r in recs.items()}
            row = {"method": method, "policy": pol, "n": len(recs),
                   "n_stories": len(set(story_of.values())),
                   "errors": sum(1 for r in recs.values() if "error" in r)}
            for m in ("f1", "rouge_l", "bleu"):
                row[m] = float(np.mean([r[m] for r in recs.values()]))
            for m in METRICS:
                ci = cluster_bootstrap(_by_story({q: r[m] for q, r in recs.items()},
                                                 story_of), args.n_boot)
                row[f"{m}_lo"], row[f"{m}_hi"] = ci["lo"], ci["hi"]
            ok = [r for r in recs.values() if "error" not in r]
            row["mean_context_tokens"] = float(np.mean([r["context_tokens"] for r in ok])) if ok else 0.0
            row["mean_nodes"] = float(np.mean([r["n_nodes"] for r in ok])) if ok else 0.0
            layers = [l for r in ok for l in r["layers"]]
            row["summary_share"] = float(np.mean([l > 0 for l in layers])) if layers else 0.0
            ops = Counter(o for r in ok for o in r["ops"])
            tot = sum(ops.values()) or 1
            row["op_mix"] = {k: round(v / tot, 3) for k, v in ops.items()}
            routes = Counter(r["route"] for r in ok if r.get("route"))
            row["route_mix"] = {k: round(v / sum(routes.values()), 3)
                                for k, v in routes.items()} if routes else {}
            row["reused_share"] = float(np.mean([r.get("answer_reused", False) for r in ok])) if ok else 0.0
            if pol != "collapsed" and base:
                common = [q for q in recs if q in base]
                row["changed_vs_collapsed"] = float(np.mean(
                    [_norm(recs[q]["predicted"]) != _norm(base[q]["predicted"])
                     for q in common])) if common else None
            for kind, label in (("gutenberg", "books"), ("movie", "scripts")):
                sub = [r["rouge_l"] for r in recs.values() if r.get("kind") == kind]
                if sub:
                    row[f"rouge_l_{label}"] = float(np.mean(sub))
            rows.append(row)

            # Paired comparison against collapsed on the same tree
            if pol != "collapsed" and base:
                common = [q for q in recs if q in base]
                pr = {"method": method, "policy": pol, "n_paired": len(common)}
                for m in METRICS:
                    diffs = {q: recs[q][m] - base[q][m] for q in common}
                    t = paired_test(_by_story(diffs, story_of), args.n_boot)
                    pr[f"{m}_diff"], pr[f"{m}_lo"], pr[f"{m}_hi"], pr[f"{m}_p"] = (
                        t["mean"], t["lo"], t["hi"], t["p"])
                paired_rows.append(pr)

        # Oracle headroom over the fixed policies present for this method
        fixed = [p for p in FIXED if (method, p) in groups]
        if len(fixed) >= 2 and "collapsed" in fixed:
            common = set.intersection(*[set(groups[(method, p)]) for p in fixed])
            if common:
                orow = {"method": method, "n": len(common), "policies": fixed}
                for m in METRICS:
                    per_pol = {p: np.mean([groups[(method, p)][q][m] for q in common])
                               for p in fixed}
                    oracle = np.mean([max(groups[(method, p)][q][m] for p in fixed)
                                      for q in common])
                    best = max(per_pol, key=per_pol.get)
                    orow[f"{m}_collapsed"] = float(per_pol["collapsed"])
                    orow[f"{m}_best_single"] = float(per_pol[best])
                    orow[f"{m}_best_policy"] = best
                    orow[f"{m}_oracle"] = float(oracle)
                    orow[f"{m}_headroom_vs_collapsed"] = float(oracle - per_pol["collapsed"])
                oracle_rows.append(orow)

    # Holm correction across all paired comparisons, per metric
    for m in METRICS:
        if paired_rows:
            adj = holm([r[f"{m}_p"] for r in paired_rows])
            for r, a in zip(paired_rows, adj):
                r[f"{m}_p_holm"] = a

    # ---------------- print ----------------
    pct = lambda x: f"{100 * x:6.2f}"
    print("\n" + "=" * 100)
    print("  Per (method, policy) — story-level 95% intervals")
    print("=" * 100)
    print(f"  {'method':<14}{'policy':<14}{'n':>5} {'F1':>7} {'[lo, hi]':>16} "
          f"{'ROUGE-L':>8} {'tokens':>7} {'summ%':>6} {'changed':>8}")
    for r in rows:
        ch = r.get("changed_vs_collapsed")
        print(f"  {r['method']:<14}{r['policy']:<14}{r['n']:>5} {pct(r['f1'])} "
              f"[{pct(r['f1_lo'])},{pct(r['f1_hi'])}] {pct(r['rouge_l'])}  "
              f"{r['mean_context_tokens']:>6.0f} {100 * r['summary_share']:>5.1f} "
              f"{'' if ch is None else f'{100 * ch:7.1f}%'}")

    # Budget fairness warning
    for method in methods:
        base = next((r for r in rows if r["method"] == method and r["policy"] == "collapsed"), None)
        if not base or not base["mean_context_tokens"]:
            continue
        for r in rows:
            if r["method"] == method and r["policy"] != "collapsed":
                dev = r["mean_context_tokens"] / base["mean_context_tokens"] - 1
                if abs(dev) > 0.10:
                    print(f"  [NOTE] {method}/{r['policy']} uses {100 * dev:+.0f}% context "
                          f"tokens vs collapsed; interpret its difference with that in mind.")

    if paired_rows:
        print("\n" + "=" * 100)
        print("  Paired vs collapsed (same tree) — story-level paired bootstrap, Holm-adjusted")
        print("=" * 100)
        print(f"  {'method':<14}{'policy':<14}{'n':>5} {'dF1':>7} {'[lo, hi]':>16} "
              f"{'p_holm':>7} {'dROUGE-L':>9} {'p_holm':>7}")
        for r in paired_rows:
            print(f"  {r['method']:<14}{r['policy']:<14}{r['n_paired']:>5} "
                  f"{pct(r['f1_diff'])} [{pct(r['f1_lo'])},{pct(r['f1_hi'])}] "
                  f"{r['f1_p_holm']:>7.3f} {pct(r['rouge_l_diff']):>9} "
                  f"{r['rouge_l_p_holm']:>7.3f}")

    if oracle_rows:
        print("\n" + "=" * 100)
        print("  Oracle headroom over fixed policies (UPPER BOUND; biased upward)")
        print("=" * 100)
        for r in oracle_rows:
            print(f"  {r['method']:<14} F1: collapsed {pct(r['f1_collapsed'])} | best single "
                  f"({r['f1_best_policy']}) {pct(r['f1_best_single'])} | oracle "
                  f"{pct(r['f1_oracle'])} | headroom {pct(r['f1_headroom_vs_collapsed'])}")

    # ---------------- write ----------------
    with open(os.path.join(args.run_dir, "structure_results.json"), "w") as f:
        json.dump({"per_policy": rows, "paired_vs_collapsed": paired_rows,
                   "oracle": oracle_rows}, f, indent=2)
    for name, data in (("structure_results.csv", rows),
                       ("paired_vs_collapsed.csv", paired_rows),
                       ("oracle_headroom.csv", oracle_rows)):
        if data:
            flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                     for k, v in d.items()} for d in data]
            keys = list(dict.fromkeys(k for d in flat for k in d))
            with open(os.path.join(args.run_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(flat)
    print(f"\nWrote structure_results.{{json,csv}}, paired_vs_collapsed.csv, "
          f"oracle_headroom.csv to {args.run_dir}")


if __name__ == "__main__":
    main()
