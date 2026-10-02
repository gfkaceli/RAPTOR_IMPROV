"""
run_structure_eval.py — Structure-aware retrieval experiment on NarrativeQA.

Runs every retrieval policy in raptor/structure_aware_retriever.py over the
trees ALREADY CACHED by run_narrativeqa_eval.py (--cache-dir). No tree is
rebuilt: the cost of this experiment is QA generation only.

Per (method, story):
  1. Load the cached tree through RetrievalAugmentation(config=cfg, tree=path),
     the same native path used when the trees were built, and wrap it in a
     StructureAwareRetriever that reuses that retriever's embedding model,
     tokenizer, and embedding key.
  2. Validate the tree structure (logged per story).
  3. On the first loaded story of each method, run two checks:
       - embedding consistency: re-embed a few leaves and compare with the
         stored vectors. A mismatch means the query-side embedding model differs
         from the one that built the trees; the run aborts, because every
         retrieval would be silently wrong.
       - RAPTOR equivalence: the collapsed ranking must reproduce RAPTOR's own
         collapsed-tree top-k on the first few questions.
  4. For each question, run every policy and generate an answer from the
     retrieved context (a plain string; no tuple/str() conversion).

Answer reuse: decoding is greedy (do_sample=False), so two policies that
retrieve an identical context for the same question produce the same answer.
Such answers are reused rather than regenerated and flagged answer_reused.

Output: one predictions file per (method, policy),
    predictions_<method>__<policy>.jsonl
appended incrementally, so an interrupted Colab session resumes with --resume.

Usage:
    python -m eval_narrativeqa.run_structure_eval \
        --data data/narrativeqa/train.json \
        --cache-dir /content/drive/MyDrive/narrativeqa_trees \
        --model-tier local-xl --methods leiden kmeans dbscan --max-stories 5
    python -m eval_narrativeqa.score_structure experiments/narrativeqa_structure/<ts>

The --data file must be the same preprocessed file used to build the cached
trees, so story ids and question ids match. The --model-tier should match the
tier of the completed NarrativeQA run, so answers come from the same QA model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Set

HIER_METHODS = ("gmm", "leiden", "kmeans", "agglomerative", "dbscan")
DEFAULT_TREE_PATTERN = "{cache_dir}/{method}/{story_id}.tree"


def load_completed(path: str) -> Set[str]:
    """Question ids already written to a predictions file (for --resume)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a truncated final line
    return done


def answer_story(sar, story: Dict, policies: List[str], qa, writers: Dict,
                 completed: Dict[str, Set[str]], method: str) -> Dict:
    """Run every policy for every question of one story. Returns counters.

    Pure with respect to raptor: takes any retriever exposing .retrieve(q, policy)
    and .index, and any QA model exposing .answer_question(context, question).
    """
    counts = {"generated": 0, "reused": 0, "skipped": 0, "errors": 0}
    answer_cache: Dict[str, str] = {}
    tree_layers = sar.index.max_layer
    tree_nodes = len(sar.index.nodes)

    for q in story["questions"]:
        qid = q["question_id"]
        for policy in policies:
            if qid in completed[policy]:
                counts["skipped"] += 1
                continue
            try:
                res = sar.retrieve(q["question"], policy)
                key = hashlib.sha1(f"{qid}\x00{res.context}".encode()).hexdigest()
                reused = key in answer_cache
                if reused:
                    pred = answer_cache[key]
                    counts["reused"] += 1
                else:
                    pred = qa.answer_question(res.context, q["question"])
                    answer_cache[key] = pred
                    counts["generated"] += 1
                rec = {
                    "method": method, "policy": policy,
                    "story_id": story["story_id"], "kind": story.get("kind", ""),
                    "question_id": qid, "question": q["question"],
                    "answers": q["answers"], "predicted": pred,
                    "context_tokens": res.tokens, "token_budget": res.budget,
                    "n_nodes": len(res.nodes), "node_ids": [n.index for n in res.nodes],
                    "layers": res.layers, "ops": res.ops,
                    "route": res.route, "fallback": res.fallback,
                    "answer_reused": reused,
                    "tree_layers": tree_layers, "tree_nodes": tree_nodes,
                }
            except Exception as exc:  # keep the run alive; record the failure
                counts["errors"] += 1
                rec = {"method": method, "policy": policy,
                       "story_id": story["story_id"], "kind": story.get("kind", ""),
                       "question_id": qid, "question": q["question"],
                       "answers": q["answers"], "predicted": "",
                       "error": repr(exc)}
            writers[policy].write(json.dumps(rec) + "\n")
        for w in writers.values():
            w.flush()
    return counts


def main():
    p = argparse.ArgumentParser(description="Structure-aware retrieval on NarrativeQA "
                                            "(reuses cached trees).")
    p.add_argument("--data", default="data/narrativeqa/train.json",
                   help="Preprocessed file used to BUILD the cached trees.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tree-pattern", default=DEFAULT_TREE_PATTERN,
                   help="Format string for cached tree paths.")
    p.add_argument("--model-tier", default="base",
                   help="Must match the tier of the completed NarrativeQA run.")
    p.add_argument("--methods", nargs="+", default=list(HIER_METHODS),
                   choices=list(HIER_METHODS))
    p.add_argument("--policies", nargs="+", default=None,
                   help="Default: all policies.")
    p.add_argument("--max-stories", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=4096,
                   help="Context budget in tokenizer tokens, shared by all policies. "
                        "1400 ~= ten 140-token leaves, the node count of the "
                        "completed run (top_k=10).")
    p.add_argument("--seed-fraction", type=float, default=0.6)
    p.add_argument("--expand-per-node", type=int, default=2)
    p.add_argument("--rollup-min-children", type=int, default=2)
    p.add_argument("--adaptive-margin", type=float, default=0.0,
                   help="Router threshold. Tune on stories DISJOINT from the "
                        "evaluation sample; do not tune on these results.")
    p.add_argument("--context-order", choices=("structured", "similarity"),
                   default="structured")
    p.add_argument("--verify-questions", type=int, default=3)
    p.add_argument("--skip-checks", action="store_true",
                   help="Do not abort on a failed embedding-consistency check.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=224)
    args = p.parse_args()

    # Heavy imports only after argument parsing
    from raptor import RetrievalAugmentation
    from .structure_aware_retriever import (
        POLICIES, StructureAwareRetriever, verify_against_raptor,
        verify_embedding_consistency)
    from eval_narrativeqa.models import load_models, MODEL_TIERS, EMB_MODEL
    from eval_narrativeqa.run_narrativeqa_eval import METHODS, set_seed

    policies = args.policies or list(POLICIES)
    bad = [x for x in policies if x not in POLICIES]
    if bad:
        sys.exit(f"Unknown policies {bad}. Choices: {POLICIES}")
    if args.model_tier not in MODEL_TIERS:
        sys.exit(f"Unknown tier '{args.model_tier}'. Choices: {list(MODEL_TIERS)}")

    set_seed(args.seed)
    with open(args.data) as f:
        stories = json.load(f)
    if args.max_stories:
        stories = stories[: args.max_stories]
    n_q = sum(len(s["questions"]) for s in stories)

    out_dir = args.output_dir or os.path.join(
        "experiments", "narrativeqa_structure", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  NarrativeQA structure-aware retrieval (cached trees, no rebuild)")
    print(f"  {len(stories)} stories, {n_q} questions | methods: {', '.join(args.methods)}")
    print(f"  policies: {', '.join(policies)}")
    print(f"  token budget: {args.token_budget} | order: {args.context_order}")
    print(f"  upper bound on QA generations: {n_q * len(args.methods) * len(policies):,} "
          f"(identical contexts are reused)")
    print(f"  output: {out_dir}")
    print("=" * 72)

    emb, summ, qa = load_models(args.model_tier)
    sar_kwargs = dict(token_budget=args.token_budget, seed_fraction=args.seed_fraction,
                      expand_per_node=args.expand_per_node,
                      rollup_min_children=args.rollup_min_children,
                      adaptive_margin=args.adaptive_margin,
                      context_order=args.context_order)

    meta = {"args": vars(args), "embedding_model": EMB_MODEL, "policies": policies,
            "n_stories": len(stories), "n_questions": n_q, "methods": {}}

    for method in args.methods:
        print(f"\n--- {method} ---")
        paths = {pol: os.path.join(out_dir, f"predictions_{method}__{pol}.jsonl")
                 for pol in policies}
        completed = {pol: (load_completed(pth) if args.resume else set())
                     for pol, pth in paths.items()}
        writers = {pol: open(pth, "a" if args.resume else "w") for pol, pth in paths.items()}
        m_meta = {"missing_trees": [], "validation": {}, "checks": None,
                  "counts": {"generated": 0, "reused": 0, "skipped": 0, "errors": 0}}
        t0 = time.time()
        try:
            for si, story in enumerate(stories):
                sid = story["story_id"]
                path = args.tree_pattern.format(cache_dir=args.cache_dir,
                                                method=method, story_id=sid)
                if not os.path.exists(path):
                    m_meta["missing_trees"].append(sid)
                    print(f"  [{si+1}/{len(stories)}] {sid}: NO CACHED TREE — skipped")
                    continue

                cfg = METHODS[method][0](emb, summ, qa)
                ra = RetrievalAugmentation(config=cfg, tree=path)
                sar = StructureAwareRetriever.from_retrieval_augmentation(ra, **sar_kwargs)

                v = sar.index.validate()
                m_meta["validation"][sid] = v
                if v["problems"]:
                    print(f"    [WARN] structure problems: {v['problems']}")

                if m_meta["checks"] is None:
                    emb_chk = verify_embedding_consistency(sar)
                    qs = [q["question"] for q in story["questions"][: args.verify_questions]]
                    rap_chk = verify_against_raptor(sar, ra, qs, k=10)
                    m_meta["checks"] = {"embedding": emb_chk, "raptor_overlap": rap_chk}
                    print(f"    checks: embedding cosines {emb_chk['cosines']} "
                          f"(ok={emb_chk['ok']}); RAPTOR top-10 overlap "
                          f"{rap_chk['mean_overlap']}")
                    if not emb_chk["ok"] and not args.skip_checks:
                        raise SystemExit(
                            "Embedding consistency check FAILED: the embedding model "
                            f"({EMB_MODEL}) does not reproduce the stored tree "
                            "embeddings. Set RAPTOR_EMB_MODEL to the model that built "
                            "the trees. Aborting.")
                    if rap_chk["mean_overlap"] is not None and rap_chk["mean_overlap"] < 1.0:
                        print("    [WARN] collapsed ranking differs from RAPTOR's; "
                              "inspect before trusting results (ties can cause "
                              "small differences).")

                c = answer_story(sar, story, policies, qa, writers, completed, method)
                for k in c:
                    m_meta["counts"][k] += c[k]
                print(f"  [{si+1}/{len(stories)}] {sid}: depth {v['max_layer']}, "
                      f"{v['n_nodes']} nodes | generated {c['generated']}, "
                      f"reused {c['reused']}, skipped {c['skipped']}, errors {c['errors']}")
        finally:
            for w in writers.values():
                w.close()
        m_meta["seconds"] = round(time.time() - t0, 1)
        meta["methods"][method] = m_meta
        if m_meta["missing_trees"]:
            print(f"  [WARN] {len(m_meta['missing_trees'])} stories had no cached tree "
                  f"for {method}; its evaluation set is incomplete.")

    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nDone. Score with: python -m eval_narrativeqa.score_structure {out_dir}")


if __name__ == "__main__":
    main()
