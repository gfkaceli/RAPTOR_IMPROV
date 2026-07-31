"""
run_narrativeqa_eval.py — Main NarrativeQA evaluation script.

For each (clustering method, story):
  1. Build a RAPTOR tree from the cleaned story text — OR load it from the tree
     cache if it was built in a previous session (see --cache-dir below).
  2. For each question: retrieve top-k with the question text, then generate a
     short free-form answer from the retrieved context only.
  3. Log the prediction with its reference answers, retrieved node layers,
     tree stats, and build time (with a cached flag).

NarrativeQA-specific design notes:
  - TREE CACHING IS THE POINT OF THIS RUNNER. Stories average tens of thousands
    of tokens, so a single (method, story) build can take minutes; 6 methods x N
    stories will not fit in one Colab session. Pass --cache-dir (ideally a
    Google Drive path) and trees are saved with RAPTOR's native ra.save(path)
    and reloaded with RetrievalAugmentation(config=cfg, tree=path). Do NOT
    replace this with manual pickling of ra.tree — that breaks the retriever
    silently and produces zero-F1 runs.
  - max_k for the silhouette-search methods (K-Means, agglomerative) is raised
    to 25 (QASPER/QuALITY used 15): a 60k-token story at 140-token chunks
    yields ~400+ leaves, and capping layer-0 at 15 clusters would force ~30
    chunks per cluster — summaries too broad to be useful. Silhouette can still
    choose fewer clusters when that fits better; the cap only widens the search.
  - num_layers=5 (vs 4) allows the deeper trees long documents naturally form;
    the builder's stop condition ends recursion earlier when nodes run out.
  - Other parameters mirror the QuALITY run for cross-experiment comparability:
    tb_max_tokens=140, summarization_length=600, tr_top_k=10, seed 224,
    Leiden schedule extended one layer deeper, DBSCAN min_samples=4 / p88.

Usage:
    python -m eval_narrativeqa.preprocess_narrativeqa --split validation --max-articles 3
    python -m eval_narrativeqa.run_narrativeqa_eval --model-tier base \
        --max-articles 3 --cache-dir /content/drive/MyDrive/narrativeqa_trees
    python -m eval_narrativeqa.score_narrativeqa experiments/narrativeqa/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.cluster_tree_builder import ClusterTreeConfig
from raptor.clustering import (
    GMMClusterer, LeidenClusterer, LeidenConfig,
    KMeansClusterer, AgglomerativeClusterer, DBSCANClusterer,
)
from raptor.flat_retriever import FlatRetriever

from eval_narrativeqa.models import load_models, MODEL_TIERS, EMB_MODEL


# ---------------------------------------------------------------------------
# Config factories (mirroring the QuALITY run; NarrativeQA deltas documented
# in the module docstring)
# ---------------------------------------------------------------------------

def make_original_config(emb, summ, qa):
    return RetrievalAugmentationConfig(
        embedding_model=emb, summarization_model=summ, qa_model=qa,
        tb_max_tokens=140, tb_num_layers=5, tb_summarization_length=600,
        tr_top_k=10, tr_selection_mode="top_k",
    )


def _tree_cfg(clusterer, emb, summ):
    return ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={}, reduction_dimension=10,
        summarization_model=summ, embedding_models={"EMB": emb},
        cluster_embedding_model="EMB", max_tokens=140, num_layers=5,
        summarization_length=600,
    )


def _wrap(tree_config, emb, qa):
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config, qa_model=qa, embedding_model=emb,
        tr_top_k=10, tr_selection_mode="top_k",
    )


def make_gmm(emb, summ, qa):
    c = GMMClusterer(reduction_dimension=10, soft_threshold=0.1,
                     force_hard_clustering=False, random_state=224)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_leiden(emb, summ, qa):
    lcfg = LeidenConfig(k_neighbors=10, use_adjacency_edges=True, adjacency_weight=0.5,
                        resolution=1.0,
                        resolution_schedule={0: 1.3, 1: 0.9, 2: 0.6, 3: 0.5},
                        partition_type="RBConfiguration", min_cluster_size=1)
    c = LeidenClusterer(config=lcfg, random_state=224,
                        reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_kmeans(emb, summ, qa):
    c = KMeansClusterer(k_strategy="silhouette", min_k=3, max_k=25, random_state=224,
                        reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_agglomerative(emb, summ, qa):
    c = AgglomerativeClusterer(cut_strategy="silhouette", linkage="average",
                               min_k=3, max_k=25, random_state=224,
                               reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_dbscan(emb, summ, qa):
    c = DBSCANClusterer(noise_strategy="nearest", min_samples=4, eps_percentile=88,
                        random_state=224, reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_flat(emb, summ, qa):
    return FlatRetriever(embedding_model=emb, qa_model=qa, top_k=10, chunk_size=140)


METHODS = {
    "flat":          (make_flat,            "Flat SBERT retrieval (no tree)"),
    "original":      (make_original_config, "RAPTOR GMM+UMAP upstream"),
    "gmm":           (make_gmm,             "GMMClusterer"),
    "leiden":        (make_leiden,          "LeidenClusterer (reduced)"),
    "kmeans":        (make_kmeans,          "KMeansClusterer (reduced)"),
    "agglomerative": (make_agglomerative,   "AgglomerativeClusterer (reduced)"),
    "dbscan":        (make_dbscan,          "DBSCANClusterer (reduced)"),
}


# ---------------------------------------------------------------------------
# Retrieval with best-effort layer logging (same hook as the QuALITY runner)
# ---------------------------------------------------------------------------

def retrieve_with_layers(ra, question: str):
    """
    Retrieve context and record the tree layer of each retrieved node.

    RetrievalAugmentation.retrieve returns (context, layer_information) where
    layer_information is a list of {"node_index", "layer_number"} dicts. The
    FlatRetriever returns just the context string (no tree, so no layers). We
    normalise both shapes here so the caller always gets (context_str, layers).
    """
    result = ra.retrieve(question)
    if isinstance(result, tuple):
        context, layer_information = result
    else:
        context, layer_information = result, []
    layers: List[int] = [
        li.get("layer_number", -1) for li in (layer_information or [])
    ]
    return context, layers


# ---------------------------------------------------------------------------
# Build-or-load with native tree caching
# ---------------------------------------------------------------------------

def get_retriever(method_name, story_id, story_text, emb, summ, qa, cache_dir):
    """
    Returns (ra, build_time_sec, cached: bool).

    Hierarchical methods: if a cached tree exists for (method, story), load it
    via RAPTOR's native RetrievalAugmentation(config=cfg, tree=path); otherwise
    build with add_documents and persist with ra.save(path). Flat retriever is
    rebuilt every time (chunk embedding is cheap relative to tree builds).
    """
    factory_fn, _ = METHODS[method_name]
    result = factory_fn(emb, summ, qa)

    # Flat retriever instance — no tree, no cache.
    if hasattr(result, "add_documents"):
        t0 = time.time()
        result.add_documents(story_text)
        return result, time.time() - t0, False

    cfg = result
    cache_path = None
    if cache_dir:
        method_dir = os.path.join(cache_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        cache_path = os.path.join(method_dir, f"{story_id}.tree")

    if cache_path and os.path.exists(cache_path):
        t0 = time.time()
        ra = RetrievalAugmentation(config=cfg, tree=cache_path)
        return ra, time.time() - t0, True

    ra = RetrievalAugmentation(config=cfg)
    t0 = time.time()
    ra.add_documents(story_text)
    build_time = time.time() - t0
    if cache_path:
        try:
            ra.save(cache_path)   # native save ONLY — never pickle ra.tree
        except Exception as exc:
            print(f"    [WARN] tree cache save failed: {exc}", file=sys.stderr)
    return ra, build_time, False


def evaluate_method(method_name, stories, emb, summ, qa, out_path,
                    cache_dir=None, verbose=False):
    description = METHODS[method_name][1]
    print(f"\n{'='*70}\n  {method_name} — {description}\n{'='*70}")

    n_stories = n_questions = n_cached = 0
    total_build = 0.0
    failures = 0

    with open(out_path, "w") as f_out:
        for si, story in enumerate(stories):
            sid = story["story_id"]
            try:
                ra, build_time, cached = get_retriever(
                    method_name, sid, story["full_text"], emb, summ, qa, cache_dir)
                total_build += build_time
                n_cached += int(cached)
            except Exception as exc:
                print(f"  [{si+1}/{len(stories)}] {sid}: BUILD FAILED — {exc}")
                failures += 1
                continue

            tree = getattr(ra, "tree", None)
            n_layers = getattr(tree, "num_layers", 0)
            n_nodes = len(getattr(tree, "all_nodes", {}) or {})
            tag = "cache" if cached else "build"
            print(f"  [{si+1}/{len(stories)}] {sid} ({story['kind']}): "
                  f"{story['n_words']:,} words -> {n_layers} layers, {n_nodes} nodes "
                  f"({tag} {build_time:.1f}s), {len(story['questions'])} questions")

            for q in story["questions"]:
                try:
                    context, layers = retrieve_with_layers(ra, q["question"])
                    predicted = qa.answer_question(context, q["question"])
                except Exception as exc:
                    if verbose:
                        print(f"      Q {q['question_id']}: ERROR {exc}")
                    predicted, layers = "", []

                f_out.write(json.dumps({
                    "method": method_name,
                    "story_id": sid,
                    "kind": story["kind"],
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "answers": q["answers"],
                    "predicted": predicted,
                    "retrieved_layers": layers,
                    "tree_layers": n_layers,
                    "tree_nodes": n_nodes,
                    "build_time_sec": round(build_time, 2),
                    "tree_cached": cached,
                }) + "\n")
                n_questions += 1
            n_stories += 1

    print(f"\n  {method_name}: {n_stories} stories ({n_cached} from cache), "
          f"{n_questions} questions, {total_build:.1f}s build, {failures} failures")
    return {"method": method_name, "n_stories": n_stories,
            "n_questions": n_questions, "n_cached_trees": n_cached,
            "build_time_sec": round(total_build, 2),
            "failures": failures, "output": out_path}


def set_seed(seed=224):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser(description="Run NarrativeQA free-form QA evaluation.")
    p.add_argument("--data", default="data/narrativeqa/validation.json")
    p.add_argument("--model-tier", default="base", choices=list(MODEL_TIERS.keys()))
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    p.add_argument("--max-articles", type=int, default=None,
                   help="Use only the first N stories from the preprocessed file.")
    p.add_argument("--cache-dir", default=None,
                   help="Directory for tree caching across sessions "
                        "(strongly recommended; e.g. a Google Drive path).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=224)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print("  NarrativeQA Free-Form QA Evaluation")
    print(f"  Data: {args.data} | Tier: {args.model_tier} | Methods: {', '.join(args.methods)}")
    print(f"  Tree cache: {args.cache_dir or 'DISABLED (not recommended for full runs)'}")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found. Run preprocess_narrativeqa first.")
        sys.exit(1)

    with open(args.data) as f:
        stories = json.load(f)
    if args.max_articles:
        stories = stories[:args.max_articles]

    n_q = sum(len(s["questions"]) for s in stories)
    total_words = sum(s["n_words"] for s in stories)
    print(f"\n  {len(stories)} stories, {n_q} questions, "
          f"{total_words:,} total story words")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or os.path.join("experiments", "narrativeqa", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output: {out_dir}")

    print("\nLoading models...")
    emb, summ, qa = load_models(args.model_tier)
    print("Models ready (short-answer QA + narrative summarization).")

    run_meta = []
    for m in args.methods:
        out_path = os.path.join(out_dir, f"predictions_{m}.jsonl")
        set_seed(args.seed)
        run_meta.append(evaluate_method(m, stories, emb, summ, qa, out_path,
                                        cache_dir=args.cache_dir,
                                        verbose=args.verbose))

    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "timestamp": ts, "model_tier": args.model_tier,
            "embedding_model": EMB_MODEL,
            "n_stories": len(stories), "n_questions_total": n_q,
            "seed": args.seed, "cache_dir": args.cache_dir,
            "data_file": args.data, "methods": run_meta,
        }, f, indent=2)

    print(f"\nDone. Run scoring: python -m eval_narrativeqa.score_narrativeqa {out_dir}")


if __name__ == "__main__":
    main()
