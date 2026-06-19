"""
run_qasper_eval.py — Main QASPER evaluation script.

For each (clustering method, paper) pair:
  1. Build a RAPTOR tree from the paper's full text
  2. Answer every question for that paper using the tree
  3. Save predictions to a JSONL file

Follows the Laitenberger pattern: precreate + answer in one script for
simplicity (since we don't need to cache trees across separate eval runs
during a single thesis experiment).

Usage:
    # Preprocess first (one-time)
    python -m eval_qasper.preprocess_qasper --max-papers 10

    # Then evaluate
    python -m eval_qasper.run_qasper_eval --model-tier base --methods original gmm leiden
    python -m eval_qasper.run_qasper_eval --model-tier api --max-papers 5

Output:
    experiments/qasper/<timestamp>/predictions_<method>.jsonl
        One JSON object per question:
        {
          "paper_id": ..., "question_id": ..., "question": ...,
          "predicted": ..., "answers": [...],  # raw QASPER annotator answers
          "method": ..., "tree_layers": ..., "tree_nodes": ...,
        }

Then run `python -m eval_qasper.score_qasper` to compute the F1 metrics.
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

from .models import load_models, MODEL_TIERS


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def make_original_config(emb, summ, qa):
    return RetrievalAugmentationConfig(
        embedding_model=emb, summarization_model=summ, qa_model=qa,
        tb_max_tokens=100, tb_num_layers=3, tb_summarization_length=100,
        tr_top_k=10, tr_selection_mode="top_k",
    )


def _make_tree_config(clusterer, emb, summ):
    return ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={},
        reduction_dimension=10,
        summarization_model=summ,
        embedding_models={"EMB": emb},
        cluster_embedding_model="EMB",
        max_tokens=100, num_layers=3, summarization_length=100,
    )


def _wrap_config(tree_config, emb, qa):
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config, qa_model=qa, embedding_model=emb,
        tr_top_k=10, tr_selection_mode="top_k",
    )


def make_gmm_config(emb, summ, qa):
    clusterer = GMMClusterer(
        reduction_dimension=10, soft_threshold=0.1,
        force_hard_clustering=False, random_state=224,
    )
    return _wrap_config(_make_tree_config(clusterer, emb, summ), emb, qa)


def make_kmeans_config(emb, summ, qa):
    clusterer = KMeansClusterer(
        k_strategy="silhouette", min_k=3, max_k=15, random_state=224,
        reduce_embeddings=True, reduction_dimension=10,   # <-- add
    )
    return _wrap_config(_make_tree_config(clusterer, emb, summ), emb, qa)

def make_agglomerative_config(emb, summ, qa):
    clusterer = AgglomerativeClusterer(
        cut_strategy="silhouette", linkage="average",
        min_k=3, max_k=15, random_state=224,
        reduce_embeddings=True, reduction_dimension=10,   # <-- add
    )
    return _wrap_config(_make_tree_config(clusterer, emb, summ), emb, qa)

def make_dbscan_config(emb, summ, qa):
    clusterer = DBSCANClusterer(
        noise_strategy="nearest", min_samples=5,
        eps_percentile=90, random_state=224,
        reduce_embeddings=True, reduction_dimension=10,   # <-- add
    )
    return _wrap_config(_make_tree_config(clusterer, emb, summ), emb, qa)

def make_leiden_config(emb, summ, qa):
    lcfg = LeidenConfig(k_neighbors=15, use_adjacency_edges=True, adjacency_weight=0.5,
        resolution=1.0, resolution_schedule={0: 1.2, 1: 0.8},
        partition_type="RBConfiguration", min_cluster_size=1)
    clusterer = LeidenClusterer(config=lcfg, random_state=224,
        reduce_embeddings=True, reduction_dimension=10)   # <-- add
    return _wrap_config(_make_tree_config(clusterer, emb, summ), emb, qa)

def make_flat(emb, summ, qa):
    return FlatRetriever(embedding_model=emb, qa_model=qa, top_k=10, chunk_size=100)


METHODS = {
    "flat":          (make_flat,                  "Flat SBERT retrieval (no tree)"),
    "original":      (make_original_config,       "RAPTOR GMM+UMAP upstream"),
    "gmm":           (make_gmm_config,            "GMMClusterer new interface"),
    "leiden":        (make_leiden_config,         "LeidenClusterer (k-NN graph)"),
    "kmeans":        (make_kmeans_config,         "KMeansClusterer (silhouette K)"),
    "agglomerative": (make_agglomerative_config,  "AgglomerativeClusterer"),
    "dbscan":        (make_dbscan_config,         "DBSCANClusterer (auto-eps)"),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_retriever(method_name: str, emb, summ, qa):
    """Build a retriever (RA or FlatRetriever) for the given method."""
    factory_fn, _ = METHODS[method_name]
    result = factory_fn(emb, summ, qa)
    # FlatRetriever returns itself; RAPTOR factories return a config
    if hasattr(result, "add_documents"):
        return result
    return RetrievalAugmentation(config=result)


def evaluate_method(
    method_name: str,
    papers: List[Dict],
    emb, summ, qa,
    out_path: str,
    verbose: bool = False,
) -> Dict:
    """Run all questions for all papers through one clustering method."""
    description = METHODS[method_name][1]
    print(f"\n{'=' * 70}")
    print(f"  Method: {method_name} — {description}")
    print(f"{'=' * 70}")

    n_papers_done = 0
    n_questions_done = 0
    total_build_time = 0.0
    failures = 0

    with open(out_path, "w") as f_out:
        for pi, paper in enumerate(papers):
            paper_id = paper["paper_id"]
            text = paper["full_text"]
            questions = paper["questions"]

            try:
                ra = build_retriever(method_name, emb, summ, qa)
                t0 = time.time()
                ra.add_documents(text)
                build_time = time.time() - t0
                total_build_time += build_time
            except Exception as exc:
                print(f"  [{pi+1}/{len(papers)}] {paper_id}: BUILD FAILED — {exc}")
                failures += 1
                continue

            tree = ra.tree
            n_layers = getattr(tree, "num_layers", 0)
            n_nodes = len(getattr(tree, "all_nodes", {}))
            n_leaves = len(getattr(tree, "leaf_nodes", {}))

            print(f"  [{pi+1}/{len(papers)}] {paper_id}: "
                  f"{paper['n_words']} words → {n_layers} layers, {n_nodes} nodes "
                  f"({n_leaves} leaves) in {build_time:.1f}s, "
                  f"answering {len(questions)} questions...")

            for q in questions:
                qid = q["question_id"]
                question = q["question"]
                try:
                    predicted = ra.answer_question(question=question)
                except Exception as exc:
                    if verbose:
                        print(f"     Q {qid}: ERROR — {exc}")
                    predicted = ""

                record = {
                    "method": method_name,
                    "paper_id": paper_id,
                    "question_id": qid,
                    "question": question,
                    "predicted": predicted,
                    "answers": q["answers"],  # raw QASPER annotator answers
                    "tree_layers": n_layers,
                    "tree_nodes": n_nodes,
                    "build_time_sec": round(build_time, 2),
                }
                f_out.write(json.dumps(record) + "\n")
                n_questions_done += 1

            n_papers_done += 1

    print(f"\n  {method_name}: {n_papers_done} papers, {n_questions_done} questions "
          f"in {total_build_time:.1f}s (build), {failures} failures")
    return {
        "method": method_name,
        "n_papers": n_papers_done,
        "n_questions": n_questions_done,
        "total_build_time_sec": round(total_build_time, 2),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int = 224):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run QASPER evaluation across clustering methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model tiers:\n" + "\n".join(
            f"  {n:14s} {t['description']}" for n, t in MODEL_TIERS.items()
        ),
    )
    parser.add_argument("--data", default="data/qasper/validation.json",
                        help="Preprocessed QASPER JSON path.")
    parser.add_argument("--model-tier", default="base", choices=list(MODEL_TIERS.keys()))
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()))
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Limit to first N papers from the preprocessed file.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save predictions. Default: experiments/qasper/<timestamp>")
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("  QASPER Evaluation")
    print(f"  Data: {args.data}")
    print(f"  Tier: {args.model_tier}")
    print(f"  Methods: {', '.join(args.methods)}")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found.")
        print("Run preprocessing first: python -m eval_qasper.preprocess_qasper")
        sys.exit(1)

    with open(args.data) as f:
        papers = json.load(f)
    if args.max_papers:
        papers = papers[:args.max_papers]

    n_questions = sum(len(p["questions"]) for p in papers)
    print(f"\n  Loaded {len(papers)} papers, {n_questions} questions total")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or os.path.join("experiments", "qasper", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output: {out_dir}")

    # Load models once
    print("\nLoading models...")
    emb, summ, qa = load_models(args.model_tier)
    print("Models ready.")

    # Run each method
    run_meta = []
    for method_name in args.methods:
        out_path = os.path.join(out_dir, f"predictions_{method_name}.jsonl")
        set_seed(args.seed)  # reset per method for reproducibility
        result = evaluate_method(
            method_name, papers, emb, summ, qa, out_path, verbose=args.verbose,
        )
        result["output"] = out_path
        run_meta.append(result)

    # Save run metadata
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "model_tier": args.model_tier,
            "n_papers": len(papers),
            "n_questions_total": n_questions,
            "seed": args.seed,
            "methods": run_meta,
        }, f, indent=2)

    print(f"\nDone. Predictions saved to {out_dir}")
    print(f"\nRun scoring: python -m eval_qasper.score_qasper {out_dir}")


if __name__ == "__main__":
    main()
