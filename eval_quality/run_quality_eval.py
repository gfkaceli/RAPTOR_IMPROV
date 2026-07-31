"""
run_quality_eval.py — Main QuALITY evaluation script.

For each (clustering method, article):
  1. Build a RAPTOR tree from the article passage
  2. For each question: retrieve top-k with the QUESTION ONLY (options are NOT
     added to the retrieval query, to avoid leaking answer text into retrieval
     and masking clustering differences), then ask the model to select an option
     using only the retrieved context.
  3. Log the predicted option, the retrieved node layers (leaf vs summary), and
     tree stats, so downstream scoring can produce accuracy, HARD/EASY breakdown,
     and a retrieved-layer distribution.

Design choices (see thesis Section on QuALITY):
  - Question-only retrieval keeps retrieval quality dependent on tree structure,
    which is the independent variable. Per-option retrieval would leak answers.
  - Multiple-choice accuracy replaces token-F1; the metric lives in
    quality_metric.py.
  - The summarization prompt is switched to a narrative-appropriate variant,
    because QuALITY passages are fiction/journalism, not scientific papers.

Usage:
    python -m eval_quality.preprocess_quality --input <QuALITY dev jsonl> --max-articles 15
    python -m eval_quality.run_quality_eval --model-tier base --max-articles 15 \
        --methods flat original gmm leiden kmeans agglomerative dbscan
    python -m eval_quality.score_quality experiments/quality/<timestamp>
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

# Dedicated QuALITY models: multiple-choice QA + narrative summarization.
from eval_quality.models import load_models, MODEL_TIERS


# ---------------------------------------------------------------------------
# Multiple-choice answering
# ---------------------------------------------------------------------------
# The MC prompt formatting and option-letter answering now live in
# eval_quality.models (LocalQAModel.answer_multiple_choice /
# OpenAIQAModel.answer_multiple_choice). The runner simply calls that method,
# so the prompt is defined in exactly one place.


def answer_mc(qa_model, context: str, question: str, options: List[str]) -> str:
    """
    Ask the QA model to select an option. Both the local and OpenAI QA models
    expose answer_multiple_choice; we call it directly. A defensive fallback
    covers any custom QA model that only implements answer_question.
    """
    if hasattr(qa_model, "answer_multiple_choice"):
        return qa_model.answer_multiple_choice(context, question, options)
    # Fallback: fold options into a plain QA call.
    letters = ["A", "B", "C", "D", "E", "F"][:len(options)]
    opt_lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, options))
    return qa_model.answer_question(
        context, f"{question}\n\nOptions:\n{opt_lines}\n\nAnswer with one letter."
    )


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------
# Note: the narrative-appropriate summarization prompt is now built into
# eval_quality.models.LocalSummarizationModel (and the OpenAI summarizer), so no
# runtime prompt-patching is needed here.


def make_original_config(emb, summ, qa):
    return RetrievalAugmentationConfig(
        embedding_model=emb, summarization_model=summ, qa_model=qa,
        tb_max_tokens=140, tb_num_layers=4, tb_summarization_length=600,
        tr_top_k=10, tr_selection_mode="top_k",
    )


def _tree_cfg(clusterer, emb, summ):
    return ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={}, reduction_dimension=10,
        summarization_model=summ, embedding_models={"EMB": emb},
        cluster_embedding_model="EMB", max_tokens=140, num_layers=4,
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
                        resolution=1.0, resolution_schedule={0: 1.3, 1: 0.9, 2: 0.6},
                        partition_type="RBConfiguration", min_cluster_size=1)
    c = LeidenClusterer(config=lcfg, random_state=224,
                        reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_kmeans(emb, summ, qa):
    c = KMeansClusterer(k_strategy="silhouette", min_k=3, max_k=15, random_state=224,
                        reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_agglomerative(emb, summ, qa):
    c = AgglomerativeClusterer(cut_strategy="silhouette", linkage="average",
                               min_k=3, max_k=15, random_state=224,
                               reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_dbscan(emb, summ, qa):
    c = DBSCANClusterer(noise_strategy="nearest", min_samples=4, eps_percentile=88,
                        random_state=224, reduce_embeddings=True, reduction_dimension=10)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_flat(emb, summ, qa):
    return FlatRetriever(embedding_model=emb, qa_model=qa, top_k=10, chunk_size=100)


METHODS = {
    "flat":          (make_flat,          "Flat SBERT retrieval (no tree)"),
    "original":      (make_original_config, "RAPTOR GMM+UMAP upstream"),
    "gmm":           (make_gmm,           "GMMClusterer"),
    "leiden":        (make_leiden,        "LeidenClusterer (reduced)"),
    "kmeans":        (make_kmeans,        "KMeansClusterer (reduced)"),
    "agglomerative": (make_agglomerative, "AgglomerativeClusterer (reduced)"),
    "dbscan":        (make_dbscan,        "DBSCANClusterer (reduced)"),
}


# ---------------------------------------------------------------------------
# Retrieval with layer logging
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


def build_retriever(method_name, emb, summ, qa):
    factory_fn, _ = METHODS[method_name]
    result = factory_fn(emb, summ, qa)
    if hasattr(result, "add_documents"):
        return result
    return RetrievalAugmentation(config=result)


def evaluate_method(method_name, articles, emb, summ, qa, out_path, verbose=False):
    description = METHODS[method_name][1]
    print(f"\n{'='*70}\n  {method_name} — {description}\n{'='*70}")

    n_articles = n_questions = 0
    total_build = 0.0
    failures = 0

    with open(out_path, "w") as f_out:
        for ai, art in enumerate(articles):
            aid = art["article_id"]
            text = art["full_text"]
            questions = art["questions"]
            try:
                ra = build_retriever(method_name, emb, summ, qa)
                t0 = time.time()
                ra.add_documents(text)
                build_time = time.time() - t0
                total_build += build_time
            except Exception as exc:
                print(f"  [{ai+1}/{len(articles)}] {aid}: BUILD FAILED — {exc}")
                failures += 1
                continue

            tree = ra.tree
            n_layers = getattr(tree, "num_layers", 0)
            n_nodes = len(getattr(tree, "all_nodes", {}))
            print(f"  [{ai+1}/{len(articles)}] {aid}: {art['n_words']} words -> "
                  f"{n_layers} layers, {n_nodes} nodes in {build_time:.1f}s, "
                  f"{len(questions)} questions")

            for q in questions:
                try:
                    context, layers = retrieve_with_layers(ra, q["question"])
                    predicted = answer_mc(qa, context, q["question"], q["options"])
                except Exception as exc:
                    if verbose:
                        print(f"      Q {q['question_id']}: ERROR {exc}")
                    predicted, layers = "", []

                f_out.write(json.dumps({
                    "method": method_name,
                    "article_id": aid,
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "options": q["options"],
                    "gold_label": q["gold_label"],
                    "difficult": q["difficult"],
                    "predicted": predicted,
                    "retrieved_layers": layers,
                    "tree_layers": n_layers,
                    "tree_nodes": n_nodes,
                    "build_time_sec": round(build_time, 2),
                }) + "\n")
                n_questions += 1
            n_articles += 1

    print(f"\n  {method_name}: {n_articles} articles, {n_questions} questions, "
          f"{total_build:.1f}s build, {failures} failures")
    return {"method": method_name, "n_articles": n_articles,
            "n_questions": n_questions, "build_time_sec": round(total_build, 2),
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
    p = argparse.ArgumentParser(description="Run QuALITY multiple-choice evaluation.")
    p.add_argument("--data", default="data/quality/dev.json")
    p.add_argument("--model-tier", default="base", choices=list(MODEL_TIERS.keys()))
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=224)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print("  QuALITY Multiple-Choice Evaluation")
    print(f"  Data: {args.data} | Tier: {args.model_tier} | Methods: {', '.join(args.methods)}")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found. Run preprocess_quality first.")
        sys.exit(1)

    with open(args.data) as f:
        articles = json.load(f)
    if args.max_articles:
        articles = articles[:args.max_articles]

    n_q = sum(len(a["questions"]) for a in articles)
    n_hard = sum(1 for a in articles for q in a["questions"] if q["difficult"] == 1)
    n_labeled = sum(1 for a in articles for q in a["questions"] if q["gold_label"] is not None)
    print(f"\n  {len(articles)} articles, {n_q} questions "
          f"({n_hard} HARD, {n_labeled} labeled)")
    if n_labeled == 0:
        print("  ERROR: no gold labels — cannot score. Use the dev split.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or os.path.join("experiments", "quality", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output: {out_dir}")

    print("\nLoading models...")
    emb, summ, qa = load_models(args.model_tier)
    print("Models ready (multiple-choice QA + narrative summarization).")

    run_meta = []
    for m in args.methods:
        out_path = os.path.join(out_dir, f"predictions_{m}.jsonl")
        set_seed(args.seed)
        run_meta.append(evaluate_method(m, articles, emb, summ, qa, out_path,
                                        verbose=args.verbose))

    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({"timestamp": ts, "model_tier": args.model_tier,
                   "n_articles": len(articles), "n_questions_total": n_q,
                   "n_hard": n_hard, "seed": args.seed, "methods": run_meta}, f, indent=2)

    print(f"\nDone. Run scoring: python -m eval_quality.score_quality {out_dir}")


if __name__ == "__main__":
    main()