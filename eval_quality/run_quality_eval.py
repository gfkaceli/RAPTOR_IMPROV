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

# Reuse the QASPER model tiers/loaders — same models, different prompt.
from eval_qasper.models import load_models, MODEL_TIERS


# ---------------------------------------------------------------------------
# Multiple-choice prompting
# ---------------------------------------------------------------------------

MC_SYSTEM = (
    "You are answering a multiple-choice reading-comprehension question about a "
    "passage. You are given retrieved excerpts from the passage and a question "
    "with four options labelled A, B, C, D. Choose the single best option based "
    "only on the excerpts. Respond with just the letter (A, B, C, or D) and "
    "nothing else."
)


def format_mc_prompt(context: str, question: str, options: List[str]) -> str:
    letters = ["A", "B", "C", "D", "E", "F"][:len(options)]
    opt_lines = "\n".join(f"{L}. {opt}" for L, opt in zip(letters, options))
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Answer with a single letter ({'/'.join(letters)})."
    )


def answer_mc(qa_model, context: str, question: str, options: List[str]) -> str:
    """
    Ask the QA model to select an option. We reuse whatever QA model the tier
    provides. For local instruct models the generator exposes a .generate()
    with a system/user split; for OpenAI models we call answer_question with a
    combined prompt. Both return raw text that quality_metric parses.
    """
    user = format_mc_prompt(context, question, options)

    # Local instruct models expose the low-level generator via ._gen.
    gen = getattr(qa_model, "_gen", None)
    if gen is not None:
        # Short answer — one letter — so cap tokens tight and use answer cleaning.
        gen.max_new_tokens = 120
        return gen.generate(MC_SYSTEM, user, clean_mode="answer")

    # OpenAI-style models: fold the system instruction into the context arg.
    return qa_model.answer_question(f"{MC_SYSTEM}\n\n{context}",
                                    f"{question}\n\nOptions:\n" +
                                    "\n".join(f"{L}. {o}" for L, o in
                                              zip(['A','B','C','D'], options)))


# ---------------------------------------------------------------------------
# Config factories (narrative-appropriate summarization)
# ---------------------------------------------------------------------------

def _patch_narrative_summary(summ):
    """
    QuALITY passages are fiction/journalism. The default summarizer prompt is
    tuned for scientific papers ("preserve key facts, names, numerical results"),
    which under-weights narrative and thematic content that HARD questions probe.
    We swap in a narrative-appropriate system prompt when the summarizer exposes
    one (local instruct models). API summarizers keep their own prompt.
    """
    if hasattr(summ, "SYSTEM"):
        summ.SYSTEM = (
            "You are summarizing an excerpt from a story or article. Produce a "
            "concise summary that preserves the main events, characters, "
            "relationships, motivations, and themes, not only surface facts. "
            "Output only the summary."
        )
    return summ


def make_original_config(emb, summ, qa):
    return RetrievalAugmentationConfig(
        embedding_model=emb, summarization_model=summ, qa_model=qa,
        tb_max_tokens=100, tb_num_layers=4, tb_summarization_length=500,
        tr_top_k=8, tr_selection_mode="top_k",
    )


def _tree_cfg(clusterer, emb, summ):
    return ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={}, reduction_dimension=10,
        summarization_model=summ, embedding_models={"EMB": emb},
        cluster_embedding_model="EMB", max_tokens=100, num_layers=4,
        summarization_length=500,
    )


def _wrap(tree_config, emb, qa):
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config, qa_model=qa, embedding_model=emb,
        tr_top_k=8, tr_selection_mode="top_k",
    )


def make_gmm(emb, summ, qa):
    c = GMMClusterer(reduction_dimension=10, soft_threshold=0.1,
                     force_hard_clustering=False, random_state=224)
    return _wrap(_tree_cfg(c, emb, summ), emb, qa)


def make_leiden(emb, summ, qa):
    lcfg = LeidenConfig(k_neighbors=15, use_adjacency_edges=True, adjacency_weight=0.5,
                        resolution=1.0, resolution_schedule={0: 1.2, 1: 0.8},
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
    return FlatRetriever(embedding_model=emb, qa_model=qa, top_k=8, chunk_size=100)


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
    Retrieve context and, when possible, record the tree layer of each retrieved
    node. Returns (context_text, layer_list). layer_list is empty for the flat
    retriever (no tree) or if the RA object does not expose per-node layers.
    """
    context = ra.retrieve(question)
    layers: List[int] = []
    # FlatRetriever has num_layers=0 tree; everything is a leaf (layer 0).
    tree = getattr(ra, "tree", None)
    # We can't always map retrieved text back to node layers without deeper
    # hooks into RAPTOR's retriever; layer logging is best-effort. If the RA
    # exposes the last retrieved nodes, use them.
    last_nodes = getattr(ra, "_last_retrieved_nodes", None)
    if last_nodes and tree is not None:
        layer_of = {}
        for li, nodes in getattr(tree, "layer_to_nodes", {}).items():
            for nd in nodes:
                layer_of[getattr(nd, "index", id(nd))] = li
        for nd in last_nodes:
            layers.append(layer_of.get(getattr(nd, "index", id(nd)), -1))
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
    summ = _patch_narrative_summary(summ)
    print("Models ready (narrative summarization prompt applied).")

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
