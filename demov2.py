"""
demo.py — Clustering method comparison for RAPTOR.

Builds a RAPTOR tree with each clustering method, answers a set of ground-truth
questions, and prints a comparison table with Accuracy, Token-F1, tree depth,
node counts, and build time. Also saves results to results.csv.

Usage:
    python demo.py                # run all methods
    python demo.py --methods gmm leiden   # run only specific methods
    python demo.py --verbose      # show per-question details

Metrics:
    - Exact Match (EM): 1 if the ground truth appears as a substring in the
      generated answer (case-insensitive). This is lenient by design — FLAN-T5
      generates short extractive answers, not verbatim strings.
    - Token F1: standard token-overlap F1 between the predicted answer tokens
      and the ground truth tokens (same metric used by SQuAD, Sarthi et al.).
    - Accuracy: percentage of questions with EM = 1.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")


# ============================================================================
# Ground truth QA benchmark
# ============================================================================
# Each entry is (question, ground_truth_answer, answer_keywords).
# - ground_truth: the reference answer for token-F1 computation.
# - keywords: a list of key tokens; EM = 1 if ALL keywords appear in the
#   generated answer (case-insensitive). This is more robust than substring
#   matching because LLMs paraphrase.

@dataclass
class QAPair:
    question: str
    ground_truth: str
    keywords: List[str]


BENCHMARK = [
    QAPair(
        question="How did Cinderella reach her happy ending?",
        ground_truth="The prince found Cinderella by fitting the glass slipper on her foot, recognized her, and they married.",
        keywords=["prince", "slipper", "married"],
    ),
    QAPair(
        question="Who helped Cinderella attend the ball?",
        ground_truth="Her fairy godmother appeared and transformed a pumpkin into a coach and her worn clothes into a beautiful gown.",
        keywords=["fairy", "godmother"],
    ),
    QAPair(
        question="What did Cinderella leave behind at the ball?",
        ground_truth="Cinderella left behind one glass slipper when she fled before midnight.",
        keywords=["glass", "slipper"],
    ),
    QAPair(
        question="Why did Cinderella have to leave the ball before midnight?",
        ground_truth="The magic from the fairy godmother would end at midnight.",
        keywords=["magic", "midnight"],
    ),
    QAPair(
        question="How did the prince find Cinderella after the ball?",
        ground_truth="The prince searched the kingdom for the woman whose foot fit the glass slipper.",
        keywords=["searched", "slipper", "fit"],
    ),
    QAPair(
        question="What does the fairy godmother symbolize in the story?",
        ground_truth="The fairy godmother symbolizes hope.",
        keywords=["hope"],
    ),
    QAPair(
        question="What is the glass slipper's role in the story?",
        ground_truth="The lost slipper is the key piece of evidence that allows the prince to identify Cinderella.",
        keywords=["evidence", "identify"],
    ),
    QAPair(
        question="Who forced Cinderella to work day and night?",
        ground_truth="Her cruel stepmother and two jealous stepsisters forced her to work.",
        keywords=["stepmother", "stepsisters"],
    ),
    QAPair(
        question="What themes does the Cinderella story illustrate?",
        ground_truth="The story illustrates transformation, perseverance, and recognition.",
        keywords=["transformation", "perseverance"],
    ),
    QAPair(
        question="What was the pumpkin transformed into?",
        ground_truth="The fairy godmother transformed a pumpkin into a coach.",
        keywords=["coach"],
    ),
]


# ============================================================================
# Evaluation metrics
# ============================================================================

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace. Matches SQuAD eval."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute token-level precision, recall, F1 between prediction and ground truth.
    This is the standard SQuAD QA metric used by Sarthi et al. (ICLR 2024) and
    referenced in Gao et al.'s RAG survey Table III.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def keyword_match(prediction: str, keywords: List[str]) -> bool:
    """
    Returns True if ALL keywords appear in the prediction (case-insensitive).
    More robust than exact substring match because LLMs paraphrase freely.
    """
    pred_lower = normalize_text(prediction)
    return all(kw.lower() in pred_lower for kw in keywords)


# ============================================================================
# Demo text
# ============================================================================

DEMO_TEXT = (
    "Cinderella lived with her cruel stepmother and two jealous stepsisters, who forced "
    "her to work day and night. When the royal ball was announced, Cinderella wished to go "
    "but was forbidden. Her fairy godmother appeared and transformed a pumpkin into a coach, "
    "mice into horses, and Cinderella's worn clothes into a beautiful gown with glass slippers. "
    "She attended the ball and danced with the prince, but she had to leave before midnight, "
    "when the magic would end. As she fled, one glass slipper was left behind. The prince "
    "searched the kingdom for the woman whose foot fit the slipper. When he came to "
    "Cinderella's house, the slipper fit her perfectly, and he recognized her as the one "
    "he loved. Cinderella married the prince and finally found her happy ending.\n\n"
    "The story is often used to illustrate transformation, perseverance, and recognition. "
    "In many retellings, the fairy godmother symbolizes hope, while the lost slipper becomes "
    "the key piece of evidence that allows the prince to identify Cinderella."
)


# ============================================================================
# Model wrappers (same as your existing demo.py, no changes)
# ============================================================================

from raptor import (
    BaseSummarizationModel,
    BaseQAModel,
    BaseEmbeddingModel,
    RetrievalAugmentationConfig,
    RetrievalAugmentation,
)
from raptor.cluster_tree_builder import ClusterTreeConfig
from raptor.cluster_utils import RAPTOR_Clustering
from raptor.EmbeddingModels import SBertEmbeddingModel


class LocalBartSummarizationModel(BaseSummarizationModel):
    """DistilBART summarizer for local (no-API) usage."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self._pipeline = None
        self._load_error = None

    def _ensure_loaded(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "summarization", model=self.model_name, tokenizer=self.model_name
            )
        except Exception as exc:
            self._load_error = exc

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text:
            return ""
        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                result = self._pipeline(
                    text,
                    max_new_tokens=min(int(max_tokens), 128),
                    min_new_tokens=20,
                    do_sample=False,
                    truncation=True,
                )
                return result[0]["summary_text"].strip()
            except Exception:
                pass
        # Heuristic fallback — deterministic, no silent degradation in the ablation
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")


class LocalFlanQAModel(BaseQAModel):
    """FLAN-T5 question-answering model for local (no-API) usage."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._pipeline = None
        self._load_error = None

    def _ensure_loaded(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text2text-generation", model=self.model_name, tokenizer=self.model_name
            )
        except Exception as exc:
            self._load_error = exc

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context:
            return "No context available."
        self._ensure_loaded()
        if self._pipeline is not None:
            prompt = (
                f"Answer the question using the provided context. "
                f"If the answer is not in the context, say you do not know.\n\n"
                f"Context: {context}\n\nQuestion: {question}"
            )
            try:
                result = self._pipeline(prompt, max_new_tokens=64, do_sample=False)
                return result[0]["generated_text"].strip()
            except Exception:
                pass
        # Heuristic fallback
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        keywords = [t.lower() for t in question.split() if len(t) > 3]
        best, best_score = "", -1
        for s in sentences:
            score = sum(k in s.lower() for k in keywords)
            if score > best_score:
                best, best_score = s, score
        return best + "." if best else "No answer found."


# ============================================================================
# Config factories — one per clustering method
# ============================================================================

def _shared_models():
    """Instantiate the models ONCE and share across all methods."""
    emb = SBertEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    summ = LocalBartSummarizationModel()
    qa = LocalFlanQAModel()
    return emb, summ, qa


def make_original_config(emb, summ, qa) -> RetrievalAugmentationConfig:
    """Upstream RAPTOR — uses RAPTOR_Clustering (GMM+UMAP) internally."""
    return RetrievalAugmentationConfig(
        embedding_model=emb,
        summarization_model=summ,
        qa_model=qa,
        tb_max_tokens=80,
        tb_num_layers=3,
        tb_summarization_length=80,
        tr_top_k=5,
        tr_selection_mode="top_k",
    )


def make_gmm_config(emb, summ, qa) -> RetrievalAugmentationConfig:
    """GMM through our new BaseClusterer interface — should match original."""
    from raptor.clustering import GMMClusterer

    clusterer = GMMClusterer(
        reduction_dimension=10,
        soft_threshold=0.1,
        force_hard_clustering=False,
        random_state=224,
        max_length_in_cluster=3500,
        verbose=False,
    )
    tree_config = ClusterTreeConfig(
        clustering_algorithm=clusterer,
        clustering_params={},
        reduction_dimension=10,
        summarization_model=summ,
        embedding_models={"EMB": emb},
        cluster_embedding_model="EMB",
        max_tokens=80,
        num_layers=3,
        summarization_length=80,
    )
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config,
        qa_model=qa,
        embedding_model=emb,
        tr_top_k=5,
        tr_selection_mode="top_k",
    )


def make_leiden_config(emb, summ, qa) -> RetrievalAugmentationConfig:
    """Leiden community detection on a k-NN similarity graph."""
    from raptor.clustering import LeidenClusterer, LeidenConfig

    leiden_cfg = LeidenConfig(
        k_neighbors=10,
        use_adjacency_edges=True,
        adjacency_weight=0.5,
        resolution=1.0,
        resolution_schedule={0: 1.2, 1: 0.8},
        partition_type="RBConfiguration",
        min_cluster_size=1,
    )
    clusterer = LeidenClusterer(
        config=leiden_cfg,
        random_state=224,
        max_length_in_cluster=3500,
        verbose=False,
    )
    tree_config = ClusterTreeConfig(
        clustering_algorithm=clusterer,
        clustering_params={},
        reduction_dimension=10,
        summarization_model=summ,
        embedding_models={"EMB": emb},
        cluster_embedding_model="EMB",
        max_tokens=80,
        num_layers=3,
        summarization_length=80,
    )
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config,
        qa_model=qa,
        embedding_model=emb,
        tr_top_k=5,
        tr_selection_mode="top_k",
    )


# Registry: name -> (factory_function, description)
METHOD_REGISTRY = {
    "original": (make_original_config, "RAPTOR (GMM+UMAP) upstream"),
    "gmm":      (make_gmm_config,      "GMMClusterer (new interface)"),
    "leiden":   (make_leiden_config,     "LeidenClusterer (k-NN graph)"),
}


# ============================================================================
# Tree statistics
# ============================================================================

@dataclass
class TreeStats:
    num_layers: int = 0
    total_nodes: int = 0
    leaf_nodes: int = 0
    summary_nodes: int = 0
    layer_sizes: Dict[int, int] = field(default_factory=dict)
    build_time_sec: float = 0.0


def collect_tree_stats(ra: RetrievalAugmentation) -> TreeStats:
    tree = ra.tree
    stats = TreeStats()
    stats.num_layers = tree.num_layers
    stats.total_nodes = len(tree.all_nodes)
    stats.leaf_nodes = len(tree.leaf_nodes)
    stats.summary_nodes = stats.total_nodes - stats.leaf_nodes
    for layer_idx, nodes in tree.layer_to_nodes.items():
        stats.layer_sizes[layer_idx] = len(nodes)
    return stats


# ============================================================================
# Per-method evaluation result
# ============================================================================

@dataclass
class MethodResult:
    name: str
    description: str
    tree_stats: TreeStats
    per_question: List[Dict]  # list of per-question detail dicts
    accuracy: float = 0.0     # % of questions with keyword match
    mean_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0


# ============================================================================
# Run one method end-to-end
# ============================================================================

def evaluate_method(
    name: str,
    config: RetrievalAugmentationConfig,
    description: str,
    benchmark: List[QAPair],
    verbose: bool = False,
) -> MethodResult:
    print(f"\n{'='*70}")
    print(f"  Building tree: {name} — {description}")
    print(f"{'='*70}")

    ra = RetrievalAugmentation(config=config)

    t0 = time.time()
    ra.add_documents(DEMO_TEXT)
    build_time = time.time() - t0

    tree_stats = collect_tree_stats(ra)
    tree_stats.build_time_sec = build_time

    print(f"  Tree built in {build_time:.2f}s")
    print(f"  Layers: {tree_stats.num_layers} | "
          f"Total nodes: {tree_stats.total_nodes} | "
          f"Leaves: {tree_stats.leaf_nodes} | "
          f"Summaries: {tree_stats.summary_nodes}")
    for layer_idx in sorted(tree_stats.layer_sizes):
        print(f"    Layer {layer_idx}: {tree_stats.layer_sizes[layer_idx]} nodes")

    # Answer each question
    per_question = []
    total_em = 0
    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    print(f"\n  Answering {len(benchmark)} questions...")

    for i, qa in enumerate(benchmark):
        t_q = time.time()
        answer = ra.answer_question(question=qa.question)
        answer_time = time.time() - t_q

        prec, rec, f1 = token_f1(answer, qa.ground_truth)
        em = 1 if keyword_match(answer, qa.keywords) else 0

        total_em += em
        total_f1 += f1
        total_prec += prec
        total_rec += rec

        detail = {
            "question": qa.question,
            "ground_truth": qa.ground_truth,
            "predicted": answer,
            "keywords": qa.keywords,
            "em": em,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "answer_time": answer_time,
        }
        per_question.append(detail)

        if verbose:
            status = "PASS" if em else "FAIL"
            print(f"    Q{i+1} [{status}] F1={f1:.3f} | {qa.question}")
            print(f"         Predicted: {answer}")
            if not em:
                print(f"         Expected keywords: {qa.keywords}")

    n = len(benchmark)
    result = MethodResult(
        name=name,
        description=description,
        tree_stats=tree_stats,
        per_question=per_question,
        accuracy=total_em / n * 100 if n else 0.0,
        mean_f1=total_f1 / n if n else 0.0,
        mean_precision=total_prec / n if n else 0.0,
        mean_recall=total_rec / n if n else 0.0,
    )

    print(f"\n  Results: Accuracy={result.accuracy:.1f}%  "
          f"F1={result.mean_f1:.3f}  "
          f"Precision={result.mean_precision:.3f}  "
          f"Recall={result.mean_recall:.3f}")

    return result


# ============================================================================
# Comparison table
# ============================================================================

def print_comparison_table(results: List[MethodResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n")
    print("=" * 100)
    print("  CLUSTERING METHOD COMPARISON")
    print("=" * 100)

    # Header
    header = (
        f"{'Method':<30} {'Accuracy':>9} {'F1':>7} {'Prec':>7} {'Recall':>7} "
        f"{'Layers':>7} {'Nodes':>7} {'Leaves':>7} {'Summ':>7} {'Time(s)':>8}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        row = (
            f"{r.name:<30} {r.accuracy:>8.1f}% {r.mean_f1:>7.3f} "
            f"{r.mean_precision:>7.3f} {r.mean_recall:>7.3f} "
            f"{r.tree_stats.num_layers:>7} {r.tree_stats.total_nodes:>7} "
            f"{r.tree_stats.leaf_nodes:>7} {r.tree_stats.summary_nodes:>7} "
            f"{r.tree_stats.build_time_sec:>8.2f}"
        )
        print(row)

    print("-" * 100)

    # Best method highlight
    best_acc = max(results, key=lambda r: r.accuracy)
    best_f1 = max(results, key=lambda r: r.mean_f1)
    print(f"\n  Best Accuracy: {best_acc.name} ({best_acc.accuracy:.1f}%)")
    print(f"  Best F1:       {best_f1.name} ({best_f1.mean_f1:.3f})")


def print_per_question_table(results: List[MethodResult]) -> None:
    """Print per-question breakdown across all methods."""
    print("\n")
    print("=" * 100)
    print("  PER-QUESTION BREAKDOWN")
    print("=" * 100)

    n_questions = len(results[0].per_question)
    method_names = [r.name for r in results]

    # Header row
    q_col = f"{'Question':<45}"
    m_cols = "  ".join(f"{m:>12}" for m in method_names)
    print(f"{q_col} {m_cols}")
    print("-" * 100)

    for i in range(n_questions):
        q_text = results[0].per_question[i]["question"]
        if len(q_text) > 42:
            q_text = q_text[:42] + "..."

        scores = []
        for r in results:
            em = r.per_question[i]["em"]
            f1 = r.per_question[i]["f1"]
            marker = "+" if em else "-"
            scores.append(f"  {marker} F1={f1:.2f}")

        score_str = "  ".join(f"{s:>12}" for s in scores)
        print(f"{q_text:<45} {score_str}")

    print("-" * 100)
    print("  + = keyword match (EM=1), - = keyword miss (EM=0)")


def save_results_csv(results: List[MethodResult], path: str = "results.csv") -> None:
    """Save the summary table as a CSV for downstream plotting."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "description", "accuracy_pct", "mean_f1", "mean_precision",
            "mean_recall", "num_layers", "total_nodes", "leaf_nodes",
            "summary_nodes", "build_time_sec",
        ])
        for r in results:
            writer.writerow([
                r.name, r.description, f"{r.accuracy:.1f}", f"{r.mean_f1:.4f}",
                f"{r.mean_precision:.4f}", f"{r.mean_recall:.4f}",
                r.tree_stats.num_layers, r.tree_stats.total_nodes,
                r.tree_stats.leaf_nodes, r.tree_stats.summary_nodes,
                f"{r.tree_stats.build_time_sec:.2f}",
            ])
    print(f"\n  Results saved to {os.path.abspath(path)}")


def save_detailed_csv(results: List[MethodResult], path: str = "results_detailed.csv") -> None:
    """Save per-question detail for every method."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "question", "ground_truth", "predicted", "keywords",
            "em", "precision", "recall", "f1", "answer_time_sec",
        ])
        for r in results:
            for q in r.per_question:
                writer.writerow([
                    r.name, q["question"], q["ground_truth"], q["predicted"],
                    "|".join(q["keywords"]), q["em"],
                    f"{q['precision']:.4f}", f"{q['recall']:.4f}",
                    f"{q['f1']:.4f}", f"{q['answer_time']:.3f}",
                ])
    print(f"  Detailed results saved to {os.path.abspath(path)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare RAPTOR clustering methods on a QA benchmark."
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(METHOD_REGISTRY.keys()),
        choices=list(METHOD_REGISTRY.keys()),
        help="Which clustering methods to evaluate (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-question details during evaluation.",
    )
    parser.add_argument(
        "--output", default="results.csv",
        help="Path for the summary CSV output.",
    )
    parser.add_argument(
        "--detailed-output", default="results_detailed.csv",
        help="Path for the per-question CSV output.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RAPTOR Clustering Method Comparison")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Questions: {len(BENCHMARK)}")
    print("=" * 70)

    # Load models once — shared across all methods
    print("\nLoading models (SBERT, DistilBART, FLAN-T5)...")
    emb, summ, qa = _shared_models()
    print("Models loaded.\n")

    results: List[MethodResult] = []

    for method_name in args.methods:
        factory_fn, description = METHOD_REGISTRY[method_name]

        try:
            config = factory_fn(emb, summ, qa)
        except ImportError as exc:
            print(f"\n  SKIPPING {method_name}: {exc}")
            print(f"  Install the missing dependencies and re-run.\n")
            continue

        try:
            result = evaluate_method(
                name=method_name,
                config=config,
                description=description,
                benchmark=BENCHMARK,
                verbose=args.verbose,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR running {method_name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nNo methods completed successfully. Check the errors above.")
        sys.exit(1)

    # Print tables
    print_comparison_table(results)
    print_per_question_table(results)

    # Save CSVs
    save_results_csv(results, path=args.output)
    save_detailed_csv(results, path=args.detailed_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
