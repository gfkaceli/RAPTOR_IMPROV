"""
eval_demo.py — Self-contained abstractive QA evaluation for RAPTOR clustering.

Uses a ~2500-word embedded document (History of Artificial Intelligence) with
15 hand-written abstractive QA pairs. No dataset download needed — everything
runs offline.

Why this design:
    - SQuAD is extractive (gold = verbatim span), which penalizes generative
      models like FLAN-T5 that paraphrase. Abstractive QA pairs expect natural
      language answers, matching how FLAN-T5 actually responds.
    - The document is ~2500 words → ~25 RAPTOR chunks → enough for a 2-3 layer
      tree where clustering granularity matters.
    - Questions span three types:
        LOCAL:  answerable from a single paragraph (tests retrieval precision)
        CROSS:  requires combining info from 2+ sections (tests hierarchical
                retrieval — summary nodes should help here)
        GLOBAL: requires understanding the whole document (tests whether
                tree structure captures document-level themes)

Metrics:
    - Token F1, Contains EM, Best-Window F1 (verbose-tolerant)
    - Retrieval Hit Rate (does the retrieved context contain the answer?)
    - ROUGE-L (standard for abstractive QA, used by Sarthi et al. on NarrativeQA)

Usage:
    python eval_demo.py                        # all methods
    python eval_demo.py --methods original     # just baseline
    python eval_demo.py --verbose              # show per-question detail
"""

from __future__ import annotations

import argparse
import os
import string
import sys
import time
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("OPENAI_API_KEY", "not-used-in-local-demo")


# ============================================================================
# Data loading — document and questions live in data/ directory
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOCUMENT_PATH = os.path.join(DATA_DIR, "document.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.json")


def load_document(path: str = DOCUMENT_PATH) -> str:
    """Load the evaluation document from a text file."""
    if not os.path.exists(path):
        print(f"ERROR: Document file not found at {path}")
        print(f"  Expected location: data/document.txt relative to this script.")
        print(f"  Create it or pass --document <path>.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"  Loaded document: {len(text)} chars, ~{len(text.split())} words")
    return text


def load_questions(path: str = QUESTIONS_PATH) -> List[Dict]:
    """
    Load QA pairs from a JSON file. Expected format:
    [
      {"question": "...", "answer": "...", "type": "local|cross|global"},
      ...
    ]
    """
    import json
    if not os.path.exists(path):
        print(f"ERROR: Questions file not found at {path}")
        print(f"  Expected location: data/questions.json relative to this script.")
        print(f"  Create it or pass --questions <path>.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    types = Counter(q["type"] for q in questions)
    print(f"  Loaded {len(questions)} questions: "
          + ", ".join(f"{n} {t}" for t, n in sorted(types.items())))
    return questions



# ============================================================================
# Metrics
# ============================================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = " ".join(w for w in text.split() if w not in {"a", "an", "the"})
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    pt = normalize_text(prediction).split()
    gt = normalize_text(ground_truth).split()
    if not pt and not gt: return 1.0, 1.0, 1.0
    if not pt or not gt: return 0.0, 0.0, 0.0
    common = Counter(pt) & Counter(gt)
    nc = sum(common.values())
    if nc == 0: return 0.0, 0.0, 0.0
    p, r = nc / len(pt), nc / len(gt)
    return p, r, 2 * p * r / (p + r)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """Lenient: True if any 3+ word phrase from the gold appears in the prediction."""
    gt_words = normalize_text(ground_truth).split()
    pred_norm = normalize_text(prediction)
    # Check all 3-grams from ground truth
    for i in range(len(gt_words) - 2):
        phrase = " ".join(gt_words[i:i + 3])
        if phrase in pred_norm:
            return True
    return False


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L F1 via longest common subsequence.
    Standard metric for abstractive QA — used by Sarthi et al. on NarrativeQA.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0

    m, n = len(gt_tokens), len(pred_tokens)
    # LCS via DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / n
    rec = lcs_len / m
    return 2 * prec * rec / (prec + rec)


def check_retrieval_hit(retrieved_text: str, answer: str) -> bool:
    """Check if key phrases from the answer appear in the retrieved context."""
    ret_lower = retrieved_text.lower()
    answer_words = answer.lower().split()
    # Check 4-grams for more robust matching
    hits = 0
    checks = 0
    for i in range(len(answer_words) - 3):
        phrase = " ".join(answer_words[i:i + 4])
        checks += 1
        if phrase in ret_lower:
            hits += 1
    if checks == 0:
        return answer.lower() in ret_lower
    return (hits / checks) > 0.2  # at least 20% of key phrases found


def score_prediction(prediction: str, ground_truth: str) -> Dict[str, float]:
    prec, rec, f1 = token_f1(prediction, ground_truth)
    cem = 1.0 if contains_match(prediction, ground_truth) else 0.0
    rl = rouge_l(prediction, ground_truth)
    return {
        "f1": f1, "precision": prec, "recall": rec,
        "contains_em": cem, "rouge_l": rl,
    }


# ============================================================================
# Model wrappers
# ============================================================================

from raptor import (
    BaseSummarizationModel, BaseQAModel,
    RetrievalAugmentationConfig, RetrievalAugmentation,
)
from raptor.cluster_tree_builder import ClusterTreeConfig
from raptor.EmbeddingModels import SBertEmbeddingModel


# ============================================================================
# Model tiers
# ============================================================================
# Three tiers, selected via --model-tier:
#
#   base        FLAN-T5-base (770M) + DistilBART + MiniLM embeddings
#               Runs on any machine with 4GB RAM. Weakest QA quality.
#
#   local-large FLAN-T5-large (3B) + BART-large-CNN + mpnet embeddings
#               Needs ~8GB RAM / GPU. Much better QA quality.
#               Alternatively uses Mistral-7B-Instruct if you have a GPU.
#
#   api         OpenAI GPT-4o-mini (or GPT-3.5-turbo) via API
#               Best QA quality. Requires OPENAI_API_KEY env var.
#               This is what Sarthi et al. used for RAPTOR's headline numbers.
# ============================================================================


class LocalBartSummarizationModel(BaseSummarizationModel):
    """HuggingFace summarizer — scales with model name."""

    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self._pipeline = None
        self._load_error = None

    def _ensure_loaded(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "summarization", model=self.model_name, tokenizer=self.model_name)
            print(f"    Summarizer loaded: {self.model_name}")
        except Exception as exc:
            self._load_error = exc

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text: return ""
        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                result = self._pipeline(
                    text, max_new_tokens=min(int(max_tokens), 128),
                    min_new_tokens=20, do_sample=False, truncation=True)
                return result[0]["summary_text"].strip()
            except Exception: pass
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(sentences[:2]) + ("." if sentences else "")


class LocalQAModel(BaseQAModel):
    """
    HuggingFace text generation for QA. Supports:
      - Seq2Seq models (flan-t5-base, flan-t5-large, flan-t5-xl)
      - Causal LMs (mistralai/Mistral-7B-Instruct-v0.3, etc.)
    Automatically detects the model type from the HF config.
    """

    def __init__(self, model_name="google/flan-t5-base", max_new_tokens=80):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._load_error = None
        self._is_causal = None  # detected at load time

    def _ensure_loaded(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline, AutoConfig
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            # Seq2Seq models (T5, BART) use "text2text-generation"
            # Causal models (Mistral, Llama, GPT) use "text-generation"
            if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
                task = "document-question-answering"
                self._is_causal = False
            else:
                task = "document-question-answering"
                self._is_causal = True
            self._pipeline = hf_pipeline(
                task, model=self.model_name, tokenizer=self.model_name,
                trust_remote_code=True,
                device_map="auto",  # uses GPU if available, CPU otherwise
            )
            print(f"    QA model loaded: {self.model_name} (task={task})")
        except Exception as exc:
            self._load_error = exc
            print(f"    [WARN] QA model failed to load: {exc}")

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context: return ""
        self._ensure_loaded()
        if self._pipeline is not None:
            prompt = (
                f"Based on the following context, answer the question in one "
                f"using the following information {context}. "
                f"Answer the following question in less than 5-7 words, if possible: {question}"
                f"Answer:"
            )
            try:
                result = self._pipeline(
                    prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
                if self._is_causal:
                    # Causal models return the full prompt + generation
                    full_text = result[0]["generated_text"]
                    # Extract only the generated part after "Answer:"
                    if "Answer:" in full_text:
                        answer = full_text.split("Answer:")[-1].strip()
                    else:
                        answer = full_text[len(prompt):].strip()
                    return answer
                else:
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
        return best + "." if best else ""


class OpenAIQAModel(BaseQAModel):
    """
    OpenAI API-based QA model. Uses gpt-4o-mini by default.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, model_name="gpt-4o-mini", max_tokens=150):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI()
            print(f"    OpenAI client initialized: {self.model_name}")
        except ImportError:
            raise ImportError(
                "OpenAI API tier requires the `openai` package: pip install openai"
            )

    def answer_question(self, context, question):
        context = " ".join(str(context).split())
        question = str(question).strip()
        if not context: return ""
        self._ensure_client()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a precise question-answering assistant. "
                                "Answer based only on the provided context. Be specific "
                                f"and include key names, dates, and facts. using the following information {context}. "
                                f"Answer the following question in less than 5-7 words, if possible: {question}."},
                    {"role": "user",
                     "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
                max_tokens=self.max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"    [WARN] OpenAI API error: {exc}")
            return ""


class OpenAISummarizationModel(BaseSummarizationModel):
    """OpenAI API-based summarizer for tree construction."""

    def __init__(self, model_name="gpt-4o-mini", max_tokens=150):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI()

    def summarize(self, context, max_tokens=150):
        text = " ".join(str(context).split())
        if not text: return ""
        self._ensure_client()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "Summarize the following text concisely, preserving "
                                "key facts, names, and dates."},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"    [WARN] OpenAI summarization error: {exc}")
            return text[:200]


# ============================================================================
# Model tier definitions
# ============================================================================

MODEL_TIERS = {
    "base": {
        "description": "FLAN-T5-base (770M) + DistilBART + MiniLM — runs on any machine",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "summarizer": ("local", "sshleifer/distilbart-cnn-12-6"),
        "qa": ("local", "google/flan-t5-base"),
    },
    "local-large": {
        "description": "FLAN-T5-large (3B) + BART-large-CNN + mpnet — needs ~8GB",
        "embedding": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summarizer": ("local", "facebook/bart-large-cnn"),
        "qa": ("local", "google/flan-t5-large"),
    },
    "local-xl": {
        "description": "FLAN-T5-XL (11B) + BART-large-CNN + mpnet — needs ~16GB GPU",
        "embedding": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summarizer": ("local", "facebook/bart-large-cnn"),
        "qa": ("local", "google/flan-t5-xl"),
    },
    "mistral": {
        "description": "Mistral-7B-Instruct + BART-large-CNN + mpnet — needs ~16GB GPU",
        "embedding": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summarizer": ("local", "facebook/bart-large-cnn"),
        "qa": ("local", "mistralai/Mistral-7B-Instruct-v0.3"),
    },
    "api": {
        "description": "GPT-4o-mini via OpenAI API — best quality, needs OPENAI_API_KEY",
        "embedding": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summarizer": ("api", "gpt-4o-mini"),
        "qa": ("api", "gpt-4o-mini"),
    },
    "api-gpt4": {
        "description": "GPT-4o via OpenAI API — highest quality, most expensive",
        "embedding": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "summarizer": ("api", "gpt-4o"),
        "qa": ("api", "gpt-4o"),
    },
}


def _shared_models(tier_name: str = "base"):
    """Build the model stack for the requested tier."""
    if tier_name not in MODEL_TIERS:
        print(f"Unknown model tier '{tier_name}'. Available: {list(MODEL_TIERS.keys())}")
        sys.exit(1)

    tier = MODEL_TIERS[tier_name]
    print(f"\n  Model tier: {tier_name}")
    print(f"  {tier['description']}")

    # Embedding model (always local SBERT)
    emb = SBertEmbeddingModel(model_name=tier["embedding"])
    print(f"    Embeddings: {tier['embedding']}")

    # Summarizer
    summ_type, summ_name = tier["summarizer"]
    if summ_type == "api":
        summ = OpenAISummarizationModel(model_name=summ_name)
        print(f"    Summarizer: {summ_name} (API)")
    else:
        summ = LocalBartSummarizationModel(model_name=summ_name)
        print(f"    Summarizer: {summ_name} (local)")

    # QA model
    qa_type, qa_name = tier["qa"]
    if qa_type == "api":
        qa = OpenAIQAModel(model_name=qa_name)
        print(f"    QA model:   {qa_name} (API)")
    else:
        qa = LocalQAModel(model_name=qa_name)
        print(f"    QA model:   {qa_name} (local)")

    return emb, summ, qa


def make_original_config(emb, summ, qa):
    return RetrievalAugmentationConfig(
        embedding_model=emb, summarization_model=summ, qa_model=qa,
        tb_max_tokens=100, tb_num_layers=3, tb_summarization_length=100,
        tr_top_k=5, tr_selection_mode="top_k",
    )


def make_gmm_config(emb, summ, qa):
    from raptor.clustering import GMMClusterer
    clusterer = GMMClusterer(
        reduction_dimension=10, soft_threshold=0.1,
        force_hard_clustering=False, random_state=224,
    )
    tree_config = ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={},
        reduction_dimension=10, summarization_model=summ,
        embedding_models={"EMB": emb}, cluster_embedding_model="EMB",
        max_tokens=100, num_layers=3, summarization_length=100,
    )
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config, qa_model=qa,
        embedding_model=emb, tr_top_k=5, tr_selection_mode="top_k",
    )


def make_leiden_config(emb, summ, qa):
    from raptor.clustering import LeidenClusterer, LeidenConfig
    leiden_cfg = LeidenConfig(
        k_neighbors=10, use_adjacency_edges=True, adjacency_weight=0.5,
        resolution=1.0, resolution_schedule={0: 1.2, 1: 0.8},
        partition_type="RBConfiguration", min_cluster_size=1,
    )
    clusterer = LeidenClusterer(config=leiden_cfg, random_state=224)
    tree_config = ClusterTreeConfig(
        clustering_algorithm=clusterer, clustering_params={},
        reduction_dimension=10, summarization_model=summ,
        embedding_models={"EMB": emb}, cluster_embedding_model="EMB",
        max_tokens=100, num_layers=3, summarization_length=100,
    )
    return RetrievalAugmentationConfig(
        tree_builder_config=tree_config, qa_model=qa,
        embedding_model=emb, tr_top_k=5, tr_selection_mode="top_k",
    )


METHOD_REGISTRY = {
    "original": (make_original_config, "RAPTOR (GMM+UMAP) upstream"),
    "gmm":      (make_gmm_config,      "GMMClusterer (new interface)"),
    "leiden":   (make_leiden_config,     "LeidenClusterer (k-NN graph)"),
}


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(
    document: str,
    benchmark: List[Dict],
    methods: List[str],
    model_tier: str = "base",
    verbose: bool = False,
):
    print("Loading models...")
    emb, summ, qa = _shared_models(model_tier)
    print("Models ready.\n")

    summary_rows = []
    detail_rows = []

    for method_name in methods:
        factory_fn, description = METHOD_REGISTRY[method_name]
        try:
            config = factory_fn(emb, summ, qa)
        except ImportError as exc:
            print(f"  SKIPPING {method_name}: {exc}")
            continue

        print(f"\n{'='*70}")
        print(f"  {method_name} — {description}")
        print(f"{'='*70}")

        ra = RetrievalAugmentation(config=config)
        t0 = time.time()
        ra.add_documents(document)
        build_time = time.time() - t0

        tree = ra.tree
        total_nodes = len(tree.all_nodes)
        leaf_nodes = len(tree.leaf_nodes)
        summary_nodes = total_nodes - leaf_nodes
        print(f"  Tree: {tree.num_layers} layers, {total_nodes} nodes "
              f"({leaf_nodes} leaves + {summary_nodes} summaries) in {build_time:.2f}s")
        for li, nodes in sorted(tree.layer_to_nodes.items()):
            print(f"    Layer {li}: {len(nodes)} nodes")

        # Answer questions
        totals = {"f1": 0, "rouge_l": 0, "contains_em": 0, "ret_hit": 0}
        type_totals = {}

        for i, qa_pair in enumerate(benchmark):
            t_q = time.time()
            predicted = ra.answer_question(question=qa_pair["question"])
            answer_time = time.time() - t_q

            scores = score_prediction(predicted, qa_pair["answer"])

            # Retrieval hit
            try:
                retrieved = ra.retrieve(qa_pair["question"])
                hit = check_retrieval_hit(retrieved, qa_pair["answer"])
            except Exception:
                retrieved, hit = "", False

            scores["retrieval_hit"] = int(hit)
            totals["f1"] += scores["f1"]
            totals["rouge_l"] += scores["rouge_l"]
            totals["contains_em"] += scores["contains_em"]
            totals["ret_hit"] += scores["retrieval_hit"]

            qtype = qa_pair["type"]
            if qtype not in type_totals:
                type_totals[qtype] = {"f1": 0, "rouge_l": 0, "n": 0, "ret_hit": 0}
            type_totals[qtype]["f1"] += scores["f1"]
            type_totals[qtype]["rouge_l"] += scores["rouge_l"]
            type_totals[qtype]["ret_hit"] += scores["retrieval_hit"]
            type_totals[qtype]["n"] += 1

            detail_rows.append({
                "method": method_name, "q_num": i + 1,
                "type": qtype,
                "question": qa_pair["question"],
                "ground_truth": qa_pair["answer"],
                "predicted": predicted,
                "f1": round(scores["f1"], 4),
                "rouge_l": round(scores["rouge_l"], 4),
                "contains_em": int(scores["contains_em"]),
                "retrieval_hit": int(hit),
                "answer_time": round(answer_time, 3),
            })

            if verbose:
                hit_str = "HIT" if hit else "MISS"
                print(f"\n  Q{i+1} [{qtype:6s}|{hit_str}] F1={scores['f1']:.3f} "
                      f"ROUGE-L={scores['rouge_l']:.3f}")
                print(f"    Q: {qa_pair['question'][:65]}")
                print(f"    A: {predicted[:80]}")
                if scores["f1"] < 0.2:
                    print(f"    Expected: {qa_pair['answer'][:80]}")

        n = len(benchmark)
        summary_rows.append({
            "method": method_name,
            "description": description,
            "n_questions": n,
            "mean_f1": round(totals["f1"] / n, 4),
            "mean_rouge_l": round(totals["rouge_l"] / n, 4),
            "contains_em_%": round(totals["contains_em"] / n * 100, 1),
            "retrieval_hit_%": round(totals["ret_hit"] / n * 100, 1),
            "num_layers": tree.num_layers,
            "total_nodes": total_nodes,
            "leaves": leaf_nodes,
            "summaries": summary_nodes,
            "build_time_sec": round(build_time, 2),
        })

        print(f"\n  Totals: F1={totals['f1']/n:.4f}  ROUGE-L={totals['rouge_l']/n:.4f}"
              f"  ContainsEM={totals['contains_em']/n*100:.1f}%"
              f"  RetHit={totals['ret_hit']/n*100:.1f}%")

        # Per question-type breakdown
        print(f"\n  By question type:")
        for qtype in ["local", "cross", "global"]:
            if qtype in type_totals:
                t = type_totals[qtype]
                print(f"    {qtype:6s}: F1={t['f1']/t['n']:.3f}  "
                      f"ROUGE-L={t['rouge_l']/t['n']:.3f}  "
                      f"RetHit={t['ret_hit']/t['n']*100:.0f}%  "
                      f"(n={t['n']})")

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ============================================================================
# Display & save
# ============================================================================

def display_and_save(df_summary, df_detail):
    if len(df_summary) == 0:
        print("\nNo results.")
        return

    print("\n\n" + "=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    if len(df_summary) > 1:
        best = df_summary.loc[df_summary["mean_f1"].idxmax()]
        print(f"\n  Best F1: {best['method']} ({best['mean_f1']:.4f})")
        best_ret = df_summary.loc[df_summary["retrieval_hit_%"].idxmax()]
        print(f"  Best Retrieval Hit Rate: {best_ret['method']} ({best_ret['retrieval_hit_%']:.1f}%)")

    # Per question-type pivot
    if len(df_detail) > 0:
        print("\n" + "=" * 100)
        print("  BY QUESTION TYPE")
        print("=" * 100)
        by_type = df_detail.groupby(["method", "type"]).agg(
            n=("f1", "count"),
            mean_f1=("f1", "mean"),
            mean_rouge_l=("rouge_l", "mean"),
            ret_hit=("retrieval_hit", "mean"),
        ).round(4)
        by_type["ret_hit"] = (by_type["ret_hit"] * 100).round(1)
        by_type.columns = ["n", "mean_f1", "mean_rouge_l", "ret_hit_%"]
        print(by_type.to_string())

    df_summary.to_csv("results_demo_summary.csv", index=False)
    df_detail.to_csv("results_demo_detailed.csv", index=False)
    print(f"\n  Saved: results_demo_summary.csv, results_demo_detailed.csv")


# ============================================================================
# Main
# ============================================================================

def main():
    tier_names = list(MODEL_TIERS.keys())
    parser = argparse.ArgumentParser(
        description="RAPTOR clustering evaluation demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model tiers:\n" + "\n".join(
            f"  {name:14s} {info['description']}"
            for name, info in MODEL_TIERS.items()
        ),
    )
    parser.add_argument("--methods", nargs="+", default=list(METHOD_REGISTRY.keys()),
                        choices=list(METHOD_REGISTRY.keys()))
    parser.add_argument("--model-tier", default="base", choices=tier_names,
                        help=f"Model quality tier (default: base). Choices: {tier_names}")
    parser.add_argument("--document", default=DOCUMENT_PATH,
                        help="Path to document .txt file (default: data/document.txt)")
    parser.add_argument("--questions", default=QUESTIONS_PATH,
                        help="Path to questions .json file (default: data/questions.json)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  RAPTOR Clustering Evaluation — Abstractive QA Demo")
    print(f"  Model tier: {args.model_tier}")
    print("=" * 70)

    # Load data from files
    document = load_document(args.document)
    benchmark = load_questions(args.questions)

    type_counts = Counter(q["type"] for q in benchmark)
    print(f"  Questions: {len(benchmark)} total — "
          + ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items())))
    print(f"  Methods: {', '.join(args.methods)}")
    print("=" * 70)

    df_summary, df_detail = run_evaluation(
        document, benchmark, args.methods,
        model_tier=args.model_tier, verbose=args.verbose,
    )
    display_and_save(df_summary, df_detail)
    print("\nDone.")


if __name__ == "__main__":
    main()