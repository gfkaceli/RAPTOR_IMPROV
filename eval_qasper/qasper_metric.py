"""
QASPER official Answer F1 metric.

Ports the evaluation logic from allenai/qasper-led-baseline/scripts/evaluator.py
(MIT licensed). This is the canonical metric used by Dasigi et al. (2021,
arXiv:2105.03011) and reported by Sarthi et al. (RAPTOR, ICLR 2024) on QASPER.

The metric:
    1. Normalize prediction and reference text (lowercase, strip articles
       and punctuation, normalize whitespace).
    2. Compute token-level F1 between prediction and each reference answer.
    3. Take MAX F1 across all references for the question.
    4. Report mean F1 over all questions, broken down by answer type.

QASPER has four answer types:
    - extractive: gold = joined extractive_spans
    - abstractive: gold = free_form_answer
    - yes_no: gold = "Yes" or "No"
    - unanswerable: gold = "Unanswerable"
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Text normalization (matches SQuAD / official QASPER eval)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles, fix whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Standard SQuAD/QASPER token-level F1."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L F-measure based on longest common subsequence.

    LCS captures in-order word overlap without requiring contiguity, so it
    rewards answers that contain the gold tokens in the right order even with
    gaps. Standard for abstractive QA (Sarthi et al. report it on NarrativeQA).

    Returns the F-measure (harmonic mean of LCS-precision and LCS-recall),
    which is what the `rouge_score` package reports as rougeL.
    """
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # LCS length via dynamic programming
    m, n = len(gold), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(gold)
    return 2 * precision * recall / (precision + recall)


def bleu(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU with smoothing for short sequences.

    NOTE for QASPER: BLEU was designed for long MT outputs and is a poor fit
    for short extractive answers. BLEU-4 is structurally 0 whenever an answer
    has fewer than 4 tokens (no 4-grams exist). We cap the effective n-gram
    order at the length of the shorter sequence and smooth only the orders
    that have at least one possible n-gram, so a 1-token exact match scores 1.0
    while a 1-token mismatch scores 0.0. Even so, BLEU remains the least
    informative of the three metrics on QASPER — weight ROUGE-L and token-F1
    more heavily in analysis.
    """
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # If unigrams don't overlap at all, BLEU is 0 — no smoothing rescue.
    pred_unigrams = Counter(pred)
    gold_unigrams = Counter(gold)
    if sum((pred_unigrams & gold_unigrams).values()) == 0:
        return 0.0

    # Cap n-gram order at the shorter sequence length so we don't penalize
    # short answers for missing higher-order n-grams that can't exist.
    effective_n = min(max_n, len(pred), len(gold))
    if effective_n < 1:
        return 0.0

    precisions = []
    for n in range(1, effective_n + 1):
        pred_ngrams = Counter(
            tuple(pred[i:i + n]) for i in range(len(pred) - n + 1)
        )
        gold_ngrams = Counter(
            tuple(gold[i:i + n]) for i in range(len(gold) - n + 1)
        )
        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        if total == 0:
            continue
        if overlap == 0:
            # Smooth only higher-order zeros (NLTK method-1 style) so a single
            # missing 3-gram doesn't zero out an otherwise-good short answer.
            precisions.append(1.0 / (2.0 * total))
        else:
            precisions.append(overlap / total)

    if not precisions:
        return 0.0

    # Geometric mean of precisions
    log_sum = sum(np.log(p) for p in precisions) / len(precisions)
    geo_mean = float(np.exp(log_sum))

    # Brevity penalty
    pred_len, gold_len = len(pred), len(gold)
    if pred_len >= gold_len:
        bp = 1.0
    elif pred_len == 0:
        bp = 0.0
    else:
        bp = float(np.exp(1.0 - gold_len / pred_len))

    return bp * geo_mean


# ---------------------------------------------------------------------------
# Reference extraction (handles all 4 QASPER answer types)
# ---------------------------------------------------------------------------

def get_answer_type_and_text(answer: Dict) -> Tuple[str, str]:
    """
    Given one annotator's answer dict, return (answer_type, reference_text).

    QASPER's answer schema (from HuggingFace allenai/qasper):
        {
          "unanswerable": bool,
          "extractive_spans": List[str],
          "yes_no": bool or null,
          "free_form_answer": str,
          "evidence": List[str],
          ...
        }
    """
    if answer.get("unanswerable", False):
        return "unanswerable", "Unanswerable"

    yes_no = answer.get("yes_no", None)
    if yes_no is not None and yes_no != "":
        # yes_no can be True / False / null; bool indicates yes/no answer type
        if isinstance(yes_no, bool):
            return "yes_no", "Yes" if yes_no else "No"

    extractive = answer.get("extractive_spans", []) or []
    if extractive:
        return "extractive", " ".join(extractive)

    free_form = answer.get("free_form_answer", "") or ""
    if free_form:
        return "abstractive", free_form

    # Fallback — empty/malformed annotation
    return "none", ""


def collect_references(answers: List[Dict]) -> List[Tuple[str, str]]:
    """
    Given the list of annotators' answers for one question, return a list of
    (answer_type, reference_text) tuples — one per annotator that gave a
    non-empty answer.
    """
    refs = []
    for ann in answers:
        # HF schema wraps each annotation in an "answer" sub-dict
        ans = ann.get("answer", ann)
        atype, text = get_answer_type_and_text(ans)
        if text or atype == "unanswerable":
            refs.append((atype, text))
    return refs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prediction(prediction: str, answers: List[Dict]) -> Dict[str, float]:
    """
    Score one prediction against the list of annotator answers for the question.

    For each metric (token-F1, ROUGE-L, BLEU) we take the MAX across references,
    following the official QASPER / SQuAD multi-reference protocol. The
    answer_type reported is the type of the reference that gave the best F1
    (F1 is the primary metric; the other two are reported alongside it).

    Returns:
        {
            "f1": float,         # MAX token-F1 across references
            "rouge_l": float,    # MAX ROUGE-L across references
            "bleu": float,       # MAX BLEU across references
            "answer_type": str,  # type of the best-F1 reference
            "n_references": int,
        }
    """
    refs = collect_references(answers)
    if not refs:
        return {"f1": 0.0, "rouge_l": 0.0, "bleu": 0.0,
                "answer_type": "none", "n_references": 0}

    best_f1 = 0.0
    best_rouge = 0.0
    best_bleu = 0.0
    best_type = refs[0][0]
    for atype, text in refs:
        f1 = token_f1(prediction, text)
        if f1 > best_f1:
            best_f1 = f1
            best_type = atype
        # Each metric maxes independently — standard multi-reference practice
        best_rouge = max(best_rouge, rouge_l(prediction, text))
        best_bleu = max(best_bleu, bleu(prediction, text))

    return {
        "f1": best_f1,
        "rouge_l": best_rouge,
        "bleu": best_bleu,
        "answer_type": best_type,
        "n_references": len(refs),
    }


def aggregate_scores(per_question: List[Dict]) -> Dict[str, float]:
    """
    Aggregate per-question scores into the QASPER report metrics.

    Input: list of dicts with keys "f1", "rouge_l", "bleu", "answer_type".
    Output: overall mean of each metric + per-answer-type breakdown.
    """
    if not per_question:
        return {"f1": 0.0, "rouge_l": 0.0, "bleu": 0.0, "n": 0}

    n = len(per_question)
    result = {
        "f1": round(sum(q["f1"] for q in per_question) / n, 4),
        "rouge_l": round(sum(q.get("rouge_l", 0.0) for q in per_question) / n, 4),
        "bleu": round(sum(q.get("bleu", 0.0) for q in per_question) / n, 4),
        "n": n,
    }

    by_type: Dict[str, List[Dict]] = {}
    for q in per_question:
        by_type.setdefault(q["answer_type"], []).append(q)

    for atype, qs in by_type.items():
        m = len(qs)
        result[f"f1_{atype}"] = round(sum(q["f1"] for q in qs) / m, 4)
        result[f"rouge_l_{atype}"] = round(sum(q.get("rouge_l", 0.0) for q in qs) / m, 4)
        result[f"bleu_{atype}"] = round(sum(q.get("bleu", 0.0) for q in qs) / m, 4)
        result[f"n_{atype}"] = m
    return result