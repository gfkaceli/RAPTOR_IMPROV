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

    Returns:
        {
            "f1": float,         # MAX token-F1 across references
            "answer_type": str,  # type of the best-matching reference
            "n_references": int,
        }
    """
    refs = collect_references(answers)
    if not refs:
        return {"f1": 0.0, "answer_type": "none", "n_references": 0}

    best_f1 = 0.0
    best_type = refs[0][0]
    for atype, text in refs:
        f1 = token_f1(prediction, text)
        if f1 > best_f1:
            best_f1 = f1
            best_type = atype

    return {"f1": best_f1, "answer_type": best_type, "n_references": len(refs)}


def aggregate_scores(per_question: List[Dict]) -> Dict[str, float]:
    """
    Aggregate per-question scores into the QASPER report metrics.

    Input: list of dicts with keys "f1" and "answer_type".
    Output: overall F1 + per-type F1 breakdown.
    """
    if not per_question:
        return {"f1": 0.0, "n": 0}

    overall = sum(q["f1"] for q in per_question) / len(per_question)

    by_type: Dict[str, List[float]] = {}
    for q in per_question:
        by_type.setdefault(q["answer_type"], []).append(q["f1"])

    result = {
        "f1": round(overall, 4),
        "n": len(per_question),
    }
    for atype, scores in by_type.items():
        result[f"f1_{atype}"] = round(sum(scores) / len(scores), 4)
        result[f"n_{atype}"] = len(scores)
    return result
