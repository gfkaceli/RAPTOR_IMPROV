"""
quality_metric.py — QuALITY multiple-choice accuracy metric.

QuALITY is scored by accuracy: does the model select the correct option?
The gold label is 1-indexed (1..4). The metric's real difficulty is not the
comparison — it's robustly parsing which option the model *chose* from free-form
text, because an instruction-tuned model may answer in many forms:

    "B"                          -> letter
    "(B)"  "B."  "Answer: B"     -> letter with decoration
    "The second option"          -> ordinal words
    "Paris"                       -> restating the option text
    "The answer is Paris."        -> option text embedded in a sentence

We parse defensively and fall back to fuzzy text matching against the options.
An unparseable answer is scored as incorrect (not skipped), so accuracy is never
inflated by silently dropping hard-to-parse predictions.

Reference: Pang et al. (2022), QuALITY. arXiv:2112.08608.
Baseline model accuracy ~55.4%, human ceiling ~93.5% (89.1% on HARD).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional


LETTERS = ["A", "B", "C", "D", "E", "F"]
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
    "one": 1, "two": 2, "three": 3, "four": 4,
}


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _token_overlap(a: str, b: str) -> float:
    """Symmetric token-overlap ratio for fuzzy option matching."""
    ta, tb = _normalize(a).split(), _normalize(b).split()
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    return common / max(len(ta), len(tb))


def parse_predicted_option(prediction: str, options: List[str]) -> Optional[int]:
    """
    Return the 1-indexed option the model selected, or None if unparseable.

    Parsing order (most to least reliable):
      1. Explicit leading letter: "B", "(B)", "B.", "Answer: B"
      2. Ordinal words: "the second option"
      3. Exact/fuzzy match of prediction against one option's text
    """
    if not prediction or not options:
        return None

    pred = prediction.strip()
    n = len(options)
    valid_letters = LETTERS[:n]

    # --- 1. Letter answer ----------------------------------------------------
    # Try progressively looser patterns, most reliable first.
    letter = None
    # 1a. Leading decorated letter: "B", "(B)", "B.", "[C]"
    m = re.match(r"^\s*[\(\[]?\s*([A-Fa-f])\s*[\)\].:]", pred)
    # 1b. bare single letter as the whole answer
    if not m:
        m = re.match(r"^\s*[\(\[]?\s*([A-Fa-f])\s*[\)\]]?\s*$", pred)
    # 1c. "answer/option/choice (is)(:) X" anywhere
    if not m:
        m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-Fa-f])\b", pred, re.I)
    if m:
        letter = m.group(1).upper()

    if letter and letter in valid_letters:
        return valid_letters.index(letter) + 1

    # --- 2. Ordinals: words ("second") and digits ("option 3") --------------
    low = pred.lower()
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", low) and idx <= n:
            return idx
    # "option 3" / "choice 2" / "number 4" style digit ordinals
    m = re.search(r"\b(?:option|choice|answer|number|#)\s*(\d)\b", low)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= n:
            return idx

    # --- 3. Text match against option content --------------------------------
    # Exact normalized containment first
    norm_pred = _normalize(pred)
    exact_hits = [
        i for i, opt in enumerate(options)
        if _normalize(opt) and _normalize(opt) in norm_pred
    ]
    if len(exact_hits) == 1:
        return exact_hits[0] + 1

    # Fuzzy: highest token overlap, with a margin to avoid near-ties
    scores = [(_token_overlap(pred, opt), i) for i, opt in enumerate(options)]
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 0.5:
        if len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15:
            return scores[0][1] + 1

    return None


def score_prediction(prediction: str, gold_label: Optional[int],
                     options: List[str]) -> Dict:
    """
    Score one QuALITY prediction.

    Returns:
        {
          "correct": 0/1,           # 1 if parsed option == gold
          "predicted_index": int|None,
          "gold_index": int|None,
          "parsed": bool,           # whether we could parse an option at all
        }
    """
    predicted_index = parse_predicted_option(prediction, options)
    parsed = predicted_index is not None

    correct = 0
    if gold_label is not None and predicted_index is not None:
        correct = 1 if predicted_index == gold_label else 0

    return {
        "correct": correct,
        "predicted_index": predicted_index,
        "gold_index": gold_label,
        "parsed": parsed,
    }


def aggregate_scores(per_question: List[Dict]) -> Dict:
    """
    Aggregate into overall accuracy plus HARD/EASY breakdown and a parse rate.

    Each input dict must have: "correct", "parsed", "difficult" (0/1).
    """
    if not per_question:
        return {"accuracy": 0.0, "n": 0}

    n = len(per_question)
    n_correct = sum(q["correct"] for q in per_question)
    n_parsed = sum(1 for q in per_question if q["parsed"])

    hard = [q for q in per_question if q.get("difficult", 0) == 1]
    easy = [q for q in per_question if q.get("difficult", 0) == 0]

    result = {
        "accuracy": round(n_correct / n, 4),
        "n": n,
        "parse_rate": round(n_parsed / n, 4),
        "n_correct": n_correct,
    }
    if hard:
        result["accuracy_hard"] = round(sum(q["correct"] for q in hard) / len(hard), 4)
        result["n_hard"] = len(hard)
    if easy:
        result["accuracy_easy"] = round(sum(q["correct"] for q in easy) / len(easy), 4)
        result["n_easy"] = len(easy)
    return result


def bootstrap_ci(per_question: List[Dict], n_boot: int = 2000,
                 seed: int = 224, subset: Optional[str] = None) -> Dict:
    """
    Bootstrap 95% CI for accuracy. Resamples questions with replacement.

    subset: None (all), "hard", or "easy" — CI for that slice.
    Returns {"mean", "lo", "hi"} as proportions.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    if subset == "hard":
        pool = [q for q in per_question if q.get("difficult", 0) == 1]
    elif subset == "easy":
        pool = [q for q in per_question if q.get("difficult", 0) == 0]
    else:
        pool = per_question

    if not pool:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}

    correct = np.array([q["correct"] for q in pool], dtype=float)
    n = len(correct)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = correct[idx].mean()

    return {
        "mean": round(float(correct.mean()), 4),
        "lo": round(float(np.percentile(means, 2.5)), 4),
        "hi": round(float(np.percentile(means, 97.5)), 4),
        "n": n,
    }
