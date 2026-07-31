"""
narrativeqa_metric.py — Free-form answer metrics for NarrativeQA.

NarrativeQA answers are short free-form references (typically <10 words), with
usually TWO reference answers per question written by different annotators.
Following standard practice, every metric is computed against each reference
and the question's score is the MAX over references — a prediction is rewarded
for matching either annotator.

Metrics (kept identical in definition to the QASPER harness for thesis-internal
comparability):
  - token_f1  : multiset token F1 after normalization (lowercase, strip
                punctuation, drop articles a/an/the, collapse whitespace)
  - rouge_l   : LCS-based F-measure — the HEADLINE metric for NarrativeQA,
                because correct answers frequently paraphrase the reference
                while preserving ordered content
  - bleu      : n-gram precision capped at the shorter sequence length with
                light smoothing for higher orders; 0 if no unigram overlap

The official NarrativeQA suite also reports METEOR; it is omitted here because
it adds NLTK/WordNet dependencies for little extra discrimination on answers
this short, and thesis-internal comparability favors the shared metric set.

bootstrap_ci resamples per-question scores (seed 224, 2000 replicates) so
method differences can be judged against sampling noise from the start — the
lesson carried over from the QASPER statistical-caution discussion.
"""

from __future__ import annotations

import string
from collections import Counter
from typing import Dict, List


ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> List[str]:
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    return [t for t in s.split() if t and t not in ARTICLES]


# ---------------------------------------------------------------------------
# Single-reference metrics
# ---------------------------------------------------------------------------

def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 1.0 if (not p and not r) else 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)]


def rouge_l(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 1.0 if (not p and not r) else 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def bleu(pred: str, ref: str) -> float:
    """Capped BLEU: n-gram orders up to min(4, len of shorter sequence)."""
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 1.0 if (not p and not r) else 0.0

    max_n = min(4, len(p), len(r))
    log_sum = 0.0
    import math
    for n in range(1, max_n + 1):
        p_ngrams = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        r_ngrams = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        overlap = sum((p_ngrams & r_ngrams).values())
        total = sum(p_ngrams.values())
        if n == 1 and overlap == 0:
            return 0.0
        if overlap == 0:
            prec = 1.0 / (2 * total)   # light smoothing for higher orders
        else:
            prec = overlap / total
        log_sum += math.log(prec)
    geo = math.exp(log_sum / max_n)

    bp = 1.0 if len(p) >= len(r) else math.exp(1 - len(r) / len(p))
    return bp * geo


# ---------------------------------------------------------------------------
# Multi-reference scoring
# ---------------------------------------------------------------------------

def score_prediction(pred: str, references: List[str]) -> Dict:
    """
    Score one prediction against all references; each metric independently
    takes its max over references.
    """
    refs = [r for r in references if r and r.strip()]
    if not refs:
        return {"f1": 0.0, "rouge_l": 0.0, "bleu": 0.0}
    return {
        "f1": max(token_f1(pred, r) for r in refs),
        "rouge_l": max(rouge_l(pred, r) for r in refs),
        "bleu": max(bleu(pred, r) for r in refs),
    }


def aggregate_scores(per_question: List[Dict]) -> Dict:
    if not per_question:
        return {"n": 0}
    n = len(per_question)
    out = {"n": n}
    for key in ("f1", "rouge_l", "bleu"):
        out[key] = round(sum(q[key] for q in per_question) / n, 4)
    return out


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 224) -> Dict:
    """95% bootstrap CI over per-question metric values."""
    import numpy as np
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = arr[rng.integers(0, n, size=n)].mean()
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(np.percentile(means, 2.5)), 4),
        "hi": round(float(np.percentile(means, 97.5)), 4),
        "n": n,
    }
