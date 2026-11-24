"""Evaluation metrics.

Relevance of a retrieved passage is judged against human gold evidence paragraphs: a passage
"covers" a gold paragraph when at least ``threshold`` of the paragraph's tokens appear in it
(paragraphs can be split across chunks, so partial coverage counts for that chunk).
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def coverage(passage: str, gold: str) -> float:
    g = Counter(tokens(gold))
    if not g:
        return 0.0
    p = Counter(tokens(passage))
    return sum(min(c, p[t]) for t, c in g.items()) / sum(g.values())


def relevant_flags(passages: Sequence[str], gold: Sequence[str], threshold: float = 0.5) -> list[set[int]]:
    """For each passage, the set of gold paragraph indices it covers."""
    return [{i for i, g in enumerate(gold) if coverage(p, g) >= threshold} for p in passages]


def recall_at_k(flags: list[set[int]], n_gold: int, k: int) -> float:
    if n_gold == 0:
        return float("nan")
    covered = set().union(*flags[:k]) if flags[:k] else set()
    return len(covered) / n_gold


def mrr(flags: list[set[int]]) -> float:
    for i, f in enumerate(flags, start=1):
        if f:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: list[set[int]], n_gold: int, k: int) -> float:
    if n_gold == 0:
        return float("nan")
    seen: set[int] = set()
    dcg = 0.0
    for i, f in enumerate(flags[:k], start=1):
        new = f - seen
        if new:  # a passage gains only for gold paragraphs not already covered
            dcg += 1.0 / math.log2(i + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_gold, k) + 1))
    return dcg / ideal if ideal else 0.0


def citation_precision(cited: Sequence[str], gold: Sequence[str], threshold: float = 0.5) -> float:
    if not cited:
        return float("nan")
    flags = relevant_flags(cited, gold, threshold)
    return sum(1 for f in flags if f) / len(cited)


# --- answer F1 (as in the official QASPER evaluator: SQuAD-style token F1, max over annotators)
def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, gold: str) -> float:
    p, g = _normalize_answer(prediction).split(), _normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, golds: Sequence[str]) -> float:
    return max((token_f1(prediction, g) for g in golds), default=0.0)


def mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return stats[int(alpha / 2 * n)], stats[int((1 - alpha / 2) * n) - 1]


def binary_scores(pred: Sequence[bool], gold: Sequence[bool]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold, strict=True))
    fp = sum(p and not g for p, g in zip(pred, gold, strict=True))
    fn = sum(g and not p for p, g in zip(pred, gold, strict=True))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and precision + recall else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
