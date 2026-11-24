"""Rank fusion."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence

RRF_K = 60


def weighted_rrf(
    rankings: Sequence[tuple[float, Sequence[tuple[uuid.UUID, float]]]], k: int = RRF_K
) -> list[tuple[uuid.UUID, float]]:
    """Weighted reciprocal rank fusion. ``rankings`` = [(weight, [(id, score), ...]), ...]
    with each list sorted best-first. Ties are broken by first appearance."""
    scores: dict[uuid.UUID, float] = defaultdict(float)
    order: dict[uuid.UUID, int] = {}
    for weight, ranking in rankings:
        if weight <= 0:
            continue
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
