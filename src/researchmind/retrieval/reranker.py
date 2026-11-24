"""Cross-encoder reranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from researchmind.config import Settings, get_settings


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class NoReranker:
    """Keeps the fused order (used for ablations and when no model is configured)."""

    name = "none"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        n = len(passages)
        return [float(n - i) / max(n, 1) for i in range(n)]


class CrossEncoderReranker:
    def __init__(self, model: str, device: str = "cpu", max_length: int = 512) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'ml' extra: pip install researchmind[ml]") from exc
        self.name = model
        self._ce = CrossEncoder(model, device=device, max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw = self._ce.predict([(query, p) for p in passages], show_progress_bar=False)
        # Logits → (0, 1) so thresholds are comparable across models.
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]


@lru_cache(maxsize=2)
def _cached_cross_encoder(model: str, device: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model, device)


def build_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    if not s.reranker_model or s.reranker_model.lower() == "none":
        return NoReranker()
    return _cached_cross_encoder(s.reranker_model, s.reranker_device)
