"""Provider-independent LLM interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMError(RuntimeError):
    """A provider call failed (network, HTTP error, empty answer)."""


@runtime_checkable
class LLMProvider(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return one completion. When ``json_schema`` is given the provider constrains the
        output to JSON (natively when supported); validation happens in ``structured.py``."""
        ...


# Prices in USD per million tokens (input, output). Local models cost nothing.
# Values are only used for reporting; unknown models are counted as 0 and flagged in logs.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICES_PER_MTOK.get(model)
    if not price:
        return 0.0
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
