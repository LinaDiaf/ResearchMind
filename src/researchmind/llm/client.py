"""LLM client used by the agent: budget enforcement, accounting, typed generation."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from researchmind.llm.base import LLMError, LLMProvider, LLMResponse, Message, estimate_cost
from researchmind.llm.structured import (
    StructuredOutputError,
    json_schema_for,
    parse_as,
    repair_instruction,
)
from researchmind.log import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the investigation's LLM-call budget."""


@dataclass
class CallRecord:
    node: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    status: str  # ok | repaired | invalid | error


Recorder = Callable[[CallRecord], None]


@dataclass
class UsageCounter:
    max_calls: int | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
            self.calls += 1

    def add(self, resp: LLMResponse, cost: float) -> None:
        with self._lock:
            self.prompt_tokens += resp.prompt_tokens
            self.completion_tokens += resp.completion_tokens
            self.cost_usd += cost
            self.latency_ms += resp.latency_ms

    @property
    def remaining(self) -> int | None:
        return None if self.max_calls is None else max(0, self.max_calls - self.calls)


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        usage: UsageCounter | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self.provider = provider
        self.usage = usage or UsageCounter()
        self.recorder = recorder

    def _call(
        self,
        node: str,
        messages: Sequence[Message],
        json_schema: dict | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        self.usage.reserve()
        try:
            resp = self.provider.complete(messages, json_schema=json_schema, max_tokens=max_tokens)
        except LLMError:
            self._record(node, None, "error")
            raise
        cost = estimate_cost(resp.model, resp.prompt_tokens, resp.completion_tokens)
        self.usage.add(resp, cost)
        return resp

    def _record(self, node: str, resp: LLMResponse | None, status: str) -> None:
        if not self.recorder:
            return
        rec = CallRecord(
            node=node,
            provider=self.provider.provider,
            model=self.provider.model,
            prompt_tokens=resp.prompt_tokens if resp else 0,
            completion_tokens=resp.completion_tokens if resp else 0,
            latency_ms=resp.latency_ms if resp else 0.0,
            cost_usd=estimate_cost(resp.model, resp.prompt_tokens, resp.completion_tokens)
            if resp
            else 0.0,
            status=status,
        )
        try:
            self.recorder(rec)
        except Exception:  # accounting must never break the run
            log.warning("llm_record_failed", node=node)

    def text(
        self, node: str, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> str:
        resp = self._call(node, messages, None, max_tokens)
        self._record(node, resp, "ok")
        return resp.text

    def structured(
        self,
        node: str,
        messages: Sequence[Message],
        schema: type[T],
        *,
        max_tokens: int | None = None,
        repair_attempts: int = 1,
    ) -> T:
        """Generate an instance of ``schema``. Raises StructuredOutputError after the repair
        attempts are used; callers decide on a deterministic fallback."""
        js = json_schema_for(schema)
        msgs = list(messages)
        last_error = ""
        for attempt in range(repair_attempts + 1):
            resp = self._call(node, msgs, js, max_tokens)
            try:
                result = parse_as(schema, resp.text)
            except StructuredOutputError as exc:
                last_error = str(exc)
                self._record(node, resp, "invalid")
                log.warning("structured_output_invalid", node=node, attempt=attempt,
                            error=last_error, finish=resp.finish_reason)
                if attempt >= repair_attempts:
                    raise StructuredOutputError(last_error, raw=resp.text) from exc
                msgs = [
                    *messages,
                    Message("assistant", resp.text[:4000]),
                    Message("user", repair_instruction(last_error)),
                ]
                continue
            self._record(node, resp, "ok" if attempt == 0 else "repaired")
            return result
        raise StructuredOutputError(last_error)  # pragma: no cover
