"""Ollama native chat API (``/api/chat``), with JSON-schema constrained output."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from researchmind.llm.base import LLMError, LLMResponse, Message


class OllamaLLM:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        num_ctx: int = 16384,
        think: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout_s)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": self.think,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        start = time.perf_counter()
        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        text = (data.get("message") or {}).get("content") or ""
        if not text.strip():
            raise LLMError(f"Ollama returned an empty answer (done_reason={data.get('done_reason')})")
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency,
            finish_reason=data.get("done_reason"),
            raw={k: v for k, v in data.items() if k != "message"},
        )
