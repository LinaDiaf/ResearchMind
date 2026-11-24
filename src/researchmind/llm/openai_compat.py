"""Any server exposing the OpenAI chat-completions API (vLLM, LM Studio, OpenRouter...)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from researchmind.llm.base import LLMError, LLMResponse, Message


class OpenAICompatLLM:
    provider = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s, headers=headers)

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
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }
        start = time.perf_counter()
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        choices = data.get("choices") or []
        text = (choices[0].get("message") or {}).get("content") if choices else ""
        if not text or not text.strip():
            raise LLMError("OpenAI-compatible server returned an empty answer")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency,
            finish_reason=choices[0].get("finish_reason") if choices else None,
        )
