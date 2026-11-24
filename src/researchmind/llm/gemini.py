"""Google Gemini through the ``google-genai`` SDK."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from researchmind.llm.base import LLMError, LLMResponse, Message


class GeminiLLM:
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        from google.genai import types

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.max_tokens,
            response_mime_type="application/json" if json_schema is not None else None,
            response_json_schema=json_schema,
        )
        start = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:  # the SDK raises several error types
            raise LLMError(f"Gemini request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        text = getattr(resp, "text", None) or ""
        if not text.strip():
            raise LLMError("Gemini returned an empty answer")
        usage = getattr(resp, "usage_metadata", None)
        finish = None
        if getattr(resp, "candidates", None):
            finish = str(resp.candidates[0].finish_reason)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            latency_ms=latency,
            finish_reason=finish,
        )
