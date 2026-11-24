"""Deterministic scripted provider for tests and offline development.

The handler receives the messages and the requested JSON schema and returns the raw text the
"model" would produce. It never pretends to be a real model: its provider name is ``fake``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from researchmind.llm.base import LLMResponse, Message

Handler = Callable[[Sequence[Message], dict[str, Any] | None], str | dict | list]


class FakeLLM:
    provider = "fake"

    def __init__(self, handler: Handler, model: str = "scripted") -> None:
        self.model = model
        self.handler = handler
        self.calls: list[tuple[list[Message], dict[str, Any] | None]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), json_schema))
        out = self.handler(messages, json_schema)
        text = out if isinstance(out, str) else json.dumps(out)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=sum(len(m.content) for m in messages) // 4,
            completion_tokens=len(text) // 4,
        )
