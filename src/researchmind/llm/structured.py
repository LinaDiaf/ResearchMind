"""Typed LLM outputs: schema generation, tolerant JSON parsing, one bounded repair retry.

Small local models (e.g. Qwen 3.5 9B) occasionally wrap JSON in prose or code fences, or stop
mid-object. Parsing is tolerant to the first two; validation errors trigger one repair attempt
in which the model sees its own answer and the validation error.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > 30:
        raise ValueError("schema nesting too deep (recursive model?)")
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            resolved = copy.deepcopy(defs[name])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            resolved.update(extra)
            return _inline_refs(resolved, defs, depth + 1)
        return {k: _inline_refs(v, defs, depth + 1) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Self-contained JSON schema (no ``$ref``), accepted by Ollama, Gemini and vLLM."""
    schema = model.model_json_schema()
    return _inline_refs(schema, schema.get("$defs", {}))


def extract_json(text: str) -> Any:
    """Parse the first JSON value in ``text`` (plain, fenced, or surrounded by prose)."""
    text = text.strip()
    # Some reasoning models emit <think>...</think> blocks even when asked not to.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for block in _FENCE.findall(text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise StructuredOutputError("no valid JSON found in model output", raw=text)


def parse_as(model: type[T], text: str) -> T:
    data = extract_json(text)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(_short_errors(exc), raw=text) from exc


def _short_errors(exc: ValidationError, limit: int = 6) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def repair_instruction(error: str) -> str:
    return (
        "Your previous answer could not be used: "
        f"{error}. Reply again with ONLY a JSON object that matches the required schema. "
        "No prose, no code fences."
    )
