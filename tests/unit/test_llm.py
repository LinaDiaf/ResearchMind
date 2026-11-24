from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from researchmind.llm import BudgetExceeded, LLMClient, LLMError, Message, UsageCounter
from researchmind.llm.fake import FakeLLM
from researchmind.llm.gemini import GeminiLLM
from researchmind.llm.ollama import OllamaLLM
from researchmind.llm.openai_compat import OpenAICompatLLM
from researchmind.llm.structured import (
    StructuredOutputError,
    extract_json,
    json_schema_for,
    parse_as,
)


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    items: list[Inner]
    score: float = Field(ge=0, le=1)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Here it is: {"a": 3} hope it helps') == {"a": 3}
    assert extract_json('<think>hmm {"x":0}</think>{"a": 4}') == {"a": 4}
    with pytest.raises(StructuredOutputError):
        extract_json("no json at all")
    with pytest.raises(StructuredOutputError):
        extract_json('{"a": 1')  # truncated


def test_schema_has_no_refs():
    schema = json_schema_for(Outer)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["items"]["items"]["properties"]["name"]["type"] == "string"


def test_parse_as_validation_error():
    with pytest.raises(StructuredOutputError) as e:
        parse_as(Outer, '{"items": [], "score": 3}')
    assert "score" in str(e.value)


def test_structured_repair_then_success():
    answers = iter(['{"items": [], "score": 7}', '{"items": [{"name": "x"}], "score": 0.5}'])
    fake = FakeLLM(lambda m, s: next(answers))
    records = []
    client = LLMClient(fake, recorder=records.append)
    out = client.structured("node", [Message("user", "q")], Outer)
    assert out.items[0].name == "x"
    assert [r.status for r in records] == ["invalid", "repaired"]
    # the repair turn shows the model its error
    assert "score" in fake.calls[1][0][-1].content


def test_structured_gives_up_after_repairs():
    client = LLMClient(FakeLLM(lambda m, s: "nope"))
    with pytest.raises(StructuredOutputError):
        client.structured("n", [Message("user", "q")], Outer, repair_attempts=1)
    assert client.usage.calls == 2


def test_budget_enforced():
    client = LLMClient(FakeLLM(lambda m, s: "ok"), usage=UsageCounter(max_calls=2))
    client.text("n", [Message("user", "a")])
    client.text("n", [Message("user", "b")])
    with pytest.raises(BudgetExceeded):
        client.text("n", [Message("user", "c")])


@respx.mock
def test_ollama_payload_and_parsing():
    route = respx.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "message": {"role": "assistant", "content": '{"name": "q"}'},
            "prompt_eval_count": 12, "eval_count": 5, "done_reason": "stop",
        })
    )
    llm = OllamaLLM("qwen3.5:9b", "http://ollama:11434", num_ctx=16384, think=False)
    resp = llm.complete([Message("system", "s"), Message("user", "u")],
                        json_schema=json_schema_for(Inner), max_tokens=100)
    body = json.loads(route.calls[0].request.content)
    assert body["think"] is False and body["stream"] is False
    assert body["options"]["num_ctx"] == 16384 and body["options"]["num_predict"] == 100
    assert body["format"]["properties"]["name"]["type"] == "string"
    assert (resp.prompt_tokens, resp.completion_tokens) == (12, 5)


@respx.mock
def test_ollama_errors():
    respx.post("http://o/api/chat").mock(return_value=httpx.Response(500))
    with pytest.raises(LLMError):
        OllamaLLM("m", "http://o").complete([Message("user", "u")])
    respx.post("http://o2/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": ""}, "done_reason": "length"})
    )
    with pytest.raises(LLMError):
        OllamaLLM("m", "http://o2").complete([Message("user", "u")])


@respx.mock
def test_openai_compat():
    route = respx.post("http://vllm/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
    )
    resp = OpenAICompatLLM("m", "http://vllm/v1", "k").complete(
        [Message("user", "u")], json_schema={"type": "object"}
    )
    body = json.loads(route.calls[0].request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    assert resp.text == "hi"


def test_gemini_maps_messages():
    captured = {}

    class Usage:
        prompt_token_count = 10
        candidates_token_count = 4

    class Resp:
        text = '{"name": "g"}'
        usage_metadata = Usage()
        candidates = []

    class Models:
        def generate_content(self, model, contents, config):
            captured.update(model=model, contents=contents, config=config)
            return Resp()

    class Client:
        models = Models()

    llm = GeminiLLM("gemini-2.5-flash", "", client=Client())
    resp = llm.complete([Message("system", "sys"), Message("user", "u"),
                         Message("assistant", "a")], json_schema={"type": "object"})
    assert captured["config"].system_instruction == "sys"
    assert [c.role for c in captured["contents"]] == ["user", "model"]
    assert captured["config"].response_mime_type == "application/json"
    assert resp.prompt_tokens == 10


def test_gemini_requires_key():
    with pytest.raises(LLMError):
        GeminiLLM("m", "")
