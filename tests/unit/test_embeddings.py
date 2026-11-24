from __future__ import annotations

import json
import math

import httpx
import pytest
import respx

from researchmind.embeddings import EmbeddingError, HashingEmbedder, OllamaEmbedder
from researchmind.retrieval.reranker import NoReranker


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(256)
    a = e.embed_query("dense passage retrieval")
    assert a == e.embed_query("dense passage retrieval")
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)
    near = e.embed_documents(["dense passage retrieval for question answering"])[0]
    far = e.embed_documents(["protein folding with molecular dynamics"])[0]
    assert _cos(a, near) > _cos(a, far)


@respx.mock
def test_ollama_embedder_prefixes_and_dim_check():
    route = respx.post("http://o/api/embed").mock(
        side_effect=lambda req: httpx.Response(
            200, json={"embeddings": [[1.0, 0.0, 0.0]] * len(json.loads(req.content)["input"])}
        )
    )
    e = OllamaEmbedder("nomic-embed-text", 3, "http://o")
    e.embed_documents(["a", "b"])
    e.embed_query("q")
    inputs = [json.loads(c.request.content)["input"] for c in route.calls]
    assert inputs[0] == ["search_document: a", "search_document: b"]
    assert inputs[1] == ["search_query: q"]
    with pytest.raises(EmbeddingError):
        OllamaEmbedder("nomic-embed-text", 768, "http://o").embed_query("q")


def test_no_reranker_keeps_order():
    s = NoReranker().score("q", ["a", "b", "c"])
    assert s == sorted(s, reverse=True)
