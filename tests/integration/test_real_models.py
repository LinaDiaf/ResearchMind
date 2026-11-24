"""Retrieval with real (small) embedding and cross-encoder models. Runs in the CI `models` job."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.models]

if not os.environ.get("RUN_MODEL_TESTS"):
    pytest.skip("set RUN_MODEL_TESTS=1 (downloads small models)", allow_module_level=True)


def test_cross_encoder_ranks_relevant_passage_first():
    from researchmind.retrieval.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = rr.score("What MRR@10 does SPARTA-X reach on MS MARCO?", [
        "The index of SPARTA-X is three times larger than a BM25 index.",
        "On MS MARCO, SPARTA-X reaches an MRR@10 of 38.2, compared with 18.4 for BM25.",
    ])
    assert scores[1] > scores[0] and all(0 < s < 1 for s in scores)


def test_sentence_transformer_embedder_semantics():
    from researchmind.embeddings import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", 384)
    q = e.embed_query("How much GPU memory is needed to train the retriever?")
    d = e.embed_documents(["Training requires 8 GPUs with 32 GB of memory each.",
                           "We evaluate on the BEIR benchmark with 18 datasets."])
    dot = [sum(a * b for a, b in zip(q, v, strict=True)) for v in d]
    assert dot[0] > dot[1]
