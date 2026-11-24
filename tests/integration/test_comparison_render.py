from __future__ import annotations

import re

import pytest

from researchmind.db.models import Chunk, Paper
from researchmind.db.session import get_sessionmaker
from researchmind.embeddings import HashingEmbedder
from researchmind.ingestion.render import render_highlight
from researchmind.ingestion.service import Library
from researchmind.llm import LLMClient
from researchmind.llm.fake import FakeLLM
from researchmind.retrieval.engine import RetrievalEngine
from tests.fixtures.synthetic_papers import build_corpus

pytestmark = pytest.mark.integration


@pytest.fixture()
def env(db, tmp_settings, tmp_path):
    sf = get_sessionmaker(tmp_settings.database_url)
    lib = Library(sf, HashingEmbedder(tmp_settings.embedding_dim), tmp_settings)
    pid = lib.create_project("Cmp")
    res = {k: lib.ingest_pdf(pid, p, reindex=False) for k, p in build_corpus(tmp_path / "pdfs").items()}
    lib.reindex(pid)
    return sf, RetrievalEngine(sf, lib.embedder, None, tmp_settings.index_dir), pid, res


def test_comparison_cells_cite_only_their_paper(env):
    from researchmind.comparison import Comparator

    sf, engine, pid, res = env

    def handler(messages, schema):
        text = messages[-1].content
        ids = re.findall(r'<passage id="(P\d+)"', text)
        return {"cells": [
            {"dimension": "Main results", "value": "MRR@10 of 38.2 on MS MARCO", "evidence_ids": ids[:1] + ["P999"],
             "status": "found"},
            {"dimension": "Limitations", "value": "", "evidence_ids": [], "status": "not_reported"}]}

    cmp = Comparator(engine.retrieve, LLMClient(FakeLLM(handler)))
    out = cmp.compare(pid, [(res["sparse"].paper_id, "A"), (res["dense"].paper_id, "B")],
                      ["Main results", "Limitations"])
    cell = out["rows"]["A"]["Main results"]
    assert cell["status"] == "found" and "P999" not in cell["citations"]
    for eid in cell["citations"]:
        assert out["evidence"][eid]["paper_id"] == str(res["sparse"].paper_id)
    assert out["rows"]["A"]["Limitations"]["status"] == "not_reported"

    extractive = Comparator(engine.retrieve, None).compare(pid, [(res["dense"].paper_id, "B")], ["Main results"])
    assert extractive["rows"]["B"]["Main results"]["source"] == "extractive"


def test_page_render_highlights_passage(env):
    sf, _, _, res = env
    with sf() as s:
        chunk = s.query(Chunk).filter(Chunk.paper_id == res["sparse"].paper_id,
                                      Chunk.text.contains("38.2")).first()
        paper = s.get(Paper, chunk.paper_id)
    png, found = render_highlight(paper.pdf_path, chunk.page_start, chunk.text)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and found > 0
