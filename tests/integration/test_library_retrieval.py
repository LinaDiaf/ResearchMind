from __future__ import annotations

import pytest

from researchmind.db.models import Chunk, Paper, PaperStatus, SectionKind
from researchmind.db.session import get_sessionmaker
from researchmind.discovery.models import PaperRecord
from researchmind.embeddings import HashingEmbedder
from researchmind.ingestion.service import Library
from researchmind.retrieval.engine import RetrievalEngine
from researchmind.retrieval.types import RetrievalFilters, RetrievalRequest
from tests.fixtures.synthetic_papers import build_corpus

pytestmark = pytest.mark.integration


@pytest.fixture()
def lib(db, tmp_settings, tmp_path):
    sf = get_sessionmaker(tmp_settings.database_url)
    library = Library(sf, HashingEmbedder(tmp_settings.embedding_dim), tmp_settings)
    corpus = build_corpus(tmp_path / "pdfs")
    pid = library.create_project("Retrieval study")
    results = {k: library.ingest_pdf(pid, p, reindex=False) for k, p in corpus.items()}
    library.reindex(pid)
    engine = RetrievalEngine(sf, library.embedder, None, tmp_settings.index_dir)
    return library, engine, pid, corpus, results, sf


def test_ingestion_persists_provenance(lib):
    library, _, pid, _, results, sf = lib
    assert all(r.status == "ingested" and r.n_chunks > 5 for r in results.values())
    with sf() as s:
        p = s.get(Paper, results["sparse"].paper_id)
        assert p.status is PaperStatus.ingested and p.arxiv_id == "2401.00001"
        assert p.abstract and "SPARTA-X" in p.abstract
        assert p.page_count == 2
    assert len(library.project_papers(pid)) == 3


def test_reingest_is_detected_as_duplicate(lib):
    library, _, pid, corpus, results, _ = lib
    again = library.ingest_pdf(pid, corpus["dense"])
    assert again.status == "duplicate" and again.paper_id == results["dense"].paper_id
    rec = PaperRecord(title="Synthetic Study B: Dense Passage Retrieval for Question Answering",
                      doi="10.5555/SYNTHETIC.0002", source="openalex", year=2024)
    imported = library.import_record(pid, rec, download_pdf=False)
    assert imported.status == "duplicate" and imported.duplicate_rule == "doi"


def test_metadata_only_import_is_searchable(lib):
    library, engine, pid, _, _, _ = lib
    rec = PaperRecord(title="Quantum annealing for portfolio optimisation", year=2022,
                      abstract="We apply quantum annealing hardware to portfolio optimisation.",
                      source="crossref")
    res = library.import_record(pid, rec, download_pdf=False)
    assert res.status == "metadata_only"
    out = engine.retrieve(RetrievalRequest(project_id=pid, queries=["quantum annealing portfolio"],
                                           strategy="lexical", k=3))
    assert out.passages[0].paper_id == res.paper_id and out.passages[0].page_start == 0


def test_lexical_finds_exact_result_with_page(lib):
    _, engine, pid, _, results, _ = lib
    out = engine.retrieve(RetrievalRequest(project_id=pid, queries=["SPARTA-X MRR@10 MS MARCO"],
                                           strategy="lexical", k=3))
    top = out.passages[0]
    assert top.paper_id == results["sparse"].paper_id
    assert "38.2" in top.text and top.page_start == 2


def test_filters_section_kind_and_papers(lib):
    _, engine, pid, _, results, _ = lib
    out = engine.retrieve(RetrievalRequest(
        project_id=pid, queries=["limitations memory index size"], strategy="hybrid", k=10,
        filters=RetrievalFilters(section_kinds=[SectionKind.limitations])))
    assert out.passages and all(p.section_kind is SectionKind.limitations for p in out.passages)
    only = engine.retrieve(RetrievalRequest(
        project_id=pid, queries=["retrieval"], k=10,
        filters=RetrievalFilters(paper_ids=[results["hybrid"].paper_id])))
    assert {p.paper_id for p in only.passages} == {results["hybrid"].paper_id}


def test_references_excluded_by_default(lib):
    _, engine, pid, _, _, sf = lib
    out = engine.retrieve(RetrievalRequest(project_id=pid, queries=["probabilistic relevance framework"],
                                           k=20))
    with sf() as s:
        ref_ids = {c.id for c in s.query(Chunk).filter(Chunk.is_reference_list.is_(True))}
    assert not ref_ids & {p.chunk_id for p in out.passages}


def test_dense_and_hybrid_return_scored_passages(lib):
    _, engine, pid, _, _, _ = lib
    dense = engine.retrieve(RetrievalRequest(project_id=pid, queries=["dual encoder question answering"],
                                             strategy="dense", k=5))
    assert dense.n_dense > 0 and all("dense" in p.scores for p in dense.passages)
    hybrid = engine.retrieve(RetrievalRequest(project_id=pid, queries=["hybrid recall BEIR"], k=5,
                                              max_per_paper=2))
    assert hybrid.n_dense > 0 and hybrid.n_lexical > 0
    per_paper = {}
    for p in hybrid.passages:
        per_paper[p.paper_id] = per_paper.get(p.paper_id, 0) + 1
    assert max(per_paper.values()) <= 2


def test_exclusion_of_seen_chunks(lib):
    _, engine, pid, _, _, _ = lib
    req = RetrievalRequest(project_id=pid, queries=["BM25 baseline"], strategy="lexical", k=3)
    first = engine.retrieve(req)
    req2 = req.model_copy(update={"exclude_chunk_ids": [p.chunk_id for p in first.passages]})
    second = engine.retrieve(req2)
    assert not {p.chunk_id for p in first.passages} & {p.chunk_id for p in second.passages}
