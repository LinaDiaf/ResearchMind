"""End-to-end investigation on the synthetic corpus: real database, real retrieval engine,
real Postgres checkpointer; the LLM is scripted."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from researchmind.config import AgentBudgetSettings
from researchmind.db.models import (
    Chunk,
    ClaimCitation,
    EvidenceRecord,
    InvestigationStatus,
    LLMCall,
    ReportClaim,
)
from researchmind.db.session import get_sessionmaker
from researchmind.embeddings import HashingEmbedder
from researchmind.ingestion.service import Library
from researchmind.retrieval.engine import RetrievalEngine
from researchmind.workspace.investigations import (
    InvestigationRunner,
    Workspace,
    make_postgres_checkpointer,
)
from tests.fixtures.agent_script import ScriptedLLM, claim_ids, passage_ids
from tests.fixtures.synthetic_papers import build_corpus

pytestmark = pytest.mark.integration


def scripted():
    def grade(p):
        return {"grades": [{"evidence_id": e, "relevance": "high", "stance": "neutral",
                            "aspects": ["reported results", "datasets and metrics", "method"]}
                           for e in passage_ids(p)], "missing_aspects": []}

    def synth(p):
        ids = passage_ids(p)
        return {"title": "Sparse vs dense retrieval", "sections": [{"heading": "Answer", "claims": [
            {"text": "SPARTA-X reaches an MRR@10 of 38.2 on MS MARCO.", "claim_type": "reported_finding",
             "evidence_ids": ids[:1]},
            {"text": "The two approaches were evaluated on different benchmarks.",
             "claim_type": "ai_synthesis", "evidence_ids": ids[:2]}]}]}

    return ScriptedLLM({
        "analyze": lambda p: {"intent": "method_comparison", "complexity": "complex",
                              "key_terms": ["SPARTA-X", "DenseQA", "MRR@10"]},
        "plan": lambda p: {"subquestions": [
            {"question": "What results does SPARTA-X report on MS MARCO?", "priority": 1,
             "required_aspects": ["reported results"], "target_sections": ["results"],
             "search_queries": ["SPARTA-X MRR@10 MS MARCO"]},
            {"question": "What results does DenseQA report on Natural Questions?", "priority": 2,
             "required_aspects": ["reported results"], "target_sections": ["results"],
             "search_queries": ["DenseQA top-20 accuracy Natural Questions"]}]},
        "grade": grade,
        "critique": lambda p: {"issues": ["different benchmarks"], "contradictions": [], "followups": []},
        "synthesize": synth,
        "verify": lambda p: {"verdicts": [{"claim_id": c, "support": "supported"} for c in claim_ids(p)]},
    })


@pytest.fixture()
def setup(db, tmp_settings, tmp_path):
    sf = get_sessionmaker(tmp_settings.database_url)
    lib = Library(sf, HashingEmbedder(tmp_settings.embedding_dim), tmp_settings)
    pid = lib.create_project("Comparison")
    for p in build_corpus(tmp_path / "pdfs").values():
        lib.ingest_pdf(pid, p, reindex=False)
    lib.reindex(pid)
    engine = RetrievalEngine(sf, lib.embedder, None, tmp_settings.index_dir)
    saver = make_postgres_checkpointer(tmp_settings.psycopg_url)
    return sf, engine, pid, saver


def test_investigation_runs_and_persists_provenance(setup):
    sf, engine, pid, saver = setup
    llm = scripted()
    runner = InvestigationRunner(sf, engine, llm, AgentBudgetSettings(), saver)
    inv_id = runner.create(pid, "How do SPARTA-X and DenseQA compare?")
    events = []
    report = runner.run(inv_id, on_event=events.append)
    assert report["sections"][0]["claims"][0]["support"] == "supported"
    ws = Workspace(sf)
    inv = ws.investigation(inv_id)
    assert inv.status is InvestigationStatus.completed and inv.report["title"]
    assert [e.event_type for e in ws.events(inv_id)][:2] == ["analysis", "plan"]
    assert ws.llm_usage(inv_id)["calls"] == llm.counts.total()
    with sf() as s:
        claim = s.scalars(select(ReportClaim).where(ReportClaim.investigation_id == inv_id)
                          .order_by(ReportClaim.ord)).first()
        cit = s.scalars(select(ClaimCitation).where(ClaimCitation.claim_id == claim.id)).first()
        ev = s.get(EvidenceRecord, cit.evidence_id)
        chunk = s.get(Chunk, ev.chunk_id)
        # the cited passage is the real chunk with the reported number, on page 2
        assert "38.2" in chunk.text and chunk.page_start == 2
        assert s.query(LLMCall).filter(LLMCall.investigation_id == inv_id).count() > 0
    assert runner.run(inv_id)["title"] == report["title"]  # completed → returned as stored


def test_interrupted_investigation_resumes_from_checkpoint(setup):
    sf, engine, pid, saver = setup
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return engine.retrieve(req)

    llm = scripted()
    runner = InvestigationRunner(sf, engine, llm, AgentBudgetSettings(), saver)
    runner.engine = type("E", (), {"retrieve": staticmethod(flaky), "embedder": engine.embedder,
                                   "reranker": engine.reranker})()
    inv_id = runner.create(pid, "How do SPARTA-X and DenseQA compare?")
    with pytest.raises(RuntimeError):
        runner.run(inv_id)
    ws = Workspace(sf)
    assert ws.investigation(inv_id).status is InvestigationStatus.interrupted
    analyze_calls = llm.counts["analyze"]
    report = runner.run(inv_id)
    assert llm.counts["analyze"] == analyze_calls  # finished steps are not re-run
    assert any(e.event_type == "resume" for e in ws.events(inv_id))
    assert ws.investigation(inv_id).status is InvestigationStatus.completed
    assert report["references"]


def test_notes_and_reusable_evidence(setup):
    sf, engine, pid, saver = setup
    runner = InvestigationRunner(sf, engine, scripted(), AgentBudgetSettings(), saver)
    inv_id = runner.create(pid, "How do SPARTA-X and DenseQA compare?")
    runner.run(inv_id)
    ws = Workspace(sf)
    from researchmind.db.models import NoteKind

    ws.add_note(pid, "Check results on BEIR", NoteKind.open_question, inv_id)
    assert ws.notes(pid)[0].kind is NoteKind.open_question
    reusable = ws.reusable_evidence(pid)
    assert reusable and all(r["relevance"] in ("high", "partial") for r in reusable)
