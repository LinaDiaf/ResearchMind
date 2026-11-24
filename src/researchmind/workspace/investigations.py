"""Running, persisting and resuming investigations."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from researchmind.agent.graph import AgentDeps, ResearchAgent, build_graph, recursion_limit
from researchmind.agent.prompts import PROMPT_VERSION
from researchmind.config import AgentBudgetSettings
from researchmind.db.models import (
    ClaimCitation,
    ClaimType,
    EvidenceRecord,
    Investigation,
    InvestigationEvent,
    InvestigationStatus,
    LLMCall,
    Note,
    NoteKind,
    Paper,
    ProjectPaper,
    ReportClaim,
    SupportLabel,
)
from researchmind.llm import LLMClient, LLMProvider, UsageCounter
from researchmind.llm.client import CallRecord
from researchmind.log import get_logger
from researchmind.retrieval.engine import RetrievalEngine

log = get_logger(__name__)
EventCallback = Callable[[dict[str, Any]], None]


def make_postgres_checkpointer(conninfo: str):
    """Postgres checkpointer backed by a connection pool (safe across threads)."""
    import atexit

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(conninfo, max_size=8, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0,
                                  "row_factory": dict_row})
    atexit.register(pool.close)
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


class InvestigationRunner:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        engine: RetrievalEngine,
        provider: LLMProvider,
        budget: AgentBudgetSettings,
        checkpointer,
    ) -> None:
        self.sf = session_factory
        self.engine = engine
        self.provider = provider
        self.budget = budget
        self.checkpointer = checkpointer

    # ------------------------------------------------------------------ setup
    def create(self, project_id: uuid.UUID, question: str,
               paper_ids: list[uuid.UUID] | None = None) -> uuid.UUID:
        with self.sf.begin() as s:
            inv = Investigation(
                project_id=project_id, question=question.strip(),
                status=InvestigationStatus.interrupted,
                config={"provider": self.provider.provider, "model": self.provider.model,
                        "paper_ids": [str(p) for p in paper_ids] if paper_ids else None,
                        "embedding_model": self.engine.embedder.model,
                        "reranker": getattr(self.engine.reranker, "name", "none"),
                        "budget": self.budget.model_dump(), "prompt_version": PROMPT_VERSION})
            s.add(inv)
            s.flush()
            return inv.id

    def _titles(self, project_id: str) -> list[str]:
        with self.sf() as s:
            return list(s.scalars(select(Paper.title).join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                                  .where(ProjectPaper.project_id == uuid.UUID(project_id))))

    def _recorder(self, investigation_id: uuid.UUID):
        def record(rec: CallRecord) -> None:
            with self.sf.begin() as s:
                s.add(LLMCall(investigation_id=investigation_id, node=rec.node,
                              provider=rec.provider, model=rec.model,
                              prompt_tokens=rec.prompt_tokens,
                              completion_tokens=rec.completion_tokens,
                              latency_ms=rec.latency_ms, cost_usd=rec.cost_usd, status=rec.status))
        return record

    def _graph(self, investigation_id: uuid.UUID, calls_so_far: int):
        usage = UsageCounter(max_calls=self.budget.max_llm_calls, calls=calls_so_far)
        llm = LLMClient(self.provider, usage=usage, recorder=self._recorder(investigation_id))
        agent = ResearchAgent(AgentDeps(llm=llm, retrieve=self.engine.retrieve, budget=self.budget,
                                        library_titles=self._titles))
        return build_graph(agent, self.checkpointer)

    def _config(self, investigation_id: uuid.UUID) -> dict[str, Any]:
        return {"configurable": {"thread_id": str(investigation_id)},
                "recursion_limit": recursion_limit(self.budget)}

    # -------------------------------------------------------------------- run
    def state(self, investigation_id: uuid.UUID) -> dict[str, Any]:
        graph = self._graph(investigation_id, 0)
        snap = graph.get_state(self._config(investigation_id))
        return dict(snap.values) if snap else {}

    def run(self, investigation_id: uuid.UUID, on_event: EventCallback | None = None) -> dict[str, Any]:
        """Run a new investigation or resume an interrupted one from its last checkpoint."""
        with self.sf() as s:
            inv = s.get(Investigation, investigation_id)
            if inv is None:
                raise ValueError(f"unknown investigation {investigation_id}")
            if inv.status is InvestigationStatus.completed and inv.report:
                return inv.report
            question, project_id = inv.question, inv.project_id
            scope = (inv.config or {}).get("paper_ids")
            seq = s.scalar(select(func.coalesce(func.max(InvestigationEvent.seq), 0))
                           .where(InvestigationEvent.investigation_id == investigation_id)) or 0

        config = self._config(investigation_id)
        probe = build_graph(ResearchAgent(AgentDeps(llm=LLMClient(self.provider),
                                                    retrieve=self.engine.retrieve,
                                                    budget=self.budget)), self.checkpointer)
        snap = probe.get_state(config)
        resuming = bool(snap and snap.values)
        calls = int(snap.values.get("llm_calls", 0)) if resuming else 0
        graph = self._graph(investigation_id, calls)
        payload = None if resuming else {
            "investigation_id": str(investigation_id), "project_id": str(project_id),
            "question": question, "subquestions": [], "evidence": {}, "critique_done": False,
            "config": {"paper_ids": scope},
            "llm_calls": 0, "elapsed_s": 0.0}
        self._set_status(investigation_id, InvestigationStatus.running)
        if resuming:
            seq = self._persist_events(investigation_id, seq, [{
                "node": "runner", "type": "resume", "message": "Resumed from last checkpoint",
                "data": {"next": list(snap.next)}}], on_event)
        try:
            for chunk in graph.stream(payload, config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    seq = self._persist_events(investigation_id, seq, update.get("events", []),
                                               on_event, node_name)
        except Exception as exc:
            log.exception("investigation_interrupted", investigation_id=str(investigation_id))
            with self.sf.begin() as s:
                inv = s.get(Investigation, investigation_id)
                inv.status, inv.error = InvestigationStatus.interrupted, f"{type(exc).__name__}: {exc}"
            raise
        final = graph.get_state(config).values
        report = dict(final.get("report") or {})
        report["stats"] = self._stats(final)
        self._persist_results(investigation_id, final, report)
        return report

    @staticmethod
    def _stats(state: dict[str, Any]) -> dict[str, Any]:
        log_ = state.get("retrieval_log", [])
        return {"llm_calls": state.get("llm_calls", 0), "elapsed_s": round(state.get("elapsed_s", 0.0), 1),
                "retrieval_rounds": len(log_), "subquestions": len(state.get("subquestions", [])),
                "evidence_retrieved": len(state.get("evidence", {})),
                "stop_reason": state.get("stop_reason")}

    def _set_status(self, investigation_id: uuid.UUID, status: InvestigationStatus) -> None:
        with self.sf.begin() as s:
            inv = s.get(Investigation, investigation_id)
            inv.status = status
            if status is InvestigationStatus.running:
                inv.error = None

    def _persist_events(self, investigation_id, seq, events, on_event, node_name=None) -> int:
        if not events:
            return seq
        with self.sf.begin() as s:
            for ev in events:
                seq += 1
                s.add(InvestigationEvent(investigation_id=investigation_id, seq=seq,
                                         node=ev.get("node") or node_name or "?",
                                         event_type=ev.get("type", "info"),
                                         message=ev.get("message", "")[:4000],
                                         payload=ev.get("data", {})))
        if on_event:
            for ev in events:
                try:
                    on_event(ev)
                except Exception:
                    log.warning("event_callback_failed")
        return seq

    def _persist_results(self, investigation_id: uuid.UUID, state: dict[str, Any],
                         report: dict[str, Any]) -> None:
        evidence = state.get("evidence", {})
        with self.sf.begin() as s:
            s.execute(delete(ReportClaim).where(ReportClaim.investigation_id == investigation_id))
            s.execute(delete(EvidenceRecord).where(EvidenceRecord.investigation_id == investigation_id))
            rows: dict[str, EvidenceRecord] = {}
            for label, ev in evidence.items():
                r = EvidenceRecord(
                    investigation_id=investigation_id, label=label,
                    subquestion_id=(ev.get("subquestion_ids") or ["?"])[0],
                    chunk_id=uuid.UUID(ev["chunk_id"]), retrieval_round=ev.get("round", 1),
                    strategy=ev.get("strategy", "hybrid"), scores=ev.get("scores", {}),
                    relevance=ev.get("relevance"), stance=ev.get("stance"),
                    rationale=ev.get("grade_note"), injection_flag=ev.get("injection_flag", False))
                s.add(r)
                rows[label] = r
            s.flush()
            order = 0
            for sec in report.get("sections", []):
                for c in sec["claims"]:
                    order += 1
                    claim = ReportClaim(investigation_id=investigation_id, ord=order,
                                        section=sec["heading"], text=c["text"],
                                        claim_type=ClaimType(c["claim_type"]),
                                        support=SupportLabel(c["support"]))
                    s.add(claim)
                    s.flush()
                    for label in c["citations"]:
                        if label in rows:
                            s.add(ClaimCitation(claim_id=claim.id, evidence_id=rows[label].id))
            inv = s.get(Investigation, investigation_id)
            inv.report = report
            inv.status = InvestigationStatus.completed
            inv.error = None


class Workspace:
    """Read side of investigations, plus research notes and open questions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.sf = session_factory

    def investigations(self, project_id: uuid.UUID) -> list[Investigation]:
        with self.sf() as s:
            return list(s.scalars(select(Investigation).where(Investigation.project_id == project_id)
                                  .order_by(Investigation.created_at.desc())))

    def investigation(self, investigation_id: uuid.UUID) -> Investigation | None:
        with self.sf() as s:
            return s.get(Investigation, investigation_id)

    def events(self, investigation_id: uuid.UUID) -> list[InvestigationEvent]:
        with self.sf() as s:
            return list(s.scalars(select(InvestigationEvent)
                                  .where(InvestigationEvent.investigation_id == investigation_id)
                                  .order_by(InvestigationEvent.seq)))

    def llm_usage(self, investigation_id: uuid.UUID) -> dict[str, Any]:
        with self.sf() as s:
            row = s.execute(select(func.count(), func.coalesce(func.sum(LLMCall.prompt_tokens), 0),
                                   func.coalesce(func.sum(LLMCall.completion_tokens), 0),
                                   func.coalesce(func.sum(LLMCall.latency_ms), 0.0),
                                   func.coalesce(func.sum(LLMCall.cost_usd), 0.0))
                            .where(LLMCall.investigation_id == investigation_id)).one()
        return {"calls": row[0], "prompt_tokens": row[1], "completion_tokens": row[2],
                "llm_seconds": round(row[3] / 1000, 1), "cost_usd": round(row[4], 4)}

    def add_note(self, project_id: uuid.UUID, body: str, kind: NoteKind = NoteKind.note,
                 investigation_id: uuid.UUID | None = None) -> uuid.UUID:
        with self.sf.begin() as s:
            n = Note(project_id=project_id, body=body.strip(), kind=kind,
                     investigation_id=investigation_id)
            s.add(n)
            s.flush()
            return n.id

    def notes(self, project_id: uuid.UUID) -> list[Note]:
        with self.sf() as s:
            return list(s.scalars(select(Note).where(Note.project_id == project_id)
                                  .order_by(Note.created_at.desc())))

    def delete_note(self, note_id: uuid.UUID) -> None:
        with self.sf.begin() as s:
            s.execute(delete(Note).where(Note.id == note_id))

    def reusable_evidence(self, project_id: uuid.UUID, limit: int = 200) -> list[dict[str, Any]]:
        """Evidence judged relevant in earlier investigations of the project."""
        from researchmind.db.models import Chunk

        with self.sf() as s:
            rows = s.execute(
                select(EvidenceRecord, Investigation.question, Chunk, Paper)
                .join(Investigation, Investigation.id == EvidenceRecord.investigation_id)
                .join(Chunk, Chunk.id == EvidenceRecord.chunk_id)
                .join(Paper, Paper.id == Chunk.paper_id)
                .where(Investigation.project_id == project_id,
                       EvidenceRecord.relevance.in_(["high", "partial"]))
                .order_by(Investigation.created_at.desc()).limit(limit)).all()
        return [{"label": e.label, "question": q, "relevance": e.relevance, "stance": e.stance,
                 "note": e.rationale, "paper_title": p.title, "year": p.year,
                 "pages": f"{c.page_start}-{c.page_end}", "text": c.text,
                 "chunk_id": str(c.id), "investigation_id": str(e.investigation_id)}
                for e, q, c, p in rows]
