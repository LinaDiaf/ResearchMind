"""SQLAlchemy ORM models. The schema is created by Alembic migrations (see migrations/)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _enum(cls: type[enum.Enum], name: str) -> Enum:
    return Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])


class PaperStatus(enum.StrEnum):
    metadata_only = "metadata_only"
    parsing = "parsing"
    ingested = "ingested"
    failed = "failed"


class SectionKind(enum.StrEnum):
    abstract = "abstract"
    introduction = "introduction"
    related_work = "related_work"
    method = "method"
    experiments = "experiments"
    results = "results"
    discussion = "discussion"
    limitations = "limitations"
    conclusion = "conclusion"
    references = "references"
    appendix = "appendix"
    other = "other"


class InvestigationStatus(enum.StrEnum):
    running = "running"
    interrupted = "interrupted"
    awaiting_input = "awaiting_input"
    completed = "completed"
    failed = "failed"


class ClaimType(enum.StrEnum):
    reported_finding = "reported_finding"
    author_interpretation = "author_interpretation"
    ai_synthesis = "ai_synthesis"
    hypothesis = "hypothesis"


class SupportLabel(enum.StrEnum):
    supported = "supported"
    partially_supported = "partially_supported"
    unsupported = "unsupported"
    not_checked = "not_checked"
    uncited = "uncited"


class NoteKind(enum.StrEnum):
    note = "note"
    open_question = "open_question"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text)
    title_norm: Mapped[str] = mapped_column(Text, index=True)
    abstract: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    venue: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    openalex_id: Mapped[str | None] = mapped_column(String(64))
    s2_id: Mapped[str | None] = mapped_column(String(64))
    url: Mapped[str | None] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32))
    pdf_path: Mapped[str | None] = mapped_column(Text)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[PaperStatus] = mapped_column(
        _enum(PaperStatus, "paper_status"), default=PaperStatus.metadata_only
    )
    parse_warnings: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sections: Mapped[list[Section]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Section.ord"
    )


class ProjectPaper(Base):
    __tablename__ = "project_papers"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    level: Mapped[int] = mapped_column(Integer, default=1)
    kind: Mapped[SectionKind] = mapped_column(_enum(SectionKind, "section_kind"))
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)

    paper: Mapped[Paper] = relationship(back_populates="sections")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_paper_ord", "paper_id", "ord"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sections.id", ondelete="SET NULL")
    )
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(Text)
    section_kind: Mapped[SectionKind] = mapped_column(_enum(SectionKind, "section_kind"))
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    is_reference_list: Mapped[bool] = mapped_column(Boolean, default=False)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[Any] = mapped_column(Vector())


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[InvestigationStatus] = mapped_column(
        _enum(InvestigationStatus, "investigation_status"), default=InvestigationStatus.running
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class InvestigationEvent(Base):
    __tablename__ = "investigation_events"
    __table_args__ = (UniqueConstraint("investigation_id", "seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvidenceRecord(Base):
    __tablename__ = "evidence_items"
    __table_args__ = (UniqueConstraint("investigation_id", "label"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(16))  # "E12": the id the LLM sees and cites
    subquestion_id: Mapped[str] = mapped_column(String(16))
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    retrieval_round: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(32))
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    relevance: Mapped[str | None] = mapped_column(String(16))
    stance: Mapped[str | None] = mapped_column(String(16))
    rationale: Mapped[str | None] = mapped_column(Text)
    injection_flag: Mapped[bool] = mapped_column(Boolean, default=False)


class ReportClaim(Base):
    __tablename__ = "report_claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[ClaimType] = mapped_column(_enum(ClaimType, "claim_type"))
    support: Mapped[SupportLabel] = mapped_column(_enum(SupportLabel, "support_label"))


class ClaimCitation(Base):
    __tablename__ = "claim_citations"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("report_claims.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evidence_items.id", ondelete="CASCADE"), primary_key=True
    )


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("investigations.id", ondelete="SET NULL")
    )
    kind: Mapped[NoteKind] = mapped_column(_enum(NoteKind, "note_kind"), default=NoteKind.note)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), index=True
    )
    node: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
