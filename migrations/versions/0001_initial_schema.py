"""Initial schema: library, chunks, embeddings, investigations, evidence, notes.

Revision ID: 0001
Revises:
Create Date: 2026-11-24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

from researchmind.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ENUMS = {
    "paper_status": ("metadata_only", "parsing", "ingested", "failed"),
    "section_kind": (
        "abstract", "introduction", "related_work", "method", "experiments", "results",
        "discussion", "limitations", "conclusion", "references", "appendix", "other",
    ),
    "investigation_status": ("running", "interrupted", "awaiting_input", "completed", "failed"),
    "claim_type": ("reported_finding", "author_interpretation", "ai_synthesis", "hypothesis"),
    "support_label": ("supported", "partially_supported", "unsupported"),
    "note_kind": ("note", "open_question"),
}


def _enum(name: str) -> pg.ENUM:
    return pg.ENUM(*ENUMS[name], name=name, create_type=False)


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def _uuid_pk() -> sa.Column:
    return sa.Column("id", pg.UUID(as_uuid=True), primary_key=True)


def upgrade() -> None:
    dim = get_settings().embedding_dim
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    bind = op.get_bind()
    for name, values in ENUMS.items():
        pg.ENUM(*values, name=name).create(bind, checkfirst=True)

    op.create_table(
        "projects",
        _uuid_pk(),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "papers",
        _uuid_pk(),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("title_norm", sa.Text, nullable=False, index=True),
        sa.Column("abstract", sa.Text),
        sa.Column("year", sa.Integer, index=True),
        sa.Column("venue", sa.Text),
        sa.Column("authors", pg.JSONB, nullable=False, server_default="[]"),
        sa.Column("doi", sa.String(255), unique=True),
        sa.Column("arxiv_id", sa.String(64), unique=True),
        sa.Column("openalex_id", sa.String(64)),
        sa.Column("s2_id", sa.String(64)),
        sa.Column("url", sa.Text),
        sa.Column("pdf_url", sa.Text),
        sa.Column("license", sa.String(128)),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("pdf_path", sa.Text),
        sa.Column("pdf_sha256", sa.String(64), unique=True),
        sa.Column("page_count", sa.Integer),
        sa.Column("status", _enum("paper_status"), nullable=False),
        sa.Column("parse_warnings", pg.JSONB, nullable=False, server_default="[]"),
        _ts("created_at"),
    )
    op.create_table(
        "project_papers",
        sa.Column("project_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("paper_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
        _ts("added_at"),
    )
    op.create_table(
        "sections",
        _uuid_pk(),
        sa.Column("paper_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("level", sa.Integer, nullable=False),
        sa.Column("kind", _enum("section_kind"), nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
    )
    op.create_table(
        "chunks",
        _uuid_pk(),
        sa.Column("paper_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("sections.id", ondelete="SET NULL")),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("section_title", sa.Text),
        sa.Column("section_kind", _enum("section_kind"), nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("is_reference_list", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_chunks_paper_ord", "chunks", ["paper_id", "ord"])
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(128), primary_key=True),
        sa.Column("embedding", Vector(dim), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_table(
        "investigations",
        _uuid_pk(),
        sa.Column("project_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("status", _enum("investigation_status"), nullable=False),
        sa.Column("config", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("report", pg.JSONB),
        sa.Column("error", sa.Text),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "investigation_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("investigation_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False,
                  index=True),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("message", sa.Text, nullable=False, server_default=""),
        sa.Column("payload", pg.JSONB, nullable=False, server_default="{}"),
        _ts("created_at"),
        sa.UniqueConstraint("investigation_id", "seq"),
    )
    op.create_table(
        "evidence_items",
        _uuid_pk(),
        sa.Column("investigation_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False,
                  index=True),
        sa.Column("label", sa.String(16), nullable=False),
        sa.Column("subquestion_id", sa.String(16), nullable=False),
        sa.Column("chunk_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("retrieval_round", sa.Integer, nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("scores", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("relevance", sa.String(16)),
        sa.Column("stance", sa.String(16)),
        sa.Column("rationale", sa.Text),
        sa.Column("injection_flag", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("investigation_id", "label"),
    )
    op.create_table(
        "report_claims",
        _uuid_pk(),
        sa.Column("investigation_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False,
                  index=True),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("section", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("claim_type", _enum("claim_type"), nullable=False),
        sa.Column("support", _enum("support_label"), nullable=False),
    )
    op.create_table(
        "claim_citations",
        sa.Column("claim_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("report_claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("evidence_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("evidence_items.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "notes",
        _uuid_pk(),
        sa.Column("project_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("investigation_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="SET NULL")),
        sa.Column("kind", _enum("note_kind"), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        _ts("created_at"),
    )
    op.create_table(
        "llm_calls",
        _uuid_pk(),
        sa.Column("investigation_id", pg.UUID(as_uuid=True),
                  sa.ForeignKey("investigations.id", ondelete="CASCADE"), index=True),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False),
        _ts("created_at"),
    )


def downgrade() -> None:
    for table in (
        "llm_calls", "notes", "claim_citations", "report_claims", "evidence_items",
        "investigation_events", "investigations", "chunk_embeddings", "chunks", "sections",
        "project_papers", "papers", "projects",
    ):
        op.drop_table(table)
    bind = op.get_bind()
    for name in ENUMS:
        pg.ENUM(name=name).drop(bind, checkfirst=True)
