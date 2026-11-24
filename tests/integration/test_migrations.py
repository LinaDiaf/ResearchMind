from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def test_schema_created(db):
    tables = set(
        db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")).scalars()
    )
    for t in ("papers", "chunks", "chunk_embeddings", "investigations", "evidence_items",
              "report_claims", "claim_citations", "notes", "llm_calls"):
        assert t in tables
    ext = db.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar()
    assert ext == "vector"
    idx = db.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_chunk_embeddings_hnsw'")
    ).scalar()
    assert "hnsw" in idx
