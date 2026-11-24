"""Dense retrieval with pgvector (cosine distance)."""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from researchmind.retrieval.types import RetrievalFilters


def _filter_sql(f: RetrievalFilters, params: dict) -> str:
    clauses = []
    if not f.include_references:
        clauses.append("c.is_reference_list = false")
    if f.paper_ids:
        clauses.append("c.paper_id = ANY(:paper_ids)")
        params["paper_ids"] = list(f.paper_ids)
    if f.year_min is not None:
        clauses.append("p.year >= :year_min")
        params["year_min"] = f.year_min
    if f.year_max is not None:
        clauses.append("p.year <= :year_max")
        params["year_max"] = f.year_max
    if f.section_kinds:
        clauses.append("c.section_kind::text = ANY(:kinds)")
        params["kinds"] = [k.value for k in f.section_kinds]
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def dense_search(
    session: Session,
    project_id: uuid.UUID,
    query_vector: list[float],
    model: str,
    filters: RetrievalFilters,
    limit: int,
) -> list[tuple[uuid.UUID, float]]:
    params: dict = {"pid": project_id, "model": model, "q": str(query_vector), "limit": limit}
    where = _filter_sql(filters, params)
    session.execute(text("SET LOCAL hnsw.ef_search = 200"))
    rows = session.execute(
        text(
            "SELECT c.id, 1 - (e.embedding <=> CAST(:q AS vector)) AS score "
            "FROM chunk_embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "JOIN papers p ON p.id = c.paper_id "
            "JOIN project_papers pp ON pp.paper_id = c.paper_id AND pp.project_id = :pid "
            f"WHERE e.model = :model{where} "
            "ORDER BY e.embedding <=> CAST(:q AS vector) LIMIT :limit"
        ),
        params,
    ).all()
    return [(r[0], float(r[1])) for r in rows]


def has_embeddings(session: Session, project_id: uuid.UUID, model: str) -> bool:
    return bool(
        session.execute(
            text(
                "SELECT 1 FROM chunk_embeddings e JOIN chunks c ON c.id = e.chunk_id "
                "JOIN project_papers pp ON pp.paper_id = c.paper_id AND pp.project_id = :pid "
                "WHERE e.model = :model LIMIT 1"
            ),
            {"pid": project_id, "model": model},
        ).first()
    )
