"""Retrieval request/result types shared by the engine and the agent."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from researchmind.db.models import SectionKind

Strategy = Literal["dense", "lexical", "hybrid"]


class RetrievalFilters(BaseModel):
    paper_ids: list[uuid.UUID] | None = None
    year_min: int | None = None
    year_max: int | None = None
    section_kinds: list[SectionKind] | None = None
    include_references: bool = False


class RetrievalRequest(BaseModel):
    project_id: uuid.UUID
    queries: list[str] = Field(min_length=1)
    strategy: Strategy = "hybrid"
    dense_weight: float = Field(1.0, ge=0)
    lexical_weight: float = Field(1.0, ge=0)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    k: int = Field(8, ge=1, le=50)
    candidates: int = Field(50, ge=5, le=500)
    rerank_candidates: int = Field(40, ge=1, le=200)
    max_per_paper: int = Field(4, ge=1)
    exclude_chunk_ids: list[uuid.UUID] = Field(default_factory=list)


class RetrievedPassage(BaseModel):
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    text: str
    paper_title: str
    year: int | None
    doi: str | None
    arxiv_id: str | None
    section_title: str | None
    section_kind: SectionKind
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def final_score(self) -> float:
        return self.scores.get("rerank", self.scores.get("fused", 0.0))


class RetrievalResult(BaseModel):
    passages: list[RetrievedPassage]
    request: RetrievalRequest
    n_dense: int = 0
    n_lexical: int = 0
    latency_ms: float = 0.0
