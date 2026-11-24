"""Investigation state.

Every value is plain JSON (dicts, lists, strings, numbers) so that the LangGraph Postgres
checkpointer can store it and an interrupted investigation can be resumed in a new process.
Pydantic models validate these dicts inside the nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

SubStatus = Literal["open", "answered", "insufficient"]


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str = ""
    required_aspects: list[str] = Field(default_factory=list)
    target_sections: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    priority: int = 2
    origin: Literal["plan", "critique", "fallback"] = "plan"
    status: SubStatus = "open"
    rounds: int = 0
    stale_rounds: int = 0
    conflict_followup_done: bool = False
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)  # accepted evidence (high/partial)
    seen_chunk_ids: list[str] = Field(default_factory=list)
    outcome_note: str = ""


class Evidence(BaseModel):
    id: str  # "E7"
    chunk_id: str
    paper_id: str
    paper_title: str
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    section_title: str | None = None
    section_kind: str
    page_start: int
    page_end: int
    char_start: int = 0
    char_end: int = 0
    text: str
    scores: dict[str, float] = Field(default_factory=dict)
    subquestion_ids: list[str] = Field(default_factory=list)
    round: int = 1
    strategy: str = "hybrid"
    relevance: str | None = None
    stance: str | None = None
    aspects: list[str] = Field(default_factory=list)
    grade_note: str = ""
    injection_flag: bool = False


class ResearchState(TypedDict, total=False):
    investigation_id: str
    project_id: str
    question: str
    config: dict[str, Any]
    analysis: dict[str, Any] | None
    subquestions: list[dict[str, Any]]
    current: str | None
    pending_request: dict[str, Any] | None
    last_new_evidence: list[str]
    last_assessment: dict[str, Any] | None
    evidence: dict[str, dict[str, Any]]
    critique: dict[str, Any] | None
    critique_done: bool
    report: dict[str, Any] | None
    stop_reason: str | None
    llm_calls: int
    elapsed_s: float
    retrieval_log: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def subquestions(state: ResearchState) -> list[SubQuestion]:
    return [SubQuestion.model_validate(s) for s in state.get("subquestions", [])]


def dump_subquestions(subs: list[SubQuestion]) -> list[dict[str, Any]]:
    return [s.model_dump(mode="json") for s in subs]


def evidence_map(state: ResearchState) -> dict[str, Evidence]:
    return {k: Evidence.model_validate(v) for k, v in state.get("evidence", {}).items()}
