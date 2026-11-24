"""Schemas of the decisions the LLM is allowed to make.

They are deliberately small: small local models are far more reliable on short, flat JSON.
Every field is validated; anything the model invents outside these fields is ignored.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from researchmind.db.models import SectionKind


class Intent(StrEnum):
    lookup = "lookup"
    single_paper = "single_paper"
    multi_paper_synthesis = "multi_paper_synthesis"
    method_comparison = "method_comparison"
    evidence_verification = "evidence_verification"
    contradiction_analysis = "contradiction_analysis"
    gap_exploration = "gap_exploration"


def _short_list(v: list[str], n: int) -> list[str]:
    out, seen = [], set()
    for x in v:
        x = " ".join(str(x).split())[:200]
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out[:n]


class QueryAnalysis(BaseModel):
    intent: Intent
    complexity: Literal["simple", "complex"] = "complex"
    key_terms: list[str] = Field(default_factory=list, description="methods, datasets, metrics, named entities")
    year_min: int | None = Field(None, ge=1900, le=2100)
    year_max: int | None = Field(None, ge=1900, le=2100)
    rationale: str = ""

    @field_validator("key_terms")
    @classmethod
    def _terms(cls, v: list[str]) -> list[str]:
        return _short_list(v, 8)


class SubQuestionDraft(BaseModel):
    question: str = Field(min_length=5, max_length=400)
    rationale: str = ""
    required_aspects: list[str] = Field(
        default_factory=list, description="1-4 short noun phrases the answer must cover"
    )
    target_sections: list[SectionKind] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list, description="1-3 search queries")
    priority: int = Field(2, ge=1, le=3, description="1 = most important")

    @field_validator("required_aspects")
    @classmethod
    def _aspects(cls, v: list[str]) -> list[str]:
        return _short_list(v, 4)

    @field_validator("search_queries")
    @classmethod
    def _queries(cls, v: list[str]) -> list[str]:
        return _short_list(v, 3)

    @field_validator("target_sections", mode="before")
    @classmethod
    def _sections(cls, v):
        valid = {k.value for k in SectionKind}
        return [x for x in (v or []) if isinstance(x, str) and x in valid][:4]


class ResearchPlan(BaseModel):
    subquestions: list[SubQuestionDraft] = Field(min_length=1)


class RoutingDecision(BaseModel):
    queries: list[str] = Field(min_length=1, description="1-3 new search queries")
    section_kinds: list[SectionKind] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("queries")
    @classmethod
    def _queries(cls, v: list[str]) -> list[str]:
        return _short_list(v, 3)

    @field_validator("section_kinds", mode="before")
    @classmethod
    def _sections(cls, v):
        valid = {k.value for k in SectionKind}
        return [x for x in (v or []) if isinstance(x, str) and x in valid][:4]


class PassageGrade(BaseModel):
    evidence_id: str
    relevance: Literal["high", "partial", "irrelevant"]
    stance: Literal["supports", "contradicts", "neutral"] = "neutral"
    aspects: list[str] = Field(default_factory=list, description="required aspects it covers")
    note: str = ""


class EvidenceAssessment(BaseModel):
    grades: list[PassageGrade]
    missing_aspects: list[str] = Field(default_factory=list)
    conflict: bool = False
    conflict_note: str = ""


class Contradiction(BaseModel):
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    issues: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    followups: list[SubQuestionDraft] = Field(default_factory=list)


ClaimTypeLit = Literal["reported_finding", "author_interpretation", "ai_synthesis", "hypothesis"]


class ClaimDraft(BaseModel):
    text: str = Field(min_length=3)
    claim_type: ClaimTypeLit
    evidence_ids: list[str] = Field(default_factory=list)


class ReportSectionDraft(BaseModel):
    heading: str
    claims: list[ClaimDraft]


class ReportDraft(BaseModel):
    title: str
    sections: list[ReportSectionDraft] = Field(min_length=1)


class ClaimVerdict(BaseModel):
    claim_id: str
    support: Literal["supported", "partially_supported", "unsupported"]
    note: str = ""


class VerificationResult(BaseModel):
    verdicts: list[ClaimVerdict]
