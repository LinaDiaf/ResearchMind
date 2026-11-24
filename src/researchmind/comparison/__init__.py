"""Cross-paper comparison tables, every cell backed by passages of that paper."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from researchmind.agent.prompts import SYSTEM_BASE
from researchmind.agent.safety import UNTRUSTED_RULE, looks_like_injection, wrap_passage
from researchmind.db.models import SectionKind
from researchmind.llm import LLMClient, LLMError, Message, StructuredOutputError
from researchmind.retrieval.types import RetrievalFilters, RetrievalRequest, RetrievedPassage

DIMENSIONS: dict[str, tuple[str, list[SectionKind]]] = {
    "Research objective": ("research objective problem addressed contribution",
                           [SectionKind.abstract, SectionKind.introduction]),
    "Method / architecture": ("proposed method model architecture approach",
                              [SectionKind.method, SectionKind.abstract]),
    "Datasets": ("datasets benchmarks corpus used for training and evaluation",
                 [SectionKind.experiments, SectionKind.results]),
    "Experimental setting": ("experimental setup hardware training settings hyperparameters",
                             [SectionKind.experiments, SectionKind.method]),
    "Evaluation metrics": ("evaluation metrics measured", [SectionKind.experiments, SectionKind.results]),
    "Main results": ("main quantitative results compared with baselines",
                     [SectionKind.results, SectionKind.abstract]),
    "Limitations": ("limitations weaknesses future work",
                    [SectionKind.limitations, SectionKind.conclusion, SectionKind.discussion]),
}

PROMPT = """Fill one row of a comparison table for a single paper, using ONLY its passages.

{untrusted_rule}

Paper: {title}

{blocks}

For every dimension listed above, return a cell:
- dimension: copied exactly
- value: a concise answer (max 40 words), numbers copied exactly with dataset and metric
- evidence_ids: the passage ids that state it
- status: "found" or "not_reported" (use not_reported when the passages do not say it;
  then value must be empty and evidence_ids empty)"""


class Cell(BaseModel):
    dimension: str
    value: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["found", "not_reported"] = "not_reported"


class Row(BaseModel):
    cells: list[Cell]


class Comparator:
    def __init__(self, retrieve, llm: LLMClient | None) -> None:
        self.retrieve = retrieve
        self.llm = llm

    def compare(self, project_id: uuid.UUID, papers: list[tuple[uuid.UUID, str]],
                dimensions: list[str], per_dim: int = 2) -> dict:
        table: dict[str, dict[str, dict]] = {}
        evidence: dict[str, dict] = {}
        counter = 0
        for paper_id, title in papers:
            passages: dict[str, list[tuple[str, RetrievedPassage]]] = {}
            for dim in dimensions:
                query, kinds = DIMENSIONS[dim]
                req = RetrievalRequest(project_id=project_id, queries=[query], k=per_dim,
                                       filters=RetrievalFilters(paper_ids=[paper_id], section_kinds=kinds))
                res = self.retrieve(req)
                if not res.passages:
                    req = req.model_copy(update={"filters": RetrievalFilters(paper_ids=[paper_id])})
                    res = self.retrieve(req)
                labelled = []
                for p in res.passages:
                    existing = next((k for k, v in evidence.items() if v["chunk_id"] == str(p.chunk_id)), None)
                    if existing is None:
                        counter += 1
                        existing = f"P{counter}"
                        evidence[existing] = {
                            "id": existing, "chunk_id": str(p.chunk_id), "paper_id": str(p.paper_id),
                            "paper_title": p.paper_title, "year": p.year, "doi": p.doi,
                            "section": p.section_title, "page_start": p.page_start,
                            "page_end": p.page_end, "text": p.text,
                            "injection_flag": looks_like_injection(p.text)}
                    labelled.append((existing, p))
                passages[dim] = labelled
            table[title] = self._row(title, dimensions, passages, evidence)
        return {"dimensions": dimensions, "rows": table, "evidence": evidence}

    def _row(self, title, dimensions, passages, evidence) -> dict[str, dict]:
        allowed = {d: {eid for eid, _ in passages[d]} for d in dimensions}
        row = {d: {"value": "", "citations": [], "status": "not_reported", "source": "none"}
               for d in dimensions}
        if self.llm is not None and any(passages.values()):
            blocks = "\n\n".join(
                f"Dimension: {d}\n" + ("\n".join(wrap_passage(eid, p.text, f'section="{p.section_kind.value}"', 800)
                                                 for eid, p in passages[d]) or "(no passage)")
                for d in dimensions)
            try:
                out = self.llm.structured("compare", [
                    Message("system", SYSTEM_BASE + "\n\n" + UNTRUSTED_RULE),
                    Message("user", PROMPT.format(untrusted_rule=UNTRUSTED_RULE, title=title, blocks=blocks))],
                    Row, max_tokens=1500)
                for c in out.cells:
                    if c.dimension not in row:
                        continue
                    # a cell may only cite passages retrieved for this paper and dimension
                    cites = [e for e in c.evidence_ids if e in allowed[c.dimension]]
                    if c.status == "found" and cites and c.value.strip():
                        row[c.dimension] = {"value": c.value.strip(), "citations": cites,
                                            "status": "found", "source": "llm"}
                return row
            except (StructuredOutputError, LLMError):
                pass
        for d in dimensions:  # extractive fallback: best passage, verbatim
            if passages[d]:
                eid, p = passages[d][0]
                row[d] = {"value": p.text[:280] + ("…" if len(p.text) > 280 else ""),
                          "citations": [eid], "status": "passage", "source": "extractive"}
        return row
