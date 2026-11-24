"""Scripted LLM for agent tests: answers each prompt type with a configurable function."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Callable

from researchmind.db.models import SectionKind
from researchmind.llm.fake import FakeLLM
from researchmind.retrieval.types import RetrievalResult, RetrievedPassage

MARKERS = {
    "analyze": "Analyse this research question",
    "plan": "Create a research plan",
    "route": "refining a literature search",
    "grade": "Grade retrieved passages",
    "critique": "Review the evidence gathered",
    "synthesize": "Write an evidence-grounded research report",
    "verify": "Check whether each claim",
}


def prompt_kind(text: str) -> str:
    for k, m in MARKERS.items():
        if m in text:
            return k
    return "unknown"


def passage_ids(text: str) -> list[str]:
    return re.findall(r'<passage id="(E\d+)"', text)


def claim_ids(text: str) -> list[str]:
    return re.findall(r"^Claim (C\d+):", text, flags=re.MULTILINE)


class ScriptedLLM(FakeLLM):
    def __init__(self, handlers: dict[str, Callable[[str], object]]) -> None:
        self.counts: Counter[str] = Counter()

        def handler(messages, schema):
            user = messages[-1].content if messages[-1].role == "user" else messages[0].content
            # repair turns: answer the original prompt kind
            original = next((m.content for m in messages if m.role == "user"), user)
            kind = prompt_kind(original)
            self.counts[kind] += 1
            fn = handlers.get(kind)
            if fn is None:
                raise AssertionError(f"unexpected prompt kind {kind}")
            return fn(original)

        super().__init__(handler)


def make_passage(text: str, paper: str = "Paper A", page: int = 1,
                 kind: SectionKind = SectionKind.results, rerank: float = 0.9,
                 paper_id: uuid.UUID | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=uuid.uuid4(), paper_id=paper_id or uuid.uuid5(uuid.NAMESPACE_DNS, paper),
        text=text, paper_title=paper, year=2024, doi=None, arxiv_id=None, section_title="Results",
        section_kind=kind, page_start=page, page_end=page, char_start=0, char_end=len(text),
        scores={"fused": 0.03, "rerank": rerank})


class StubRetriever:
    """Returns pre-defined passages per call, honouring exclude_chunk_ids."""

    def __init__(self, batches: list[list[RetrievedPassage]], fail_on_call: int | None = None):
        self.batches = batches
        self.requests = []
        self.fail_on_call = fail_on_call

    def __call__(self, req):
        self.requests.append(req)
        if self.fail_on_call is not None and len(self.requests) == self.fail_on_call:
            raise RuntimeError("simulated crash during retrieval")
        idx = min(len(self.requests) - 1, len(self.batches) - 1)
        excluded = set(req.exclude_chunk_ids)
        passages = [p for p in self.batches[idx] if p.chunk_id not in excluded]
        return RetrievalResult(passages=passages, request=req, n_dense=len(passages),
                               n_lexical=len(passages))
