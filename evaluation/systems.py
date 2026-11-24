"""Systems compared in the benchmark.

Baselines answer from a single retrieval with one LLM call; the agentic systems run the full
ResearchMind graph. Every system sees the same corpus, models and per-paper scope.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from researchmind.agent.prompts import SYSTEM_BASE
from researchmind.agent.safety import UNTRUSTED_RULE, wrap_passage
from researchmind.config import AgentBudgetSettings
from researchmind.llm import LLMClient, LLMError, Message, StructuredOutputError, UsageCounter
from researchmind.retrieval.reranker import NoReranker
from researchmind.retrieval.types import RetrievalFilters, RetrievalRequest

BASELINE_PROMPT = """Answer the question about a scientific paper using ONLY the passages.

{untrusted_rule}

Question: {question}

{passages}

Return:
- answer: a concise answer (copy exact phrases from the passages when possible); for yes/no
  questions start with Yes or No
- unanswerable: true if the passages do not contain the answer (then answer = "Unanswerable")
- evidence_ids: ids of the passages that support the answer"""


class BaselineAnswer(BaseModel):
    answer: str
    unanswerable: bool = False
    evidence_ids: list[str] = Field(default_factory=list)


@dataclass
class SystemOutput:
    answer: str
    unanswerable: bool
    ranked_passages: list[str]  # texts, in rank order (retrieval metrics)
    cited_passages: list[str]  # texts of cited evidence (citation metrics)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    retrieval_rounds: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class RetrievalBaseline:
    """1. vector RAG · 2. hybrid RAG · 3. hybrid RAG + reranking."""

    def __init__(self, name: str, engine, provider, *, strategy: str, rerank: bool, k: int = 8):
        self.name, self.engine, self.provider = name, engine, provider
        self.strategy, self.rerank, self.k = strategy, rerank, k

    def run(self, project_id: uuid.UUID, paper_id: uuid.UUID, question: str) -> SystemOutput:
        start = time.perf_counter()
        req = RetrievalRequest(project_id=project_id, queries=[question], strategy=self.strategy,
                               filters=RetrievalFilters(paper_ids=[paper_id]), k=self.k)
        saved = self.engine.reranker
        if not self.rerank:
            self.engine.reranker = NoReranker()
        try:
            result = self.engine.retrieve(req)
        finally:
            self.engine.reranker = saved
        passages = result.passages
        labels = {f"E{i}": p for i, p in enumerate(passages, start=1)}
        client = LLMClient(self.provider, usage=UsageCounter())
        prompt = BASELINE_PROMPT.format(
            untrusted_rule=UNTRUSTED_RULE, question=question,
            passages="\n\n".join(wrap_passage(e, p.text, f'section="{p.section_title or ""}"', 1200)
                                 for e, p in labels.items()) or "(no passage)")
        try:
            out = client.structured("baseline", [Message("system", SYSTEM_BASE + "\n\n" + UNTRUSTED_RULE),
                                                 Message("user", prompt)], BaselineAnswer, max_tokens=500)
            answer, unans = out.answer, out.unanswerable
            cited = [labels[e].text for e in out.evidence_ids if e in labels]
        except (StructuredOutputError, LLMError) as exc:
            answer, unans, cited = "", True, []
            err = str(exc)
        else:
            err = None
        u = client.usage
        return SystemOutput(answer=answer, unanswerable=unans, ranked_passages=[p.text for p in passages],
                            cited_passages=cited, llm_calls=u.calls, prompt_tokens=u.prompt_tokens,
                            completion_tokens=u.completion_tokens,
                            latency_s=time.perf_counter() - start, extra={"error": err})


class AgenticSystem:
    """4. agentic hybrid RAG (+ ablations through budget settings)."""

    def __init__(self, name: str, runner_factory, budget: AgentBudgetSettings):
        self.name = name
        self.runner = runner_factory(budget)
        self.budget = budget

    def run(self, project_id: uuid.UUID, paper_id: uuid.UUID, question: str) -> SystemOutput:
        start = time.perf_counter()
        inv = self.runner.create(project_id, question, [paper_id])
        report = self.runner.run(inv)
        state = self.runner.state(inv)
        evidence = state.get("evidence", {})
        # ranked list: evidence in the order it was first retrieved (round, then rank)
        ranked = [evidence[e]["text"] for a in state.get("retrieval_log", [])
                  for e in a["evidence_ids"] if e in evidence]
        ranked = list(dict.fromkeys(ranked))
        claims = [c for s in report.get("sections", []) for c in s["claims"]]
        answer_claims = next((s["claims"] for s in report.get("sections", [])
                              if s["heading"].lower().startswith("answer")), claims[:2])
        cited_ids = list(dict.fromkeys(e for c in claims for e in c["citations"]))
        has_grounded = any(c["citations"] for c in claims)
        stats = report.get("stats", {})
        usage = _usage(self.runner, inv)
        return SystemOutput(
            answer=" ".join(c["text"] for c in answer_claims) if has_grounded else "Unanswerable",
            unanswerable=not has_grounded, ranked_passages=ranked,
            cited_passages=[evidence[e]["text"] for e in cited_ids if e in evidence],
            llm_calls=usage["calls"], prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"], latency_s=time.perf_counter() - start,
            retrieval_rounds=stats.get("retrieval_rounds", 0),
            extra={"investigation_id": str(inv), "stop_reason": stats.get("stop_reason"),
                   "subquestions": stats.get("subquestions"),
                   "claims": len(claims), "removed_claims": len(report.get("removed_claims", [])),
                   "supported": sum(c["support"] == "supported" for c in claims),
                   "unsupported_removed": sum("not supported" in r["reason"] for r in report.get("removed_claims", []))})


def _usage(runner, inv) -> dict[str, Any]:
    from researchmind.workspace.investigations import Workspace

    return Workspace(runner.sf).llm_usage(inv)
