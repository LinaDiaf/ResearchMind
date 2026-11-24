"""The ResearchMind agent: a bounded LangGraph workflow.

analyze → plan → select_next ─┬─► route → retrieve → evaluate → decide ─► select_next
                              ├─► critique ─► select_next
                              └─► synthesize → verify → END

The LLM only produces typed decisions. Retrieval, routing rules, stopping rules, budgets and
citation checks are deterministic code.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from langgraph.graph import END, START, StateGraph

from researchmind.agent import prompts as P
from researchmind.agent.policy import (
    INTENT_ASPECTS,
    INTENT_SECTIONS,
    decide,
    heuristic_intent,
    heuristic_terms,
    match_aspects,
    plan_route,
)
from researchmind.agent.safety import UNTRUSTED_RULE, looks_like_injection, wrap_passage
from researchmind.agent.schemas import (
    Critique,
    EvidenceAssessment,
    Intent,
    QueryAnalysis,
    ReportDraft,
    ResearchPlan,
    RoutingDecision,
    SubQuestionDraft,
    VerificationResult,
)
from researchmind.agent.state import (
    Evidence,
    ResearchState,
    SubQuestion,
    dump_subquestions,
    evidence_map,
    subquestions,
)
from researchmind.config import AgentBudgetSettings
from researchmind.db.models import SectionKind
from researchmind.llm import BudgetExceeded, LLMClient, LLMError, Message, StructuredOutputError
from researchmind.log import get_logger
from researchmind.retrieval.types import RetrievalFilters, RetrievalRequest
from researchmind.synthesis.report import (
    apply_verdicts,
    evidence_reference,
    validate_draft,
)

log = get_logger(__name__)

SYNTHESIS_RESERVE = 2  # LLM calls kept for synthesize + verify
RELEVANCE_RANK = {"irrelevant": 0, "partial": 1, "high": 2}


@dataclass
class AgentDeps:
    llm: LLMClient
    retrieve: Callable[[RetrievalRequest], Any]  # RetrievalEngine.retrieve
    budget: AgentBudgetSettings
    library_titles: Callable[[str], list[str]] = lambda project_id: []
    keep_unsupported: bool = False


def _event(node: str, kind: str, message: str, **data: Any) -> dict[str, Any]:
    return {"node": node, "type": kind, "message": message, "data": data, "t": time.time()}


def _sys(*extra: str) -> Message:
    return Message("system", "\n\n".join([P.SYSTEM_BASE, *extra]))


def node(name: str):
    """Common node wrapper: timing, budget accounting, LLM-budget exhaustion handling."""

    def deco(fn):
        @wraps(fn)
        def wrapped(self: ResearchAgent, state: ResearchState) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                update = fn(self, state) or {}
            except BudgetExceeded as exc:
                update = {"stop_reason": "llm_budget",
                          "events": [_event(name, "budget", str(exc))]}
            update["elapsed_s"] = float(state.get("elapsed_s", 0.0)) + time.perf_counter() - start
            update["llm_calls"] = self.deps.llm.usage.calls
            return update

        wrapped.node_name = name  # type: ignore[attr-defined]
        return wrapped

    return deco


class ResearchAgent:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.b = deps.budget

    # ------------------------------------------------------------------ helpers
    def _llm(self, node_name: str, messages: list[Message], schema, max_tokens: int = 1500):
        return self.deps.llm.structured(node_name, messages, schema, max_tokens=max_tokens)

    def _remaining_calls(self) -> int:
        r = self.deps.llm.usage.remaining
        return 10**6 if r is None else r

    def _out_of_time(self, state: ResearchState) -> bool:
        return float(state.get("elapsed_s", 0.0)) >= self.b.max_seconds

    @staticmethod
    def _intent(state: ResearchState) -> Intent:
        return Intent((state.get("analysis") or {}).get("intent", Intent.lookup.value))

    # -------------------------------------------------------------------- nodes
    @node("analyze")
    def analyze(self, state: ResearchState) -> dict[str, Any]:
        q = state["question"]
        try:
            a = self._llm("analyze", [_sys(), Message("user", P.ANALYZE.format(question=q))],
                          QueryAnalysis, 400)
            source = "llm"
        except (StructuredOutputError, LLMError) as exc:
            a = QueryAnalysis(intent=heuristic_intent(q), key_terms=heuristic_terms(q),
                              rationale="heuristic fallback")
            source = "fallback"
            log.warning("analyze_fallback", error=str(exc))
        if not a.key_terms:
            a.key_terms = heuristic_terms(q)
        data = a.model_dump(mode="json")
        data["source"] = source
        return {"analysis": data, "events": [_event(
            "analyze", "analysis", f"Intent: {a.intent.value} ({a.complexity})",
            key_terms=a.key_terms, source=source, rationale=a.rationale)]}

    @node("plan")
    def plan(self, state: ResearchState) -> dict[str, Any]:
        q = state["question"]
        analysis = QueryAnalysis.model_validate(
            {k: v for k, v in state["analysis"].items() if k != "source"})
        intent = analysis.intent
        drafts: list[SubQuestionDraft]
        origin = "plan"
        if analysis.complexity == "simple" or intent in (Intent.lookup, Intent.single_paper):
            drafts = [SubQuestionDraft(question=q, required_aspects=INTENT_ASPECTS[intent],
                                       search_queries=[" ".join(analysis.key_terms)] if analysis.key_terms else [],
                                       priority=1)]
            source = "rule (simple question)"
        else:
            titles = self.deps.library_titles(state["project_id"])[:25]
            prompt = P.PLAN.format(question=q, intent=intent.value,
                                   key_terms=", ".join(analysis.key_terms) or "-",
                                   library="; ".join(titles) or "-",
                                   max_sub=self.b.max_subquestions)
            try:
                drafts = self._llm("plan", [_sys(), Message("user", prompt)], ResearchPlan,
                                   1500).subquestions
                source = "llm"
            except (StructuredOutputError, LLMError) as exc:
                log.warning("plan_fallback", error=str(exc))
                drafts = [SubQuestionDraft(question=q, required_aspects=INTENT_ASPECTS[intent],
                                           priority=1)]
                source, origin = "fallback", "fallback"
        drafts = sorted(drafts, key=lambda d: d.priority)[: self.b.max_subquestions]
        subs = []
        for i, d in enumerate(drafts, start=1):
            subs.append(SubQuestion(
                id=f"S{i}", question=d.question, rationale=d.rationale,
                required_aspects=d.required_aspects or INTENT_ASPECTS[intent],
                target_sections=[s.value for s in d.target_sections] or INTENT_SECTIONS[intent],
                search_queries=d.search_queries, priority=d.priority, origin=origin))
        return {"subquestions": dump_subquestions(subs), "events": [_event(
            "plan", "plan", f"{len(subs)} subquestion(s) planned", source=source,
            subquestions=[{"id": s.id, "question": s.question, "aspects": s.required_aspects,
                           "sections": s.target_sections} for s in subs])]}

    @node("select_next")
    def select_next(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        events = []
        stop = state.get("stop_reason")
        if not stop and self._out_of_time(state):
            stop = "time_budget"
        # a full round (route + evaluate) must fit before the synthesis reserve
        if not stop and self._remaining_calls() < SYNTHESIS_RESERVE + 2:
            stop = "llm_budget"
        if stop:
            for s in subs:
                if s.status == "open":
                    s.status = "insufficient"
                    s.missing_aspects = [a for a in s.required_aspects if a not in s.covered_aspects]
                    s.outcome_note = f"stopped: {stop.replace('_', ' ')} exhausted"
            events.append(_event("select_next", "budget", f"Budget reached ({stop}); writing report"))
            return {"subquestions": dump_subquestions(subs), "current": None, "stop_reason": stop,
                    "events": events}
        current = state.get("current")
        open_subs = [s for s in subs if s.status == "open"]
        chosen = next((s for s in open_subs if s.id == current), None)
        if chosen is None and open_subs:
            chosen = sorted(open_subs, key=lambda s: (s.priority, s.id))[0]
        if chosen:
            events.append(_event("select_next", "select",
                                 f"{chosen.id}: {chosen.question}", round=chosen.rounds + 1))
        return {"current": chosen.id if chosen else None, "events": events}

    def after_select(self, state: ResearchState) -> str:
        if state.get("current"):
            return "route"
        if (self.b.enable_critique and not state.get("critique_done") and not state.get("stop_reason")
                and self._remaining_calls() > SYNTHESIS_RESERVE + 1 and state.get("evidence")):
            return "critique"
        return "synthesize"

    @node("route")
    def route(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        sub = next(s for s in subs if s.id == state["current"])
        analysis = state.get("analysis") or {}
        round_no = sub.rounds + 1
        conflict = bool((state.get("last_assessment") or {}).get("conflict")) and \
            (state.get("last_assessment") or {}).get("subquestion") == sub.id
        tried = [q for a in state.get("retrieval_log", []) if a["subquestion"] == sub.id
                 for q in a["queries"]]
        llm_sections: list[str] = []
        source = "rules"
        if round_no == 1:
            queries = [sub.question, *sub.search_queries][:3]
        else:
            missing = [a for a in sub.required_aspects if a not in sub.covered_aspects]
            ev = evidence_map(state)
            ctx = "\n".join(f"- {ev[e].text[:300]}" for e in sub.evidence_ids[:3] if e in ev) or "(none)"
            situation = ("two papers report conflicting findings; find the experimental "
                         "conditions (datasets, settings) behind each side" if conflict
                         else "previous searches did not cover the missing aspects")
            prompt = P.ROUTE.format(question=sub.question, missing=", ".join(missing) or "-",
                                    situation=situation, tried="; ".join(tried) or "-", context=ctx)
            try:
                d = self._llm("route", [_sys(), Message("user", prompt)], RoutingDecision, 400)
                queries = [q for q in d.queries if q.lower() not in {t.lower() for t in tried}] or d.queries
                llm_sections = [s.value for s in d.section_kinds]
                source = "llm"
            except (StructuredOutputError, LLMError):
                terms = " ".join(analysis.get("key_terms", [])[:4])
                queries = [f"{sub.question} {' '.join(missing)}".strip(), f"{terms} {' '.join(missing)}".strip()]
                source = "fallback"
        rp = plan_route(sub, queries, round_no, self.b.top_k, conflict)
        sections = llm_sections if (llm_sections and rp.mode != "broaden") else rp.section_kinds
        scope = (state.get("config") or {}).get("paper_ids") or None
        filters = RetrievalFilters(
            paper_ids=[uuid.UUID(p) for p in scope] if scope else None,
            year_min=analysis.get("year_min"), year_max=analysis.get("year_max"),
            section_kinds=[SectionKind(s) for s in sections] or None)
        req = RetrievalRequest(
            project_id=uuid.UUID(state["project_id"]), queries=queries, strategy=rp.strategy,
            dense_weight=rp.dense_weight, lexical_weight=rp.lexical_weight, filters=filters,
            k=rp.k, rerank_candidates=self.b.rerank_candidates,
            exclude_chunk_ids=[uuid.UUID(c) for c in sub.seen_chunk_ids])
        return {"pending_request": req.model_dump(mode="json"), "events": [_event(
            "route", "route", f"Round {round_no} ({rp.mode}): {rp.strategy} "
            f"dense={rp.dense_weight:g} bm25={rp.lexical_weight:g}"
            + (f", sections={','.join(sections)}" if sections else ""),
            queries=queries, source=source, subquestion=sub.id)]}

    @node("retrieve")
    def retrieve(self, state: ResearchState) -> dict[str, Any]:
        req = RetrievalRequest.model_validate(state["pending_request"])
        sub_id = state["current"]
        result = self.deps.retrieve(req)
        # Section filters can be too narrow: fall back once to all sections.
        widened = False
        if not result.passages and req.filters.section_kinds:
            req = req.model_copy(update={"filters": req.filters.model_copy(update={"section_kinds": None})})
            result = self.deps.retrieve(req)
            widened = True
        evidence = dict(state.get("evidence", {}))
        by_chunk = {v["chunk_id"]: k for k, v in evidence.items()}
        new_ids = []
        round_no = next(s for s in subquestions(state) if s.id == sub_id).rounds + 1
        for p in result.passages:
            cid = str(p.chunk_id)
            if cid in by_chunk:
                eid = by_chunk[cid]
                ev = Evidence.model_validate(evidence[eid])
                if sub_id not in ev.subquestion_ids:
                    ev.subquestion_ids.append(sub_id)
            else:
                eid = f"E{len(evidence) + 1}"
                ev = Evidence(
                    id=eid, chunk_id=cid, paper_id=str(p.paper_id), paper_title=p.paper_title,
                    year=p.year, doi=p.doi, arxiv_id=p.arxiv_id, section_title=p.section_title,
                    section_kind=p.section_kind.value, page_start=p.page_start, page_end=p.page_end,
                    char_start=p.char_start, char_end=p.char_end, text=p.text, scores=p.scores,
                    subquestion_ids=[sub_id], round=round_no, strategy=req.strategy,
                    injection_flag=looks_like_injection(p.text))
                by_chunk[cid] = eid
            evidence[eid] = ev.model_dump(mode="json")
            new_ids.append(eid)
        action = {"subquestion": sub_id, "round": round_no, "queries": req.queries,
                  "strategy": req.strategy, "dense_weight": req.dense_weight,
                  "lexical_weight": req.lexical_weight,
                  "section_kinds": [k.value for k in req.filters.section_kinds or []],
                  "k": req.k, "n_dense": result.n_dense, "n_lexical": result.n_lexical,
                  "returned": len(result.passages), "evidence_ids": new_ids,
                  "latency_ms": round(result.latency_ms, 1), "widened": widened}
        flagged = [e for e in new_ids if evidence[e]["injection_flag"]]
        msg = f"{len(new_ids)} passage(s) from {len({evidence[e]['paper_id'] for e in new_ids})} paper(s)"
        if widened:
            msg += " (section filter widened)"
        events = [_event("retrieve", "retrieve", msg, evidence_ids=new_ids)]
        if flagged:
            events.append(_event("retrieve", "safety",
                                 f"Instruction-like text detected in {', '.join(flagged)}; treated as data"))
        return {"evidence": evidence, "last_new_evidence": new_ids, "retrieval_log": [action],
                "events": events}

    @node("evaluate")
    def evaluate(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        sub = next(s for s in subs if s.id == state["current"])
        evidence = evidence_map(state)
        new_ids = [e for e in state.get("last_new_evidence", []) if e in evidence]
        sub.rounds += 1
        sub.seen_chunk_ids = list(dict.fromkeys(sub.seen_chunk_ids + [evidence[e].chunk_id for e in new_ids]))
        assessment: dict[str, Any] = {"subquestion": sub.id, "conflict": False, "source": "none"}
        accepted_new: list[str] = []
        has_high = any(evidence[e].relevance == "high" for e in sub.evidence_ids if e in evidence)
        if new_ids:
            passages = "\n\n".join(
                wrap_passage(e, evidence[e].text,
                             f'paper="{evidence[e].paper_title[:80]}" section="{evidence[e].section_kind}"'
                             + (' warning="contains instruction-like text"' if evidence[e].injection_flag else ""))
                for e in new_ids[:10])
            prompt = P.GRADE.format(untrusted_rule=UNTRUSTED_RULE, question=sub.question,
                                    aspects="; ".join(sub.required_aspects),
                                    covered="; ".join(sub.covered_aspects) or "none",
                                    passages=passages)
            try:
                a = self._llm("evaluate", [_sys(UNTRUSTED_RULE), Message("user", prompt)],
                              EvidenceAssessment, 1500)
                grades = {g.evidence_id: g for g in a.grades if g.evidence_id in new_ids}
                for eid in new_ids:
                    g = grades.get(eid)
                    ev = evidence[eid]
                    if g is None:
                        continue  # ungraded passages are not accepted
                    if RELEVANCE_RANK[g.relevance] >= RELEVANCE_RANK.get(ev.relevance or "irrelevant", 0):
                        ev.relevance, ev.stance, ev.grade_note = g.relevance, g.stance, g.note[:300]
                    ev.aspects = list(dict.fromkeys(ev.aspects + g.aspects))
                    if g.relevance in ("high", "partial"):
                        accepted_new.append(eid)
                        sub.covered_aspects = list(dict.fromkeys(
                            sub.covered_aspects + match_aspects(g.aspects, sub.required_aspects)))
                        has_high = has_high or g.relevance == "high"
                assessment.update(conflict=a.conflict, conflict_note=a.conflict_note[:300],
                                  missing=a.missing_aspects, source="llm")
            except (StructuredOutputError, LLMError):
                # Deterministic fallback: accept strongly reranked passages as partial evidence.
                for eid in new_ids:
                    ev = evidence[eid]
                    score = ev.scores.get("rerank")
                    if score is not None and score >= 0.5:
                        ev.relevance = ev.relevance or "partial"
                        ev.grade_note = "accepted by reranker score (LLM grading unavailable)"
                        accepted_new.append(eid)
                assessment["source"] = "fallback"
        sub.evidence_ids = list(dict.fromkeys(sub.evidence_ids + accepted_new))
        sub.stale_rounds = 0 if accepted_new else sub.stale_rounds + 1
        sub.missing_aspects = [a for a in sub.required_aspects if a not in sub.covered_aspects]
        assessment.update(new_accepted=len(accepted_new), has_high=has_high,
                          covered=sub.covered_aspects, missing_after=sub.missing_aspects)
        msg = (f"{len(accepted_new)}/{len(new_ids)} new passage(s) accepted; "
               f"covered: {', '.join(sub.covered_aspects) or 'none'}; "
               f"missing: {', '.join(sub.missing_aspects) or 'none'}")
        if assessment.get("conflict"):
            msg += " — conflict detected"
        return {"subquestions": dump_subquestions(subs),
                "evidence": {k: v.model_dump(mode="json") for k, v in evidence.items()},
                "last_assessment": assessment,
                "events": [_event("evaluate", "evaluate", msg, subquestion=sub.id,
                                  accepted=accepted_new, source=assessment["source"],
                                  conflict_note=assessment.get("conflict_note", ""))]}

    @node("decide")
    def decide(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        sub = next(s for s in subs if s.id == state["current"])
        a = state.get("last_assessment") or {}
        d = decide(sub, new_accepted=a.get("new_accepted", 0), has_high=a.get("has_high", False),
                   conflict=bool(a.get("conflict")), max_rounds=self.b.max_rounds_per_subquestion)
        current: str | None = sub.id
        if d.action == "answered":
            sub.status, sub.outcome_note, current = "answered", d.reason, None
        elif d.action == "insufficient":
            sub.status, sub.outcome_note, current = "insufficient", d.reason, None
        elif d.conflict:
            sub.conflict_followup_done = True
        return {"subquestions": dump_subquestions(subs), "current": current, "events": [_event(
            "decide", "decide", f"{sub.id}: {d.action} — {d.reason}", subquestion=sub.id,
            action=d.action)]}

    @node("critique")
    def critique(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        ev = evidence_map(state)
        accepted = [e for s in subs for e in s.evidence_ids if e in ev]
        sub_txt = "\n".join(f"- {s.id} [{s.status}] {s.question} (missing: {', '.join(s.missing_aspects) or 'none'})"
                            for s in subs)
        ev_txt = "\n".join(f"- {e} | {ev[e].paper_title[:70]} | {ev[e].grade_note or ev[e].text[:200]}"
                           for e in dict.fromkeys(accepted))
        prompt = P.CRITIQUE.format(untrusted_rule=UNTRUSTED_RULE, question=state["question"],
                                   subquestions=sub_txt, evidence=ev_txt or "(none)",
                                   max_followups=self.b.max_followups)
        try:
            c = self._llm("critique", [_sys(UNTRUSTED_RULE), Message("user", prompt)], Critique, 1200)
        except (StructuredOutputError, LLMError):
            return {"critique_done": True, "critique": {"issues": [], "contradictions": [], "source": "failed"},
                    "events": [_event("critique", "critique", "Critique unavailable (model output invalid)")]}
        existing = {s.question.lower() for s in subs}
        added = []
        room = max(0, self.b.max_subquestions + self.b.max_followups - len(subs))
        for f in c.followups[: min(self.b.max_followups, room)]:
            if f.question.lower() in existing:
                continue
            added.append(SubQuestion(
                id=f"S{len(subs) + len(added) + 1}", question=f.question, rationale=f.rationale,
                required_aspects=f.required_aspects or ["direct answer"],
                target_sections=[s.value for s in f.target_sections],
                search_queries=f.search_queries, priority=f.priority, origin="critique"))
        crit = c.model_dump(mode="json", exclude={"followups"})
        crit["contradictions"] = [x for x in crit["contradictions"]
                                  if any(e in ev for e in x["evidence_ids"])]
        msg = f"{len(c.issues)} issue(s), {len(crit['contradictions'])} contradiction(s), {len(added)} follow-up(s)"
        return {"critique": crit, "critique_done": True,
                "subquestions": dump_subquestions(subs + added),
                "events": [_event("critique", "critique", msg, issues=c.issues,
                                  followups=[s.question for s in added])]}

    @node("synthesize")
    def synthesize(self, state: ResearchState) -> dict[str, Any]:
        subs = subquestions(state)
        ev = evidence_map(state)
        accepted_ids = list(dict.fromkeys(e for s in subs for e in s.evidence_ids if e in ev))
        insufficient = []
        tried_by_sub: dict[str, list[str]] = {}
        for a in state.get("retrieval_log", []):
            tried_by_sub.setdefault(a["subquestion"], []).extend(a["queries"])
        for s in subs:
            if s.status != "answered":
                insufficient.append({
                    "id": s.id, "question": s.question, "missing_aspects": s.missing_aspects,
                    "note": s.outcome_note,
                    "suggested_queries": list(dict.fromkeys(tried_by_sub.get(s.id, [])))[:3]})
        base = {"question": state["question"], "intent": self._intent(state).value,
                "insufficient": insufficient, "removed_claims": [],
                "critique": state.get("critique"), "stop_reason": state.get("stop_reason")}
        if not accepted_ids:
            report = {**base, "title": "Insufficient evidence",
                      "notice": "No passage in the project's library was judged relevant to this "
                                "question. No answer is given. Import more papers and try again.",
                      "sections": [], "references": []}
            return {"report": report, "events": [_event(
                "synthesize", "synthesize", "No relevant evidence: no answer generated")]}
        sub_txt = "\n".join(f"- {s.id} [{s.status}] {s.question} — evidence: {', '.join(s.evidence_ids) or 'none'}"
                            + (f"; missing: {', '.join(s.missing_aspects)}" if s.missing_aspects else "")
                            for s in subs)
        ev_txt = "\n\n".join(wrap_passage(
            e, ev[e].text,
            f'paper="{ev[e].paper_title[:90]}" year="{ev[e].year or "?"}" section="{ev[e].section_kind}" relevance="{ev[e].relevance}"',
            900) for e in accepted_ids[:24])
        contradictions = "; ".join(c["description"] for c in (state.get("critique") or {}).get("contradictions", [])) or "none reported"
        prompt = P.SYNTHESIZE.format(untrusted_rule=UNTRUSTED_RULE, question=state["question"],
                                     intent=base["intent"], subquestions=sub_txt, evidence=ev_txt,
                                     contradictions=contradictions)
        try:
            draft = self._llm("synthesize", [_sys(UNTRUSTED_RULE), Message("user", prompt)],
                              ReportDraft, 2500)
        except (StructuredOutputError, LLMError, BudgetExceeded) as exc:
            log.warning("synthesis_fallback", error=str(exc))
            report = {**base, "title": "Evidence summary (synthesis unavailable)",
                      "notice": "The model could not produce a valid report. The most relevant "
                                "passages are listed verbatim instead.",
                      "sections": [{"heading": s.question, "claims": [
                          {"id": f"C{i}", "text": ev[e].text[:400], "claim_type": "reported_finding",
                           "citations": [e], "invalid_citations": [], "support": "supported",
                           "note": "verbatim passage"}
                          for i, e in enumerate(s.evidence_ids[:3], start=1) if e in ev]}
                          for s in subs if s.evidence_ids],
                      "references": [evidence_reference(ev[e]) for e in accepted_ids]}
            return {"report": report, "events": [_event("synthesize", "synthesize",
                                                         "Synthesis failed; extractive summary produced")]}
        sections, removed = validate_draft(draft, set(accepted_ids))
        n_claims = sum(len(s["claims"]) for s in sections)
        report = {**base, "title": draft.title, "sections": sections, "removed_claims": removed}
        return {"report": report, "events": [_event(
            "synthesize", "synthesize", f"Draft: {n_claims} claim(s); {len(removed)} removed by citation checks",
            removed=removed)]}

    @node("verify")
    def verify(self, state: ResearchState) -> dict[str, Any]:
        report = dict(state.get("report") or {})
        ev = evidence_map(state)
        sections = report.get("sections", [])
        to_check = [c for s in sections for c in s["claims"] if c["citations"] and c["support"] == "not_checked"]
        verdicts: dict[str, tuple[str, str]] = {}
        source = "none"
        if to_check and report.get("references") is None:
            items = "\n\n".join(
                f'Claim {c["id"]}: {c["text"]}\n' + "\n".join(
                    wrap_passage(e, ev[e].text, f'paper="{ev[e].paper_title[:80]}"', 700)
                    for e in c["citations"] if e in ev)
                for c in to_check[:25])
            try:
                v = self._llm("verify", [_sys(UNTRUSTED_RULE), Message("user", P.VERIFY.format(
                    untrusted_rule=UNTRUSTED_RULE, items=items))], VerificationResult, 1500)
                verdicts = {x.claim_id: (x.support, x.note[:300]) for x in v.verdicts}
                source = "llm"
            except (StructuredOutputError, LLMError, BudgetExceeded):
                source = "unavailable"
        removed = apply_verdicts(sections, verdicts, keep_unsupported=self.deps.keep_unsupported)
        report["removed_claims"] = report.get("removed_claims", []) + removed
        if report.get("references") is None:
            cited = list(dict.fromkeys(e for s in sections for c in s["claims"] for e in c["citations"]))
            report["references"] = [evidence_reference(ev[e]) for e in cited if e in ev]
        report["verification"] = source
        counts: dict[str, int] = {}
        for s in sections:
            for c in s["claims"]:
                counts[c["support"]] = counts.get(c["support"], 0) + 1
        return {"report": report, "events": [_event(
            "verify", "verify", "Claims: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none")
            + f"; {len(removed)} removed as unsupported", source=source)]}


def _unless_stopped(next_node: str):
    def router(state: ResearchState) -> str:
        return "select_next" if state.get("stop_reason") else next_node

    return router


def build_graph(agent: ResearchAgent, checkpointer=None):
    g = StateGraph(ResearchState)
    for name in ("analyze", "plan", "select_next", "route", "retrieve", "evaluate", "decide",
                 "critique", "synthesize", "verify"):
        g.add_node(name, getattr(agent, name))
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "plan")
    g.add_edge("plan", "select_next")
    g.add_conditional_edges("select_next", agent.after_select,
                            {"route": "route", "critique": "critique", "synthesize": "synthesize"})
    g.add_conditional_edges("route", _unless_stopped("retrieve"),
                            {"retrieve": "retrieve", "select_next": "select_next"})
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges("evaluate", _unless_stopped("decide"),
                            {"decide": "decide", "select_next": "select_next"})
    g.add_edge("decide", "select_next")
    g.add_edge("critique", "select_next")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=checkpointer)


def recursion_limit(b: AgentBudgetSettings) -> int:
    per_round = 5  # select, route, retrieve, evaluate, decide
    total_subs = b.max_subquestions + b.max_followups
    return 10 + total_subs * b.max_rounds_per_subquestion * per_round + 10
