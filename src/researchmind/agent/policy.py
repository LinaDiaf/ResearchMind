"""Deterministic routing and stopping rules.

The LLM proposes (queries, gradings); these rules decide. Keeping them in plain code makes the
agent's control flow testable and its stopping behaviour predictable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from researchmind.agent.schemas import Intent
from researchmind.agent.state import SubQuestion

# Section kinds searched first for each intent (the planner may override).
INTENT_SECTIONS: dict[Intent, list[str]] = {
    Intent.method_comparison: ["method", "experiments", "results"],
    Intent.evidence_verification: ["results", "experiments", "abstract"],
    Intent.contradiction_analysis: ["results", "discussion", "experiments"],
    Intent.gap_exploration: ["limitations", "conclusion", "discussion"],
    Intent.lookup: [],
    Intent.single_paper: [],
    Intent.multi_paper_synthesis: [],
}

INTENT_ASPECTS: dict[Intent, list[str]] = {
    Intent.method_comparison: ["method", "datasets and metrics", "reported results"],
    Intent.evidence_verification: ["direct evidence", "experimental conditions"],
    Intent.contradiction_analysis: ["findings on each side", "experimental conditions"],
    Intent.gap_exploration: ["reported limitations", "future work"],
    Intent.lookup: ["direct answer"],
    Intent.single_paper: ["direct answer"],
    Intent.multi_paper_synthesis: ["findings across papers"],
}

_EXACT = re.compile(r"\b(?:[A-Z]{2,}[\w-]*|\w+-\w+|\w*\d[\w.@]*)\b|\"[^\"]+\"")


def exact_term_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(_EXACT.findall(text)) / len(words)


@dataclass
class RoutePlan:
    strategy: str
    dense_weight: float
    lexical_weight: float
    section_kinds: list[str]
    k: int
    mode: str  # initial | refine_missing | refine_conflict | broaden


def plan_route(sub: SubQuestion, queries: list[str], round_no: int, base_k: int,
               conflict: bool = False) -> RoutePlan:
    """Strategy escalation across rounds.

    round 1: hybrid, weights from the query's vocabulary, intent/planner section targeting
    round 2: hybrid aimed at missing aspects (or both sides of a conflict), lexical weight up,
             sections widened
    round 3+: lexical-heavy, no section filter, larger k
    """
    ratio = max((exact_term_ratio(q) for q in queries), default=0.0)
    dense_w, lex_w = (0.8, 1.5) if ratio >= 0.25 else (1.2, 0.8)
    if round_no <= 1:
        return RoutePlan("hybrid", dense_w, lex_w, list(sub.target_sections), base_k, "initial")
    if round_no == 2:
        mode = "refine_conflict" if conflict else "refine_missing"
        sections = list(sub.target_sections)
        if sections and not conflict:
            sections = sorted(set(sections) | {"results", "discussion", "abstract"})
        return RoutePlan("hybrid", dense_w, lex_w + 0.4, sections, base_k + 2, mode)
    return RoutePlan("hybrid", 0.5, 1.8, [], base_k + 4, "broaden")


def match_aspects(found: list[str], required: list[str], threshold: float = 70.0) -> list[str]:
    """Map the model's aspect strings onto the required aspects (tolerant to rewording)."""
    out = []
    for req in required:
        for f in found:
            if fuzz.token_set_ratio(req.lower(), f.lower()) >= threshold:
                out.append(req)
                break
    return out


@dataclass
class Decision:
    action: str  # answered | refine | insufficient
    reason: str
    conflict: bool = False


def decide(sub: SubQuestion, *, new_accepted: int, has_high: bool, conflict: bool,
           max_rounds: int, stale_limit: int = 2) -> Decision:
    missing = [a for a in sub.required_aspects if a not in sub.covered_aspects]
    if conflict and not sub.conflict_followup_done and sub.rounds < max_rounds:
        return Decision("refine", "conflicting evidence: retrieve experimental conditions",
                        conflict=True)
    if not missing and has_high:
        return Decision("answered", "all required aspects covered")
    if sub.rounds >= max_rounds:
        return Decision("insufficient", f"retrieval rounds exhausted; missing: {', '.join(missing) or 'strong evidence'}")
    if sub.stale_rounds >= stale_limit:
        return Decision("insufficient", f"{stale_limit} rounds without new relevant evidence")
    if not missing and not has_high and new_accepted == 0 and sub.evidence_ids:
        return Decision("answered", "aspects covered by partial evidence only")
    return Decision("refine", f"missing: {', '.join(missing) or 'high-relevance evidence'}")


def heuristic_intent(question: str) -> Intent:
    q = question.lower()
    if re.search(r"\b(compare|comparison|versus|vs\.?|differ|better than|outperform)\b", q):
        return Intent.method_comparison
    if re.search(r"\b(contradict|conflict|disagree|inconsistent)\b", q):
        return Intent.contradiction_analysis
    if re.search(r"\b(gap|open problem|future work|limitation|underexplored|unexplored)\b", q):
        return Intent.gap_exploration
    if re.search(r"\b(is it true|verify|evidence that|does .* really|confirm)\b", q):
        return Intent.evidence_verification
    return Intent.lookup


def heuristic_terms(question: str) -> list[str]:
    terms = [m.group(0).strip('"') for m in _EXACT.finditer(question)]
    return list(dict.fromkeys(terms))[:8]
