"""Prompt templates (versioned with the code; the version is stored with each investigation)."""

from __future__ import annotations

PROMPT_VERSION = "2026-11-24.1"

SYSTEM_BASE = (
    "You are ResearchMind, a careful scientific research assistant. You work only from the "
    "evidence you are given. You never invent papers, results, numbers, DOIs or page numbers. "
    "When evidence is missing you say so. Answer with JSON only, matching the requested schema."
)

ANALYZE = """Analyse this research question.

Question: {question}

Return:
- intent: one of lookup, single_paper, multi_paper_synthesis, method_comparison,
  evidence_verification, contradiction_analysis, gap_exploration
- complexity: "simple" if one search can answer it, else "complex"
- key_terms: up to 8 exact technical terms from the question (methods, models, datasets,
  metrics, named entities). Do not add terms that are not implied by the question.
- year_min / year_max: only if the question restricts publication years, else null
- rationale: one short sentence"""

PLAN = """Create a research plan for this question.

Question: {question}
Intent: {intent}
Key terms: {key_terms}
Papers in the library (titles): {library}

Decompose the question into at most {max_sub} subquestions that together answer it. For
each subquestion give:
- question: a self-contained question
- rationale: why it is needed (one sentence)
- required_aspects: 1-4 short noun phrases that an answer must contain
  (for example "dataset", "MRR@10 value", "training hardware")
- target_sections: section kinds most likely to contain the answer, chosen from
  abstract, introduction, related_work, method, experiments, results, discussion,
  limitations, conclusion (may be empty)
- search_queries: 1-3 short search queries using the technical vocabulary of papers
- priority: 1 (essential) to 3 (nice to have)
Use a single subquestion when the question is simple."""

ROUTE = """You are refining a literature search that did not yet find everything needed.

Subquestion: {question}
Required aspects still missing: {missing}
Situation: {situation}
Queries already tried: {tried}
Most relevant passages found so far (for vocabulary only):
{context}

Write 1-3 NEW search queries (different from those already tried) aimed at the missing
aspects. Prefer the exact terms used in the passages (method names, dataset names, metrics).
Optionally list section_kinds to search (results, experiments, method, limitations,
discussion, conclusion, abstract, introduction, related_work)."""

GRADE = """Grade retrieved passages for a subquestion.

{untrusted_rule}

Subquestion: {question}
Required aspects: {aspects}
Aspects already covered by earlier evidence: {covered}

Passages:
{passages}

For EVERY passage id above return a grade:
- relevance: "high" (directly answers part of the subquestion), "partial" (useful context),
  or "irrelevant"
- stance: "supports", "contradicts" or "neutral" with respect to the subquestion's premise
- aspects: which required aspects it covers, copied exactly from the list
- note: at most one short sentence about what the passage reports
Then give missing_aspects (required aspects not covered by any evidence yet) and set
conflict=true only if two passages report incompatible findings (explain in conflict_note)."""

CRITIQUE = """Review the evidence gathered for a research question before the report is written.

{untrusted_rule}

Question: {question}
Subquestions and their outcome:
{subquestions}

Evidence (id, paper, finding):
{evidence}

Identify:
- issues: important gaps or unsupported directions (short sentences)
- contradictions: findings that disagree, with the evidence ids involved
- followups: at most {max_followups} new subquestions that would close the most important gap
  (same fields as a research plan; leave empty if the evidence is adequate)"""

SYNTHESIZE = """Write an evidence-grounded research report.

{untrusted_rule}

Question: {question}
Intent: {intent}

Subquestions and outcome:
{subquestions}

Evidence you may cite (ONLY these ids exist):
{evidence}

Known contradictions: {contradictions}

Rules:
- Every claim cites the evidence ids that support it in evidence_ids, e.g. ["E3", "E7"].
- claim_type:
  - "reported_finding": a result or fact reported by a paper (must cite evidence)
  - "author_interpretation": the authors' own interpretation or claim (must cite evidence)
  - "ai_synthesis": your own combination or comparison of several pieces of evidence
  - "hypothesis": a possible explanation or research direction not established by the evidence
- Report numbers exactly as written in the passages, with their dataset/metric/setting.
- Never mention papers, numbers or facts that are not in the evidence.
- If a subquestion has insufficient evidence, say so in a claim of type "ai_synthesis".
- Start with a section "Answer" giving the direct answer in 1-4 claims, then sections for the
  main findings, conflicting evidence (if any) and limitations of the evidence."""

VERIFY = """Check whether each claim is supported by the passages it cites.

{untrusted_rule}

Claims and cited passages:
{items}

For every claim id return support:
- "supported": the cited passages state it
- "partially_supported": the passages support part of it or a weaker version
- "unsupported": the passages do not state it (or contradict it)
and a short note when it is not fully supported."""
