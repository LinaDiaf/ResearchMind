"""Report assembly, citation validation and rendering.

The guarantees enforced here do not depend on the model:
- a citation can only reference an evidence id gathered during the investigation;
- a claim presented as a paper's finding or the authors' interpretation must keep at least one
  valid citation, otherwise it is removed;
- bibliographic details (title, year, DOI, pages) come from the database, never from the model.
"""

from __future__ import annotations

import re
from typing import Any

from researchmind.agent.schemas import ReportDraft
from researchmind.agent.state import Evidence

CITED_TYPES = {"reported_finding", "author_interpretation"}
_INLINE_ID = re.compile(r"\[(E\d+)\]")

CLAIM_TYPE_LABELS = {
    "reported_finding": "Reported finding",
    "author_interpretation": "Author interpretation",
    "ai_synthesis": "AI synthesis",
    "hypothesis": "Hypothesis",
}


def validate_draft(draft: ReportDraft, allowed: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (sections, removed_claims). Invalid citation ids are dropped."""
    sections: list[dict] = []
    removed: list[dict] = []
    n = 0
    for sec in draft.sections:
        claims = []
        for c in sec.claims:
            ids = list(dict.fromkeys(c.evidence_ids + _INLINE_ID.findall(c.text)))
            valid = [i for i in ids if i in allowed]
            invalid = [i for i in ids if i not in allowed]
            text = _INLINE_ID.sub("", c.text).strip()
            text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text))
            if c.claim_type in CITED_TYPES and not valid:
                removed.append({"text": text, "claim_type": c.claim_type,
                                "reason": "no valid citation" + (f" (cited unknown {', '.join(invalid)})" if invalid else "")})
                continue
            n += 1
            claims.append({
                "id": f"C{n}", "text": text, "claim_type": c.claim_type, "citations": valid,
                "invalid_citations": invalid, "support": "not_checked" if valid else "uncited",
                "note": "",
            })
        if claims:
            sections.append({"heading": sec.heading.strip() or "Findings", "claims": claims})
    return sections, removed


def apply_verdicts(sections: list[dict], verdicts: dict[str, tuple[str, str]],
                   keep_unsupported: bool = False) -> list[dict]:
    """Attach support labels; unsupported claims are removed unless keep_unsupported."""
    removed = []
    for sec in sections:
        kept = []
        for c in sec["claims"]:
            if c["id"] in verdicts:
                c["support"], c["note"] = verdicts[c["id"]]
            if c["support"] == "unsupported" and not keep_unsupported:
                removed.append({"text": c["text"], "claim_type": c["claim_type"],
                                "reason": "not supported by its cited passages"
                                + (f": {c['note']}" if c["note"] else "")})
                continue
            kept.append(c)
        sec["claims"] = kept
    sections[:] = [s for s in sections if s["claims"]]
    return removed


def evidence_reference(ev: Evidence) -> dict[str, Any]:
    pages = (f"p. {ev.page_start}" if ev.page_start == ev.page_end else
             f"pp. {ev.page_start}-{ev.page_end}") if ev.page_start > 0 else "abstract (metadata)"
    return {
        "id": ev.id, "chunk_id": ev.chunk_id, "paper_id": ev.paper_id,
        "paper_title": ev.paper_title, "year": ev.year, "doi": ev.doi, "arxiv_id": ev.arxiv_id,
        "section": ev.section_title, "section_kind": ev.section_kind, "pages": pages,
        "page_start": ev.page_start, "page_end": ev.page_end, "text": ev.text,
        "relevance": ev.relevance, "stance": ev.stance, "injection_flag": ev.injection_flag,
        "scores": ev.scores,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report.get('title') or 'Research report'}", "",
             f"**Question:** {report.get('question', '')}", ""]
    if report.get("notice"):
        lines += [f"> {report['notice']}", ""]
    for sec in report.get("sections", []):
        lines += [f"## {sec['heading']}", ""]
        for c in sec["claims"]:
            cites = " ".join(f"[{i}]" for i in c["citations"])
            tag = CLAIM_TYPE_LABELS.get(c["claim_type"], c["claim_type"])
            flag = ""
            if c["support"] == "partially_supported":
                flag = " _(partially supported)_"
            elif c["support"] == "uncited":
                flag = " _(no direct citation)_"
            elif c["support"] == "unsupported":
                flag = " _(not supported by cited passages)_"
            lines.append(f"- **{tag}:** {c['text']} {cites}{flag}".rstrip())
        lines.append("")
    if report.get("insufficient"):
        lines += ["## Insufficient evidence", ""]
        for item in report["insufficient"]:
            miss = ", ".join(item.get("missing_aspects") or []) or "strong evidence"
            lines.append(f"- {item['question']} — missing: {miss}. {item.get('note', '')}".rstrip())
            if item.get("suggested_queries"):
                lines.append("  - Suggested literature searches: "
                             + "; ".join(f"`{q}`" for q in item["suggested_queries"]))
        lines.append("")
    if report.get("removed_claims"):
        lines += ["## Removed claims", "",
                  "_Claims the model wrote that failed citation checks (shown for transparency)._", ""]
        lines += [f"- {r['text']} — {r['reason']}" for r in report["removed_claims"]]
        lines.append("")
    refs = report.get("references", [])
    if refs:
        lines += ["## Evidence", ""]
        for r in refs:
            ident = f"doi:{r['doi']}" if r.get("doi") else (f"arXiv:{r['arxiv_id']}" if r.get("arxiv_id") else "")
            year = f" ({r['year']})" if r.get("year") else ""
            sec = f", {r['section']}" if r.get("section") else ""
            lines.append(f"- **[{r['id']}]** {r['paper_title']}{year}{sec}, {r['pages']}. {ident}".rstrip())
    return "\n".join(lines).strip() + "\n"
