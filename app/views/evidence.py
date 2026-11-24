from __future__ import annotations

import uuid

import streamlit as st
from components.common import reference_line, require_project, services

from researchmind.db.models import Chunk, Paper
from researchmind.ingestion.render import render_highlight

st.title("Evidence Explorer")
svc = services()
pid = require_project()

invs = [i for i in svc.workspace.investigations(pid) if i.report]
tab_inv, tab_all = st.tabs(["Evidence of an investigation", "Reusable evidence (project)"])


def show_page(chunk_id: str, text: str, key: str) -> None:
    with svc.sessionmaker() as s:
        chunk = s.get(Chunk, uuid.UUID(chunk_id))
        paper = s.get(Paper, chunk.paper_id) if chunk else None
    if not chunk or not paper or not paper.pdf_path or chunk.page_start < 1:
        st.caption("No PDF page for this passage (metadata-only paper).")
        return
    if st.toggle("Show original page", key=key):
        try:
            png, n = render_highlight(paper.pdf_path, chunk.page_start, text)
            st.image(png, caption=f"{paper.title} — page {chunk.page_start}"
                     + ("" if n else " (passage could not be located for highlighting)"))
        except Exception as exc:
            st.error(f"Cannot render page: {exc}")


with tab_inv:
    if not invs:
        st.info("No completed investigation in this project.")
    else:
        inv = st.selectbox("Investigation", invs, format_func=lambda i: i.question[:100])
        state = svc.runner.state(inv.id)
        evidence = state.get("evidence", {})
        cited = {r["id"] for r in inv.report.get("references", [])}
        c = st.columns(4)
        rel = c[0].multiselect("Relevance", ["high", "partial", "irrelevant", None], default=["high", "partial"],
                               format_func=lambda x: x or "not graded")
        stance = c[1].multiselect("Stance", ["supports", "contradicts", "neutral", None],
                                  default=["supports", "contradicts", "neutral", None],
                                  format_func=lambda x: x or "not graded")
        only_cited = c[2].checkbox("Cited in report only", value=False)
        flagged = c[3].checkbox("Injection-flagged only", value=False)
        items = [e for e in evidence.values() if e.get("relevance") in rel and e.get("stance") in stance
                 and (not only_cited or e["id"] in cited) and (not flagged or e.get("injection_flag"))]
        st.caption(f"{len(items)} of {len(evidence)} retrieved passages")
        for e in sorted(items, key=lambda x: int(x["id"][1:])):
            with st.container(border=True):
                pages = f"p. {e['page_start']}" if e["page_start"] == e["page_end"] else f"pp. {e['page_start']}-{e['page_end']}"
                ref = {"paper_title": e["paper_title"], "year": e.get("year"), "section": e.get("section_title"),
                       "pages": pages if e["page_start"] > 0 else "abstract (metadata)",
                       "doi": e.get("doi"), "arxiv_id": e.get("arxiv_id")}
                st.markdown(f"`{e['id']}` {reference_line(ref)}" + (" · **cited**" if e["id"] in cited else ""))
                scores = " · ".join(f"{k} {v:.3f}" for k, v in e.get("scores", {}).items())
                st.caption(f"relevance: {e.get('relevance') or 'not graded'} · stance: "
                           f"{e.get('stance') or '-'} · round {e.get('round')} · {scores}"
                           + (f" · subquestions: {', '.join(e.get('subquestion_ids', []))}"))
                if e.get("grade_note"):
                    st.caption(f"Grader: {e['grade_note']}")
                if e.get("injection_flag"):
                    st.warning("Instruction-like text detected in this passage; it was treated as data.")
                st.markdown(f"> {e['text']}")
                show_page(e["chunk_id"], e["text"], key=f"pg-{inv.id}-{e['id']}")

with tab_all:
    reusable = svc.workspace.reusable_evidence(pid)
    if not reusable:
        st.info("Evidence judged relevant in past investigations appears here.")
    query = st.text_input("Filter", placeholder="word in passage or question")
    for r in reusable:
        if query and query.lower() not in (r["text"] + r["question"]).lower():
            continue
        with st.container(border=True):
            st.markdown(f"**{r['paper_title']}** ({r['year'] or '?'}) · pp. {r['pages']} · {r['relevance']}")
            st.caption(f"From: {r['question']}")
            st.markdown(f"> {r['text']}")
