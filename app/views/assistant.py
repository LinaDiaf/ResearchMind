from __future__ import annotations

import uuid

import streamlit as st
from components.common import background, render_report, require_project, services

from researchmind.db.models import InvestigationStatus

st.title("Research Assistant")
svc = services()
runs = background()
pid = require_project()

with st.form("ask"):
    question = st.text_area("Research question", height=90,
                            placeholder="e.g. How do learned sparse and dense retrievers compare "
                                        "out of domain, and what limitations do the authors report?")
    if st.form_submit_button("Investigate", type="primary") and question.strip():
        if not svc.library.project_papers(pid):
            st.error("This project has no papers yet.")
        else:
            inv_id = svc.runner.create(pid, question)
            runs.start(inv_id, lambda: svc.runner.run(inv_id))
            st.session_state["investigation_id"] = inv_id

invs = svc.workspace.investigations(pid)
if not invs:
    st.stop()
ids = [i.id for i in invs]
current = st.session_state.get("investigation_id")
choice = st.selectbox("Investigation", invs, index=ids.index(current) if current in ids else 0,
                      format_func=lambda i: f"{i.question[:90]} — {i.status.value}")
st.session_state["investigation_id"] = choice.id


@st.fragment(run_every=3)
def progress(inv_id: uuid.UUID) -> None:
    inv = svc.workspace.investigation(inv_id)
    running = runs.is_running(inv_id)
    events = svc.workspace.events(inv_id)
    status = inv.status
    if status in (InvestigationStatus.running, InvestigationStatus.interrupted) and not running:
        st.warning("This investigation is not running"
                   + (f" (last error: {inv.error})" if inv.error else "")
                   + ". It can be resumed from its last checkpoint.")
        if st.button("Resume", type="primary", key=f"resume-{inv_id}"):
            runs.start(inv_id, lambda: svc.runner.run(inv_id))
            st.rerun()
    plan = next((e for e in events if e.event_type == "plan"), None)
    if plan:
        with st.container(border=True):
            st.markdown("**Research plan**")
            for sq in plan.payload.get("subquestions", []):
                st.markdown(f"- `{sq['id']}` {sq['question']}  \n  "
                            f"<small>aspects: {', '.join(sq.get('aspects', []))}"
                            + (f" · sections: {', '.join(sq['sections'])}" if sq.get("sections") else "")
                            + "</small>", unsafe_allow_html=True)
    label = "Running…" if running else f"Execution trace ({len(events)} steps)"
    with st.status(label, state="running" if running else "complete", expanded=running):
        for e in events:
            st.markdown(f"`{e.node}` {e.message}")
            if e.event_type == "route" and e.payload.get("queries"):
                st.caption("queries: " + " | ".join(e.payload["queries"]))
    if status is InvestigationStatus.completed and not running:
        usage = svc.workspace.llm_usage(inv_id)
        st.caption(f"LLM calls: {usage['calls']} · tokens in/out: {usage['prompt_tokens']}/"
                   f"{usage['completion_tokens']} · LLM time: {usage['llm_seconds']} s")
        render_report(inv.report)
        st.download_button("Download report (Markdown)", _markdown(inv.report),
                           file_name="researchmind-report.md")
    if err := runs.error(inv_id):
        st.error(err)


def _markdown(report: dict) -> str:
    from researchmind.synthesis.report import render_markdown

    return render_markdown(report)


progress(choice.id)
