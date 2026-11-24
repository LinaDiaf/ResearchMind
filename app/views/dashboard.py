from __future__ import annotations

import streamlit as st
from components.common import project_selector, services
from sqlalchemy import func, select

from researchmind.db.models import Investigation, InvestigationStatus, Paper, PaperStatus, Project

st.title("ResearchMind")
st.caption("Agentic RAG for scientific literature: planned investigations, adaptive hybrid "
           "retrieval, evaluated evidence and verifiable citations.")

pid = project_selector()
svc = services()
try:
    with svc.sessionmaker() as s:
        n_projects = s.scalar(select(func.count()).select_from(Project))
        n_papers = s.scalar(select(func.count()).select_from(Paper).where(Paper.status == PaperStatus.ingested))
        n_meta = s.scalar(select(func.count()).select_from(Paper).where(Paper.status == PaperStatus.metadata_only))
        n_inv = s.scalar(select(func.count()).select_from(Investigation))
        n_done = s.scalar(select(func.count()).select_from(Investigation)
                          .where(Investigation.status == InvestigationStatus.completed))
        recent = s.execute(select(Investigation, Project.name).join(Project)
                           .order_by(Investigation.created_at.desc()).limit(8)).all()
except Exception as exc:
    st.error(f"Cannot reach the database: {exc}")
    st.stop()

c = st.columns(4)
c[0].metric("Projects", n_projects)
c[1].metric("Full-text papers", n_papers, help=f"{n_meta} more with metadata only")
c[2].metric("Investigations", n_inv)
c[3].metric("Completed", n_done)

st.subheader("Recent investigations")
if not recent:
    st.info("No investigation yet. Add papers in **Literature Library**, then ask a question in "
            "**Research Assistant**.")
for inv, project_name in recent:
    with st.container(border=True):
        st.markdown(f"**{inv.question}**")
        st.caption(f"{project_name} · {inv.status.value} · {inv.created_at:%Y-%m-%d %H:%M}"
                   + (f" · model {inv.config.get('model')}" if inv.config else ""))

st.subheader("Configuration")
st.caption(f"LLM: `{svc.settings.llm_provider}:{svc.settings.llm_model}` · embeddings: "
           f"`{svc.settings.embedding_provider}:{svc.settings.embedding_model}` · reranker: "
           f"`{svc.settings.reranker_model}` · budget: {svc.budget.max_llm_calls} LLM calls, "
           f"{svc.budget.max_rounds_per_subquestion} rounds per subquestion")
