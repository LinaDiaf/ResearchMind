from __future__ import annotations

import pandas as pd
import streamlit as st
from components.common import require_project, services

from researchmind.comparison import DIMENSIONS
from researchmind.db.models import PaperStatus

st.title("Paper Comparison")
st.caption("Each cell is filled from passages of that paper only and cites them. Cells marked "
           "“not reported” mean the retrieved passages do not state it.")
svc = services()
pid = require_project()

papers = [p for p in svc.library.project_papers(pid) if p.status is PaperStatus.ingested]
if len(papers) < 2:
    st.info("Ingest at least two full-text papers to compare them.")
    st.stop()

chosen = st.multiselect("Papers (2–5)", papers, max_selections=5, format_func=lambda p: p.title)
dims = st.multiselect("Dimensions", list(DIMENSIONS),
                      default=["Method / architecture", "Datasets", "Evaluation metrics",
                               "Main results", "Limitations"])
if st.button("Compare", type="primary", disabled=len(chosen) < 2 or not dims):
    with st.spinner("Retrieving passages and filling the table…"):
        st.session_state["comparison"] = svc.comparator().compare(
            pid, [(p.id, p.title) for p in chosen], dims)

result = st.session_state.get("comparison")
if result:
    def cell(c: dict) -> str:
        if c["status"] == "not_reported":
            return "not reported"
        return f"{c['value']} [{', '.join(c['citations'])}]"

    table = pd.DataFrame({title: {d: cell(c) for d, c in row.items()}
                          for title, row in result["rows"].items()})
    st.dataframe(table, width="stretch")
    if any(c["source"] == "extractive" for row in result["rows"].values() for c in row.values()):
        st.caption("Some cells show the most relevant passage verbatim (model output was unavailable).")
    st.download_button("Download CSV", table.to_csv().encode(), "comparison.csv")
    with st.expander("Cited passages"):
        cited = {e for row in result["rows"].values() for c in row.values() for e in c["citations"]}
        for eid in sorted(cited, key=lambda x: int(x[1:])):
            ev = result["evidence"][eid]
            st.markdown(f"`{eid}` **{ev['paper_title']}** · {ev.get('section') or ''} · p. {ev['page_start']}")
            st.markdown(f"> {ev['text']}")
