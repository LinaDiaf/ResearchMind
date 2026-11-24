from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from components.common import require_project, services

st.title("Literature Library")
svc = services()
pid = require_project()

tab_upload, tab_discover, tab_papers = st.tabs(["Upload PDFs", "Discover papers", "Papers in project"])

with tab_upload:
    files = st.file_uploader("Scientific PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Ingest", type="primary"):
        progress = st.progress(0.0)
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(files, 1):
                path = Path(tmp) / Path(f.name).name
                path.write_bytes(f.getvalue())
                res = svc.library.ingest_pdf(pid, path, reindex=False)
                rows.append({"file": f.name, "status": res.status, "title": res.title,
                             "chunks": res.n_chunks, "duplicate rule": res.duplicate_rule,
                             "warnings": "; ".join(res.warnings), "error": res.error})
                progress.progress(i / len(files))
        svc.library.reindex(pid)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_discover:
    with st.form("search"):
        q = st.text_input("Search query", placeholder="e.g. retrieval augmented generation evaluation")
        c = st.columns(3)
        sources = c[0].multiselect("Sources", list(svc.discovery.sources),
                                   default=list(svc.discovery.sources))
        year_min = c[1].number_input("From year", min_value=0, max_value=2100, value=0, step=1)
        limit = c[2].slider("Results per source", 5, 25, 10)
        submitted = st.form_submit_button("Search", type="primary")
    if submitted and q.strip():
        with st.spinner("Searching…"):
            records, errors = svc.discovery.search(q, sources, limit, int(year_min) or None)
        st.session_state["discovery_results"] = records
        for name, err in errors.items():
            st.warning(f"{name}: {err}")
    records = st.session_state.get("discovery_results", [])
    if records:
        df = pd.DataFrame([{"import": False, "title": r.title, "year": r.year, "venue": r.venue,
                            "citations": r.citation_count, "open PDF": bool(r.pdf_url),
                            "id": r.doi or (f"arXiv:{r.arxiv_id}" if r.arxiv_id else ""),
                            "source": r.source} for r in records])
        edited = st.data_editor(df, hide_index=True, width="stretch",
                                disabled=[c for c in df.columns if c != "import"])
        download = st.checkbox("Download open-access PDFs when available", value=True)
        if st.button("Import selected"):
            chosen = [records[i] for i in edited.index[edited["import"]]]
            for r in chosen:
                with st.spinner(f"Importing {r.title[:60]}…"):
                    res = svc.library.import_record(pid, r, download_pdf=download)
                st.write(f"{res.status}: {res.title}")

with tab_papers:
    papers = svc.library.project_papers(pid)
    if not papers:
        st.info("No papers yet.")
    else:
        df = pd.DataFrame([{"title": p.title, "year": p.year, "status": p.status.value,
                            "pages": p.page_count, "doi": p.doi, "arxiv": p.arxiv_id,
                            "source": p.source,
                            "warnings": "; ".join(map(str, p.parse_warnings or []))} for p in papers])
        st.dataframe(df, hide_index=True, width="stretch")
        target = st.selectbox("Remove a paper from this project", [None, *papers],
                              format_func=lambda p: "—" if p is None else p.title)
        if target is not None and st.button("Remove from project"):
            svc.library.remove_from_project(pid, target.id)
            st.rerun()
        if st.button("Rebuild keyword index"):
            st.success(f"BM25 index rebuilt over {svc.library.reindex(pid)} chunks.")
