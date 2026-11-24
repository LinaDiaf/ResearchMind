"""Shared helpers for the Streamlit pages."""

from __future__ import annotations

import uuid

import streamlit as st

from researchmind.services import Services, build_services
from researchmind.workspace.background import BackgroundRuns


@st.cache_resource(show_spinner=False)
def services() -> Services:
    return build_services()


@st.cache_resource(show_spinner=False)
def background() -> BackgroundRuns:
    return BackgroundRuns()


def project_selector() -> uuid.UUID | None:
    """Active project, chosen in the sidebar and shared by all pages."""
    svc = services()
    try:
        projects = svc.library.list_projects()
    except Exception as exc:
        st.sidebar.error(f"Database unavailable: {exc}")
        return None
    if not projects:
        st.sidebar.info("Create a project in **Research Projects** to start.")
        return None
    ids = [p.id for p in projects]
    current = st.session_state.get("project_id")
    index = ids.index(current) if current in ids else 0
    chosen = st.sidebar.selectbox("Active project", projects, index=index,
                                  format_func=lambda p: p.name, key="project_select")
    st.session_state["project_id"] = chosen.id
    return chosen.id


def require_project() -> uuid.UUID:
    pid = project_selector()
    if pid is None:
        st.warning("No project selected. Create one in **Research Projects**.")
        st.stop()
    return pid


CLAIM_BADGES = {
    "reported_finding": ("Reported finding", "blue"),
    "author_interpretation": ("Author interpretation", "violet"),
    "ai_synthesis": ("AI synthesis", "orange"),
    "hypothesis": ("Hypothesis", "gray"),
}
SUPPORT_BADGES = {
    "supported": ("supported", "green"),
    "partially_supported": ("partially supported", "orange"),
    "unsupported": ("unsupported", "red"),
    "not_checked": ("not checked", "gray"),
    "uncited": ("no citation", "gray"),
}


def badge(label: str, color: str) -> str:
    return f":{color}-badge[{label}]"


def reference_line(ref: dict) -> str:
    ident = f" · doi:{ref['doi']}" if ref.get("doi") else (f" · arXiv:{ref['arxiv_id']}" if ref.get("arxiv_id") else "")
    year = f" ({ref['year']})" if ref.get("year") else ""
    section = f" · {ref['section']}" if ref.get("section") else ""
    return f"**{ref['paper_title']}**{year}{section} · {ref['pages']}{ident}"


def render_report(report: dict) -> None:
    if not report:
        return
    st.subheader(report.get("title") or "Report")
    if report.get("notice"):
        st.warning(report["notice"])
    refs = {r["id"]: r for r in report.get("references", [])}
    for sec in report.get("sections", []):
        st.markdown(f"#### {sec['heading']}")
        for c in sec["claims"]:
            ctype = badge(*CLAIM_BADGES.get(c["claim_type"], (c["claim_type"], "gray")))
            sup = badge(*SUPPORT_BADGES.get(c["support"], (c["support"], "gray")))
            cites = " ".join(f"`{i}`" for i in c["citations"])
            st.markdown(f"{ctype} {sup} {c['text']} {cites}")
            if c.get("note"):
                st.caption(f"Verification note: {c['note']}")
            if c["citations"]:
                with st.expander("Evidence", expanded=False):
                    for i in c["citations"]:
                        if r := refs.get(i):
                            st.markdown(f"`{i}` {reference_line(r)}")
                            if r.get("injection_flag"):
                                st.caption(":warning: passage contains instruction-like text (treated as data)")
                            st.markdown(f"> {r['text']}")
    if report.get("insufficient"):
        st.markdown("#### Insufficient evidence")
        for item in report["insufficient"]:
            miss = ", ".join(item.get("missing_aspects") or []) or "strong evidence"
            st.markdown(f"- **{item['question']}** — missing: {miss}. {item.get('note', '')}")
            if item.get("suggested_queries"):
                st.caption("Try importing papers for: " + "; ".join(item["suggested_queries"]))
    if report.get("removed_claims"):
        with st.expander(f"Removed claims ({len(report['removed_claims'])})"):
            st.caption("Statements the model wrote that failed citation or support checks.")
            for r in report["removed_claims"]:
                st.markdown(f"- {r['text']} — _{r['reason']}_")
    if stats := report.get("stats"):
        st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats.items() if v is not None))
