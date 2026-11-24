"""ResearchMind — Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="ResearchMind", page_icon=":material/biotech:", layout="wide")

VIEWS = Path(__file__).resolve().parent / "views"
pages = [
    st.Page(VIEWS / "dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page(VIEWS / "projects.py", title="Research Projects", icon=":material/folder:"),
    st.Page(VIEWS / "library.py", title="Literature Library", icon=":material/library_books:"),
    st.Page(VIEWS / "assistant.py", title="Research Assistant", icon=":material/psychology:"),
    st.Page(VIEWS / "comparison.py", title="Paper Comparison", icon=":material/table_chart:"),
    st.Page(VIEWS / "evidence.py", title="Evidence Explorer", icon=":material/fact_check:"),
    st.Page(VIEWS / "knowledge_graph.py", title="Knowledge Graph", icon=":material/hub:"),
    st.Page(VIEWS / "settings.py", title="Settings", icon=":material/settings:"),
]
st.navigation(pages).run()
