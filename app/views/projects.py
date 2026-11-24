from __future__ import annotations

import streamlit as st
from components.common import project_selector, services

from researchmind.db.models import NoteKind

st.title("Research Projects")
svc = services()

with st.form("new_project", clear_on_submit=True):
    st.markdown("**New project**")
    name = st.text_input("Name", max_chars=200)
    desc = st.text_area("Description", height=80)
    if st.form_submit_button("Create", type="primary") and name.strip():
        st.session_state["project_id"] = svc.library.create_project(name.strip(), desc)
        st.success(f"Project “{name}” ready.")

pid = project_selector()
if pid is None:
    st.stop()

papers = svc.library.project_papers(pid)
invs = svc.workspace.investigations(pid)
c = st.columns(3)
c[0].metric("Papers", len(papers))
c[1].metric("Investigations", len(invs))
c[2].metric("Notes", len(svc.workspace.notes(pid)))

st.subheader("Investigations")
if not invs:
    st.caption("None yet.")
for inv in invs:
    with st.container(border=True):
        st.markdown(f"**{inv.question}**")
        st.caption(f"{inv.status.value} · {inv.created_at:%Y-%m-%d %H:%M}"
                   + (f" · error: {inv.error}" if inv.error else ""))
        if st.button("Open", key=f"open-{inv.id}"):
            st.session_state["investigation_id"] = inv.id
            st.switch_page("views/assistant.py")

st.subheader("Notes and open questions")
with st.form("note", clear_on_submit=True):
    body = st.text_area("Note", height=80)
    kind = st.radio("Type", [NoteKind.note, NoteKind.open_question], horizontal=True,
                    format_func=lambda k: "Note" if k is NoteKind.note else "Open question")
    if st.form_submit_button("Add") and body.strip():
        svc.workspace.add_note(pid, body, kind)
        st.rerun()
for note in svc.workspace.notes(pid):
    cols = st.columns([0.85, 0.15])
    icon = ":material/help:" if note.kind is NoteKind.open_question else ":material/notes:"
    cols[0].markdown(f"{icon} {note.body}")
    cols[0].caption(f"{note.created_at:%Y-%m-%d %H:%M}")
    if cols[1].button("Delete", key=f"del-{note.id}"):
        svc.workspace.delete_note(note.id)
        st.rerun()
