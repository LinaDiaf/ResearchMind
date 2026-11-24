from __future__ import annotations

import streamlit as st
from components.common import services

st.title("Settings")
st.caption("Configuration is read from environment variables (the `.env` file). Restart the app "
           "after changing it. Secrets are never displayed.")
svc = services()
s = svc.settings

st.subheader("Models")
st.table({
    "LLM provider": s.llm_provider, "LLM model": s.llm_model,
    "Ollama URL": s.ollama_url if s.llm_provider == "ollama" or s.embedding_provider == "ollama" else "-",
    "Ollama context / thinking": f"{s.ollama_num_ctx} / {s.ollama_think}",
    "Gemini key": "set" if s.gemini_api_key else "not set",
    "Embeddings": f"{s.embedding_provider}:{s.embedding_model} (dim {s.embedding_dim})",
    "Reranker": s.reranker_model,
})
st.subheader("Agent budget")
st.table(svc.budget.model_dump())

st.subheader("Connection test")
if st.button("Test LLM and embeddings"):
    from pydantic import BaseModel

    from researchmind.llm import LLMClient, Message

    class Probe(BaseModel):
        answer: str

    try:
        out = LLMClient(svc.provider).structured(
            "settings_check", [Message("user", "Reply with JSON: the answer to 2+3.")], Probe, max_tokens=100)
        st.success(f"LLM answered: {out.answer}")
    except Exception as exc:
        st.error(f"LLM: {exc}")
    try:
        st.success(f"Embeddings: dimension {len(svc.embedder.embed_query('test'))}")
    except Exception as exc:
        st.error(f"Embeddings: {exc}")
