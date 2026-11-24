# ResearchMind

**An agentic RAG system for scientific literature intelligence.**

ResearchMind investigates research questions over a library of scientific publications. A
bounded LangGraph agent analyses the question, decomposes it into subquestions, chooses a
retrieval strategy for each one, grades the evidence it retrieves, retrieves again where coverage
is missing, and writes a report in which every statement links to the original passage and page.
When the evidence is not sufficient, it says so.

> **Project status: design phase.** The architecture is described in
> [ARCHITECTURE.md](ARCHITECTURE.md) and the roadmap in
> [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Nothing below is claimed as working until the
> corresponding milestone is marked as implemented and tested.

## Key ideas

- **Agentic, bounded workflow**: planning, adaptive retrieval routing, evidence evaluation,
  targeted re-retrieval, critique and synthesis, with explicit budgets on every loop.
- **Hybrid retrieval**: pgvector dense search, BM25, metadata filters, reciprocal rank fusion and
  cross-encoder reranking (graph retrieval in a later phase).
- **Verifiable provenance**: citations can only point to retrieved passages; each resolves to a
  paper, a page and character offsets, shown highlighted on the original PDF page.
- **Claim types**: reported findings, author interpretations, AI synthesis and hypotheses are
  labelled separately.
- **Provider-independent**: Gemini, local models through Ollama (first target: Qwen 3.5 9B) or
  any OpenAI-compatible server.
- **Evaluation-driven**: comparison against vector, hybrid and reranked RAG baselines with
  ablations (planned for Phase 2).

## Stack

Python · Streamlit · LangGraph · PostgreSQL + pgvector · BM25 (bm25s) · PyMuPDF ·
sentence-transformers · Docker Compose · Pytest

## License

[MIT](LICENSE)
