# ResearchMind — Implementation Plan

Status legend: ⬜ not started · 🟨 in progress · ✅ implemented **and tested** (with the test that
proves it). A milestone is only marked ✅ when its tests pass; what was not tested is listed.

## Environments

| Environment | Used for | Limits |
|---|---|---|
| Development workspace | writing code, unit tests (no network models) | no GPU, no pgvector, no access to Hugging Face or scholarly APIs |
| GitHub Actions CI | lint, types, unit tests, **integration tests with a real PostgreSQL + pgvector service** and small real models | CPU only |
| Evaluation machine (Windows, Docker Desktop, Ollama with `qwen3.5:9b` and `nomic-embed-text`) | end-to-end runs with Qwen, real API discovery, benchmarks | single local GPU, long runs |

## Phase 1 — Functional agentic RAG MVP

| # | Milestone | Content | Tests that close it | Status |
|---|---|---|---|---|
| M0 | Foundations | `pyproject`, config, logging, Docker Compose (pgvector + app), Alembic baseline, CI | config/unit tests; migration up/down on pgvector in CI | ⬜ |
| M1 | Model providers | LLM protocol, Ollama, Gemini, OpenAI-compatible; structured output with repair/retry; accounting; embedders; cross-encoder | unit tests with a fake HTTP server; smoke test with Qwen on the evaluation machine | ⬜ |
| M2 | Ingestion | PyMuPDF parser, sections, chunker with page/offset provenance, metadata extraction, dedup, persistence | fixture PDFs (open-access) → expected sections/pages; dedup cases; DB round-trip in CI | ⬜ |
| M3 | Discovery | OpenAlex, Semantic Scholar, Crossref, arXiv clients, normalization, import, open-access PDF download | recorded responses (unit); live smoke test on the evaluation machine | ⬜ |
| M4 | Hybrid retrieval | dense (pgvector), BM25 (bm25s), filters, RRF fusion, diversity, reranking | ranking unit tests; integration test on a fixture corpus with known relevant chunks | ⬜ |
| M5 | Agent core | state, schemas, nodes, deterministic policy, budgets, Postgres checkpointer, CLI runner | policy unit tests (every branch); graph tests with a scripted fake LLM (refine, stagnation, budget exhaustion, resume after interruption) | ⬜ |
| M6 | Grounded synthesis | report schema, claim types, citation verification, injection safeguards | fabricated-id removal, unsupported-claim handling, adversarial documents | ⬜ |
| M7 | Streamlit workspace | Dashboard, Projects, Library, Research Assistant (live plan & progress), Evidence Explorer (PDF page highlight), basic Comparison, Settings, save/resume | page smoke tests (`streamlit.testing`); manual run on the evaluation machine | ⬜ |
| M8 | Phase 1 validation | end-to-end investigation with Qwen 3.5 9B on a small real corpus; README run instructions | run log + stored outputs from the evaluation machine | ⬜ |

## Phase 2 — Advanced research intelligence

Scientific entity/result/claim extraction · contradiction analysis · research gap discovery ·
benchmark (QASPER, SciFact, ResearchMind multi-paper set) · baselines and ablations · Langfuse.

## Phase 3 — Knowledge graph

Neo4j · graph construction from extracted entities and claims · graph retrieval strategy ·
cross-paper relationship discovery · graph exploration page · baseline 5 in the benchmark.

## Main technical risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen 3.5 9B produces invalid or truncated JSON | small schemas, Ollama JSON-schema `format`, `think=false`, explicit `num_predict`, repair + one retry, deterministic fallback per node |
| Latency of a 9B model on many LLM calls | rules instead of LLM where possible, batched grading, skip planning for simple lookups, strict call budget |
| Context window (16k) | top-k bounded, passages truncated by tokens, evidence summarized per subquestion |
| Weak LLM judge when the same small model is evaluated | retrieval metrics on human gold labels; a stronger judge (Gemini) when available, always reported |
| PDF layout (two columns, tables, equations, scans) | PyMuPDF reading order, section heuristics tested on varied fixtures, failures flagged |
| API rate limits (Semantic Scholar without key, OpenAlex polite pool) | rate limiter, retries with backoff, on-disk cache, optional API keys |
| Long-running agent inside Streamlit reruns | background thread + checkpoints; the page only reads state |
| Embedding model change | vectors tagged by model; mismatches refused; re-embed command |
| Docker on Windows cannot use the AMD GPU | LLM and embeddings in Ollama on the host; reranker on CPU in the container (small model) |
