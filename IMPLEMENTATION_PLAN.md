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

The column “Tests that close it” lists what was actually run.

| # | Milestone | Content | Tests that close it | Status |
|---|---|---|---|---|
| M0 | Foundations | `pyproject`, config, logging, Docker Compose (pgvector + app), Alembic baseline, CI | migration up/down on pgvector; CI (`tests`, `docker`) | ✅ |
| M1 | Model providers | LLM protocol, Ollama, Gemini, OpenAI-compatible; structured output with repair/retry; accounting; embedders; cross-encoder | mocked HTTP for Ollama/OpenAI-compatible, fake Gemini client, repair and budget tests; real-model CI job (`models`). Qwen smoke test on the evaluation machine: pending | ✅ |
| M2 | Ingestion | PyMuPDF parser, sections, chunker with page/offset provenance, metadata extraction, dedup, persistence | synthetic PDF corpus: sections, page/offset provenance, dedup, DB round-trip | ✅ |
| M3 | Discovery | OpenAlex, Semantic Scholar, Crossref, arXiv clients, normalization, import, open-access PDF download | parsers on hand-written responses in each API's documented format, merging, retries, cache. Live API test on the evaluation machine: pending | ✅ |
| M4 | Hybrid retrieval | dense (pgvector), BM25 (bm25s), filters, RRF fusion, diversity, reranking | lexical exact hit with page, filters, references excluded, diversity, exclusion, dense/hybrid; real cross-encoder and embedder in CI | ✅ |
| M5 | Agent core | state, schemas, nodes, deterministic policy, budgets, Postgres checkpointer, CLI runner | scripted-LLM graph tests: refine, conflict, stagnation, budget exhaustion, invalid outputs; crash and resume with the Postgres checkpointer | ✅ |
| M6 | Grounded synthesis | report schema, claim types, citation verification, injection safeguards | fabricated ids removed, unsupported claims removed, injection flagged and unable to create citations | ✅ |
| M7 | Streamlit workspace | Dashboard, Projects, Library, Research Assistant (live plan & progress), Evidence Explorer (PDF page highlight), basic Comparison, Settings, save/resume | every page rendered with `streamlit.testing` on the real database, report and evidence pages, comparison citation constraints, page highlighting, CLI ingestion | ✅ |
| M8 | Phase 1 validation | end-to-end investigation with Qwen 3.5 9B on a small real corpus; README run instructions | end-to-end run with Qwen 3.5 9B on the evaluation machine: pending | 🟨 |

## Phase 2 — Advanced research intelligence

| Item | Status |
|---|---|
| QASPER benchmark runner: systems vector, hybrid, hybrid + rerank, agentic, agentic single-round, agentic without critique; retrieval, answer, citation, unanswerable-detection and cost metrics; resumable runs; report with bootstrap CIs | 🟨 implemented and tested on a fixture with a scripted model; real runs pending |
| SciFact (claim verification) and a multi-paper ResearchMind set (human-validated gold) | ⬜ |
| LLM-as-judge faithfulness with a stronger judge model | ⬜ |
| Scientific entity / result / claim extraction | ⬜ |
| Contradiction analysis and research gap discovery | ⬜ |
| Langfuse tracing | ⬜ |

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
