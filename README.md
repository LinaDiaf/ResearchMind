# ResearchMind

**An agentic RAG system for scientific literature intelligence.**

ResearchMind answers research questions over a library of scientific publications. A bounded
LangGraph agent analyses the question, breaks it into subquestions, chooses a retrieval strategy
for each one, grades what it retrieves, searches again where coverage is missing, critiques the
evidence, and writes a report in which every statement is typed and linked to a passage and page
of an indexed paper. When the evidence is not sufficient, the report says so.

[![CI](https://github.com/LinaDiaf/ResearchMind/actions/workflows/ci.yml/badge.svg)](https://github.com/LinaDiaf/ResearchMind/actions/workflows/ci.yml)

## What makes it agentic

```
analyze → plan → select_next ─┬─► route → retrieve → evaluate → decide ─► select_next
                              ├─► critique ─► select_next
                              └─► synthesize → verify → END
```

- **Planning**: intent detection (lookup, comparison, verification, contradiction, gap
  exploration…) and decomposition into subquestions, each with the *aspects* an answer must
  cover and the paper sections most likely to contain them.
- **Adaptive retrieval**: per subquestion and per round, the agent chooses queries, dense/BM25
  weights (exact technical vocabulary pushes towards BM25), section filters and depth; it
  excludes passages it has already seen and escalates its strategy when a round brings nothing.
- **Evidence evaluation**: every passage is graded for relevance and stance; coverage of the
  required aspects decides whether to stop, refine, investigate a conflict, or declare the
  evidence insufficient.
- **Bounded**: limits on subquestions, rounds, follow-ups, LLM calls (with a reserve for the
  report) and time. Stopping rules are deterministic, tested code; the LLM proposes, code decides.
- **Resumable**: state is checkpointed in PostgreSQL after each step; an interrupted
  investigation continues where it stopped.

## Evidence guarantees

- Citations can only point to passages retrieved during the investigation; unknown ids are
  removed, and a claim presented as a paper's finding without a valid citation is dropped.
- Each claim is checked against its cited passages (supported / partially / unsupported);
  unsupported claims are removed and listed separately for transparency.
- Claims are typed: **reported finding**, **author interpretation**, **AI synthesis**,
  **hypothesis**.
- Titles, years, DOIs and pages come from the database, never from model output. The Evidence
  Explorer shows the original PDF page with the passage highlighted.
- Retrieved text is treated as untrusted: wrapped and escaped in prompts, instruction-like
  passages flagged, and model output never triggers actions.

## Features (Phase 1)

| Area | Implemented |
|---|---|
| Library | PDF upload (PyMuPDF), section detection and typing, chunks with page and character offsets, DOI/arXiv extraction, deduplication (DOI, arXiv id, file hash, fuzzy title) |
| Discovery | OpenAlex, Semantic Scholar, Crossref and arXiv search with merging, rate limits, caching; import with open-access PDF download |
| Retrieval | pgvector dense search, BM25 (bm25s), metadata/section filters, weighted RRF, per-paper diversity, cross-encoder reranking |
| Agent | LangGraph workflow above, typed decisions, deterministic fallbacks when the model's output is invalid |
| Workspace | projects, saved and resumable investigations, execution trace, notes and open questions, evidence reuse |
| Comparison | paper × dimension tables where each cell cites passages of that paper only |
| Evaluation | QASPER benchmark runner: vector, hybrid, hybrid + rerank, agentic and two ablations; Recall@K, MRR, nDCG, answer F1, citation precision, unanswerable detection, cost |

Planned: scientific claim/result extraction, contradiction and gap analysis, Langfuse (Phase 2);
Neo4j knowledge graph and graph retrieval (Phase 3). See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Quick start (Docker)

Prerequisites: Docker, and [Ollama](https://ollama.com) on the host with the two models:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Then:

```bash
git clone https://github.com/LinaDiaf/ResearchMind.git
cd ResearchMind
cp .env.example .env            # edit CONTACT_EMAIL; defaults use Ollama on the host
cd docker
docker compose up -d --build    # PostgreSQL + pgvector, migrations, Streamlit
docker compose exec app researchmind check
```

Open <http://localhost:8501>. The app container reaches Ollama on the host through
`host.docker.internal`; the cross-encoder runs on CPU inside the container (downloaded on first
use).

### Using Gemini instead of a local model

Create a key at <https://aistudio.google.com/apikey>, then in `.env`:

```ini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash-lite   # any model id from https://ai.google.dev/gemini-api/docs/models
GEMINI_API_KEY=your-key
```

Restart with `docker compose up -d`. Never commit `.env`.

## Command line

```bash
researchmind ingest "My project" path/to/pdfs/
researchmind search "retrieval augmented generation evaluation" --limit 5 --import-to "My project" --pick 1 3
researchmind ask "My project" "How do sparse and dense retrievers compare out of domain?" --out results/q1
researchmind resume <investigation-id>
```

Inside Docker, prefix with `docker compose exec app`.

## Evaluation

```bash
docker compose exec app python -m evaluation.run evaluation/configs/smoke.json
docker compose exec app python -m evaluation.report results/eval/qasper_smoke
```

The runner samples QASPER papers (questions with human-annotated evidence paragraphs), ingests
them, runs every configured system on the same questions and stores raw outputs per question in
`results/eval/<series>/runs/`; it is resumable. `report` writes `summary.md` with bootstrap
confidence intervals. **No benchmark results are published yet**: they will be added from real
runs, with their configuration.

## Development

```bash
pip install -e ".[dev,ml]"
export DATABASE_URL_TEST=postgresql+psycopg://user:pass@localhost:5432/researchmind_test  # pgvector
pytest                       # unit + integration tests
RUN_MODEL_TESTS=1 pytest -m models   # small real embedding and cross-encoder models
```

CI runs lint, the unit and integration tests against a PostgreSQL + pgvector service, the
real-model tests and the Docker build on every pull request.

## Project layout

```
app/                     Streamlit UI (views/, components/)
src/researchmind/
  agent/                 LangGraph workflow, policy, schemas, prompts, safety
  retrieval/             dense, BM25, fusion, reranking, engine
  ingestion/             PDF parsing, sections, chunking, dedup, library service
  discovery/             OpenAlex, Semantic Scholar, Crossref, arXiv
  llm/ embeddings/       provider-independent model layer
  synthesis/             report validation and rendering
  comparison/ workspace/ comparison tables, investigations, notes
migrations/              Alembic
evaluation/              QASPER benchmark, systems, metrics, reports
tests/                   unit/, integration/, fixtures/ (synthetic test papers)
```

Design details: [ARCHITECTURE.md](ARCHITECTURE.md).

## License

[MIT](LICENSE)
