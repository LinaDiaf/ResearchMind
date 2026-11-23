# ResearchMind — Architecture

> Status: **design proposal** (Phase 1 not implemented yet). This document describes the target
> architecture. Sections are marked when they belong to Phase 2 or Phase 3.

ResearchMind is an agentic RAG system for scientific literature. Its core is a bounded LangGraph
workflow that plans an investigation, chooses retrieval strategies, evaluates the evidence it
finds, retrieves again when coverage is insufficient, and writes a report in which every
statement is linked to a passage and a page of an indexed publication.

---

## 1. Design principles

1. **Evidence first.** The language model never cites from memory. A citation can only point to
   an evidence item that the retrieval engine actually returned during the investigation. Citation
   identifiers that do not exist in the evidence set are removed deterministically before a report
   is shown.
2. **The LLM decides, code acts.** The model produces typed decisions (plans, queries, gradings).
   Tools (database, retrieval, external APIs) are only ever invoked by deterministic code that
   validates those decisions. Text inside retrieved documents can therefore not trigger actions.
3. **Bounded autonomy.** Every loop has an explicit budget: subquestions, retrieval rounds,
   LLM calls, tokens and wall-clock time. When a budget is exhausted, the agent reports what is
   missing instead of continuing.
4. **Honest uncertainty.** "Insufficient evidence" is a first-class outcome, not an error.
5. **Provider independence.** LLM, embedding model and reranker are configured, not hard-coded.
   The first target models are Gemini (API) and Qwen 3.5 9B (local, through Ollama).
6. **Small, typed components.** Pydantic schemas for every LLM output and every state field;
   deterministic components wherever a rule is sufficient.

---

## 2. System overview

```
┌──────────────────────────────── Streamlit application ────────────────────────────────┐
│ Dashboard · Projects · Library · Research Assistant · Comparison · Evidence · Settings │
└───────────────┬──────────────────────────────┬──────────────────────────┬─────────────┘
                │                              │                          │
        ┌───────▼────────┐            ┌────────▼─────────┐        ┌───────▼────────┐
        │   Ingestion    │            │  Agent (LangGraph)│        │   Workspace    │
        │ PDF · metadata │            │  plan → retrieve  │        │ projects, notes│
        │ dedup · chunks │            │  → evaluate → …   │        │ investigations │
        └───────┬────────┘            └────────┬─────────┘        └───────┬────────┘
                │                              │                          │
        ┌───────▼────────┐            ┌────────▼─────────┐                │
        │   Discovery    │            │ Retrieval engine │                │
        │ OpenAlex · S2  │            │ dense · BM25 ·   │                │
        │ Crossref·arXiv │            │ filters · rerank │                │
        └───────┬────────┘            └────────┬─────────┘                │
                │                              │                          │
        ┌───────▼──────────────────────────────▼──────────────────────────▼────────┐
        │ PostgreSQL 16 + pgvector  ·  BM25 index files  ·  PDF store  ·  LangGraph  │
        │                                                     checkpoints (Postgres) │
        └────────────────────────────────────────────────────────────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Model providers   │  Gemini API · Ollama (Qwen 3.5 9B,
                         │ (LLM, embeddings, │  nomic-embed-text) · sentence-transformers
                         │  cross-encoder)   │  (embeddings, cross-encoder)
                         └───────────────────┘
```

Everything runs in one Python application (Streamlit) plus PostgreSQL, orchestrated with Docker
Compose. There is no separate API server.

---

## 3. Directory structure

```
researchmind/
├── app/                          # Streamlit UI only (no business logic)
│   ├── Home.py                   # Dashboard
│   ├── pages/
│   │   ├── 1_Projects.py
│   │   ├── 2_Library.py          # upload PDFs, discover & import papers
│   │   ├── 3_Research_Assistant.py
│   │   ├── 4_Paper_Comparison.py
│   │   ├── 5_Evidence_Explorer.py
│   │   └── 9_Settings.py
│   └── components/               # shared widgets (citation card, plan view, PDF page viewer)
├── src/researchmind/
│   ├── config.py                 # pydantic-settings, all configuration from env
│   ├── logging.py                # structlog JSON logging, run/investigation ids in context
│   ├── db/
│   │   ├── models.py             # SQLAlchemy 2 ORM models
│   │   ├── session.py
│   │   └── repositories/         # papers, chunks, projects, investigations, evidence
│   ├── llm/
│   │   ├── base.py               # LLMProvider protocol: generate(), generate_structured()
│   │   ├── gemini.py             # google-genai
│   │   ├── ollama.py             # native /api/chat, JSON-schema format, think=false
│   │   ├── openai_compat.py      # vLLM, LM Studio, OpenRouter …
│   │   ├── structured.py         # schema → JSON, parse, repair, bounded retry
│   │   └── accounting.py         # tokens, latency, cost per call
│   ├── embeddings/               # Embedder protocol: sentence-transformers, Ollama
│   ├── ingestion/
│   │   ├── pdf_parser.py         # PyMuPDF: text blocks, fonts, pages, char offsets
│   │   ├── sections.py           # heading detection, section typing
│   │   ├── chunker.py            # section-aware chunks with page provenance
│   │   ├── metadata.py           # DOI / arXiv id extraction, enrichment
│   │   └── dedup.py
│   ├── discovery/
│   │   ├── openalex.py · semantic_scholar.py · crossref.py · arxiv.py
│   │   ├── normalize.py          # common PaperRecord schema
│   │   └── http.py               # rate limiting, retries, on-disk cache
│   ├── retrieval/
│   │   ├── dense.py              # pgvector cosine search
│   │   ├── lexical.py            # BM25 (bm25s), one index per project
│   │   ├── filters.py            # year, venue, paper ids, section kinds
│   │   ├── fusion.py             # weighted reciprocal rank fusion
│   │   ├── reranker.py           # cross-encoder
│   │   └── engine.py             # RetrievalRequest → RetrievalResult
│   ├── agent/
│   │   ├── state.py              # ResearchState (typed)
│   │   ├── schemas.py            # LLM decision schemas
│   │   ├── nodes/                # one module per node
│   │   ├── policy.py             # deterministic routing & stopping rules
│   │   ├── budget.py
│   │   ├── safety.py             # untrusted-content wrapping, injection heuristics
│   │   ├── prompts/              # versioned prompt templates
│   │   └── graph.py              # StateGraph assembly + Postgres checkpointer
│   ├── synthesis/
│   │   ├── report.py
│   │   └── citations.py          # citation validation and claim support checks
│   ├── comparison/               # cross-paper comparison tables
│   └── workspace/                # projects, investigations, notes, resume
├── migrations/                   # Alembic
├── evaluation/                   # benchmark, baselines, metrics, ablations (Phase 2)
├── tests/
│   ├── unit/                     # no network, no model downloads
│   ├── integration/              # PostgreSQL + pgvector, real models
│   └── fixtures/                 # small open-access PDFs, recorded API responses
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/ci.yml      # lint, type check, unit + integration tests
├── pyproject.toml
├── .env.example
├── README.md · ARCHITECTURE.md · IMPLEMENTATION_PLAN.md
└── LICENSE
```

---

## 4. Database schema (Phase 1)

PostgreSQL 16 with the `vector` extension. Primary keys are UUIDs. Migrations are managed with
Alembic.

```
projects            id, name, description, created_at, updated_at

papers              id, title, title_norm, abstract, year, venue, authors (jsonb),
                    doi (unique, normalized, nullable), arxiv_id (unique, nullable),
                    openalex_id, s2_id, url, license, source (upload|openalex|s2|crossref|arxiv),
                    pdf_path, pdf_sha256 (unique, nullable),
                    status (metadata_only|parsing|ingested|failed), parse_warnings (jsonb),
                    created_at

project_papers      project_id → projects, paper_id → papers, added_at   PK(project_id, paper_id)

sections            id, paper_id, ord, title, level,
                    kind (abstract|introduction|related_work|method|experiments|results|
                          discussion|limitations|conclusion|references|appendix|other),
                    page_start, page_end

chunks              id, paper_id, section_id, ord, text, token_count,
                    page_start, page_end, char_start, char_end,   -- offsets in page text
                    content_hash, is_reference_list (bool)
                    INDEX (paper_id, ord)

chunk_embeddings    chunk_id → chunks, model (text), embedding vector(D)
                    PK(chunk_id, model), HNSW index (vector_cosine_ops)

investigations      id, project_id, question, status (running|interrupted|completed|failed),
                    thread_id (LangGraph), config (jsonb: models, budgets, strategy flags),
                    report (jsonb), created_at, updated_at

investigation_events id, investigation_id, seq, node, event_type, payload (jsonb),
                    latency_ms, created_at         -- human-readable execution trace

evidence_items      id, investigation_id, subquestion_id, chunk_id → chunks,
                    retrieval_round, strategy, scores (jsonb: dense, bm25, fused, rerank),
                    relevance (high|partial|irrelevant), stance (supports|contradicts|neutral),
                    rationale, injection_flag (bool)

report_claims       id, investigation_id, ord, text,
                    claim_type (reported_finding|author_interpretation|ai_synthesis|hypothesis),
                    support (supported|partially_supported|unsupported)

claim_citations     claim_id → report_claims, evidence_id → evidence_items   PK(both)

notes               id, project_id, investigation_id (nullable), kind (note|open_question),
                    body, created_at

llm_calls           id, investigation_id (nullable), node, provider, model,
                    prompt_tokens, completion_tokens, latency_ms, cost_usd, status, created_at
```

LangGraph checkpoint tables are created and managed by `langgraph-checkpoint-postgres`
(`PostgresSaver.setup()`), in the same database.

**Embedding dimension.** HNSW indexes need a fixed dimension, so `D` is fixed when the database is
initialised (`EMBEDDING_DIM`). The `model` column records which model produced each vector; the
retrieval engine refuses to query with a model whose vectors are not present, so changing the
embedding model means re-embedding, never silently mixing vector spaces.

**Phase 2 tables** (not created in Phase 1): `scientific_entities` (method, dataset, metric,
task), `experimental_results` (paper, method, dataset, metric, value, unit, setting, chunk_id),
`atomic_claims` (paper-level claims linked to chunks), `claim_relations` (supports/contradicts).

### Provenance guarantee

A citation shown to the user resolves as
`claim → evidence_item → chunk → (paper, page_start..page_end, char_start..char_end)`.
The Evidence Explorer uses the page number and character offsets to render the original PDF
page with the passage highlighted (PyMuPDF), so the user can check the source directly.

### Deduplication

Applied on every import, in this order:
1. normalized DOI (lower-case, prefix `https://doi.org/` removed);
2. arXiv identifier without version suffix;
3. SHA-256 of the PDF file;
4. normalized title (casefold, punctuation removed) with fuzzy similarity ≥ 0.93
   (rapidfuzz), same year ±1, and the same first-author surname.

A preprint and its published version are linked (same record, both identifiers kept) instead of
being indexed twice.

---

## 5. Ingestion pipeline

1. **Parse** with PyMuPDF (`get_text("dict")`, reading order sorted): text spans with font size,
   font flags and position, per page.
2. **Detect sections**: headings are spans whose font size or weight stands out from the page's
   body font, or that match numbered-heading patterns (`3.2 Results`). Each section is typed
   (`results`, `limitations` …) by keyword rules, so retrieval can target section kinds.
3. **Detect references** and appendices; reference lists are stored but excluded from retrieval
   by default.
4. **Chunk** inside sections: ~350 tokens, 60-token overlap, sentence boundaries respected; each
   chunk keeps `page_start/page_end` and character offsets.
5. **Metadata**: DOI and arXiv identifiers are extracted from the first pages, then enriched
   through Crossref / OpenAlex when the network is available.
6. **Index**: embeddings written to `chunk_embeddings`; the project's BM25 index is rebuilt.

Known limits (documented, not hidden): scanned PDFs without a text layer (no OCR in Phase 1,
flagged as `failed` with a reason), equations and tables extracted as plain text, two-column
layouts depend on PyMuPDF's reading order.

---

## 6. Hybrid retrieval

```
RetrievalRequest(query(s), strategy, filters, k, section_kinds, paper_ids)
        │
        ├─ dense:   embed query → pgvector cosine top-N  (metadata filters in SQL)
        ├─ lexical: BM25 top-N over the project index    (same filters applied)
        │
        ├─ fusion:  weighted reciprocal rank fusion  score = Σ w_s / (60 + rank_s)
        ├─ dedup:   near-identical chunks merged, max M chunks per paper (diversity)
        └─ rerank:  cross-encoder on top-R (default R = 40) → top-k
```

- **Strategies**: `dense`, `lexical`, `hybrid` (with weights), each with optional filters.
  The weights are a decision of the retrieval router, not constants.
- **BM25**: `bm25s`, one index per project, persisted on disk and rebuilt after ingestion.
  It is a derived artefact: it can always be rebuilt from the `chunks` table.
- **Reranker**: a cross-encoder (default `BAAI/bge-reranker-base`, configurable; `none` allowed
  for ablations).
- **Graph retrieval**: Phase 3, as an additional strategy behind the same interface.

---

## 7. Agent workflow (LangGraph)

### 7.1 Graph

```
            ┌──────────────┐
  question ─►analyze_query │  intent, entities, constraints, complexity
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ plan_research│  subquestions + what evidence would answer each
            └──────┬───────┘     (single subquestion for simple lookups: no LLM call)
                   ▼
            ┌──────────────┐  deterministic: choose the open subquestion with the
     ┌─────►│ select_next  │  highest priority, or go to critique when none is left
     │      └──────┬───────┘  or the budget is exhausted
     │             ▼
     │      ┌──────────────┐
     │      │route_retrieval│ strategy, query rewrites, filters, section kinds, k
     │      └──────┬───────┘  (rules + LLM; uses feedback from previous rounds)
     │             ▼
     │      ┌──────────────┐
     │      │  retrieve    │  hybrid engine (deterministic tool)
     │      │  + rerank    │
     │      └──────┬───────┘
     │             ▼
     │      ┌──────────────┐
     │      │evaluate_evid.│  per-passage relevance & stance (batched LLM grading)
     │      │              │  + coverage of the subquestion's required aspects
     │      └──────┬───────┘
     │             ▼
     │      ┌──────────────┐   sufficient ─────────────► mark answered ──┐
     │      │   decide     │   insufficient & budget & progress ─► refine ┤ (back to
     │      │ (policy.py)  │   stagnation or budget out ─► mark          │  route_retrieval
     │      └──────────────┘        "insufficient evidence"             │  for this subq)
     └────────────────────────────────────────────────────────────────────┘
                   │ all subquestions closed
                   ▼
            ┌──────────────┐  contradictions, unsupported needs, missing comparisons;
            │   critique   │  may add ≤ 2 follow-up subquestions (one critique round)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  synthesize  │  report built from evidence ids only; claim types labelled
            └──────┬───────┘
                   ▼
            ┌──────────────┐  deterministic: unknown ids removed; LLM support check of
            │verify_report │  each claim against its cited passages; unsupported claims
            └──────┬───────┘  flagged or dropped
                   ▼
                persist → END
```

The graph is compiled with a `PostgresSaver` checkpointer; `thread_id` = investigation id.
An investigation interrupted at any node (crash, closed browser, stopped container) resumes from
its last checkpoint.

### 7.2 State

```python
class ResearchState(TypedDict):
    investigation_id: str
    project_id: str
    question: str
    analysis: QueryAnalysis | None          # intent, entities, constraints, complexity
    plan: list[SubQuestion]                 # id, text, rationale, required_aspects,
                                            # priority, status (open|answered|insufficient),
                                            # rounds_used
    current_subq: str | None
    retrieval_log: Annotated[list[RetrievalAction], add]   # every query actually run
    evidence: dict[str, list[EvidenceItem]] # by subquestion id, deduplicated
    assessments: dict[str, CoverageAssessment]  # covered/missing aspects, conflicts, score
    critique: Critique | None
    report: Report | None                   # sections → claims → evidence ids
    budget: Budget                          # limits and counters
    events: Annotated[list[TraceEvent], add]   # shown live in the UI
    errors: Annotated[list[str], add]
```

`QueryAnalysis.intent` ∈ `lookup`, `single_paper`, `multi_paper_synthesis`, `method_comparison`,
`evidence_verification`, `contradiction_analysis`, `gap_exploration`.

### 7.3 Bounds (defaults, configurable)

| Limit | Default |
|---|---|
| subquestions in the plan | 5 |
| retrieval rounds per subquestion | 3 |
| critique rounds / follow-up subquestions | 1 / 2 |
| LLM calls per investigation | 40 |
| tokens per investigation | provider-dependent |
| wall-clock per investigation | 15 min |
| LangGraph `recursion_limit` | derived from the limits above |

The policy checks the budget before every LLM call. Budget exhaustion produces a report with the
evidence gathered so far and an explicit list of unanswered subquestions.

---

## 8. Adaptive retrieval and evidence evaluation

This is what makes the workflow agentic rather than a fixed pipeline.

**Initial routing** combines rules and an LLM decision:
- exact terms, acronyms, dataset names, numbers → lexical weight increased;
- conceptual or paraphrased questions → dense weight increased;
- intent-driven section targeting: `method_comparison` → method/experiments/results,
  `gap_exploration` → limitations/future work/conclusion, `evidence_verification` → results;
- constraints from the analysis (years, named papers) → metadata filters;
- the LLM writes 1–3 query reformulations, scientific vocabulary expanded.

**Evaluation** after each round:
- each new passage is graded `high | partial | irrelevant` and labelled
  `supports | contradicts | neutral` with respect to the subquestion (batched, one LLM call);
- coverage is computed against the subquestion's `required_aspects`
  (e.g. *dataset*, *metric value*, *experimental setting*), which the planner defined;
- deterministic signals complete the LLM judgment: rerank score distribution, number of distinct
  papers, number of *new* relevant passages compared with the previous round (progress).

**Decision** (`policy.py`, deterministic, unit-tested):

| Situation | Action |
|---|---|
| all required aspects covered, ≥ 1 high-relevance passage | subquestion answered |
| aspects missing, budget left, last round brought new relevant evidence | refine: new queries aimed at the missing aspects, strategy escalation |
| conflicting evidence | targeted retrieval for experimental conditions of both sides |
| two rounds without new relevant evidence, or rounds exhausted | mark **insufficient evidence**, with the missing aspects |

**Strategy escalation** on refinement: hybrid → lexical with extracted key terms → widened
filters / larger k / other section kinds. Phase 1 can also propose papers to import from the
discovery APIs when the library itself lacks evidence; importing requires the user's approval
(LangGraph `interrupt`), it is never automatic.

---

## 9. Grounded synthesis and citation verification

- The synthesis prompt receives evidence items as `[E12]`-tagged passages and must cite those
  tags only. The output schema is `Report → sections → claims(text, claim_type, evidence_ids)`.
- `claim_type` distinguishes **reported finding** (stated by the paper's results),
  **author interpretation** (the authors' discussion), **AI synthesis** (ResearchMind's own
  combination of several sources) and **hypothesis** (a proposed direction).
- `verify_report`:
  1. removes evidence ids that are not in the investigation's evidence set (deterministic);
  2. checks each claim against its cited passages (LLM, batched) → supported / partially /
     unsupported;
  3. unsupported claims are dropped or kept with a visible warning, depending on settings;
  4. claims with no citation are only allowed as `ai_synthesis` or `hypothesis`, and are labelled.
- DOIs, page numbers and titles in the report come from the database, never from model output.

---

## 10. Prompt-injection protection

Retrieved documents are untrusted input.
- Passages are wrapped in delimiters with an id and an explicit "data, not instructions" system
  rule; delimiter-like sequences inside passages are escaped.
- A heuristic detector flags passages that contain instruction-like patterns (e.g. "ignore
  previous instructions", role markers); flagged passages are shown in the UI and down-weighted.
- LLM outputs are constrained to schemas; free text from a model is never executed, and tool
  calls are issued by code, so injected text cannot trigger imports, deletions or external calls.
- Tests include adversarial fixture documents.

---

## 11. Model providers

| Role | Default | Alternatives |
|---|---|---|
| LLM | `qwen3.5:9b` via Ollama (local evaluation machine), Gemini via API | any OpenAI-compatible server |
| Embeddings | `nomic-embed-text` via Ollama, or `BAAI/bge-base-en-v1.5` via sentence-transformers | configurable |
| Reranker | `BAAI/bge-reranker-base` (sentence-transformers CrossEncoder) | `cross-encoder/ms-marco-MiniLM-L-6-v2`, `none` |

Qwen 3.5 9B is called with thinking disabled, a context window of 16 384 tokens and
JSON-schema-constrained output, following what worked in previous projects with this model.
Small local models make structured output the main reliability risk; mitigations are listed in
`IMPLEMENTATION_PLAN.md`.

---

## 12. Streamlit application

- The agent runs in a background thread per investigation; the page streams `events` from the
  graph (`graph.stream`) to show the plan, the current subquestion, each retrieval round and its
  assessment. Because state is checkpointed, closing the page does not lose the investigation.
- Business logic lives in `src/researchmind`; pages only call services.

---

## 13. Observability and operations

- structlog JSON logs with `investigation_id`, `node` and `llm_call_id`.
- `llm_calls` and `investigation_events` tables provide latency, token and cost accounting for
  evaluation; Langfuse tracing is added in Phase 2.
- Docker Compose services: `db` (pgvector/pgvector:pg16), `app` (Streamlit); the LLM server
  (Ollama) runs on the host and is reached through `host.docker.internal`.
- Secrets only through environment variables (`.env`, never committed).

---

## 14. Evaluation (Phase 2, designed now)

Systems compared, all on the same corpus and models:
1. vector RAG; 2. hybrid RAG; 3. hybrid + reranking; 4. agentic hybrid; 5. agentic + graph (Phase 3).

Data:
- **QASPER** (questions on NLP papers with human-annotated evidence paragraphs) for single-paper
  retrieval and answer quality;
- **SciFact** (scientific claims with annotated supporting/refuting abstracts) for evidence
  verification;
- a **ResearchMind multi-paper set** written for this project (synthesis, comparison,
  contradiction, gap questions) on open-access arXiv papers; gold evidence is annotated and
  checked by a human before use.

Metrics: Recall@K, MRR, nDCG for retrieval; answer correctness and faithfulness; citation
precision and coverage; insufficiency detection (questions deliberately unanswerable from the
corpus); task success; retrieval rounds; latency; tokens and cost. Ablations switch off one
component at a time (reranker, router, evaluator loop, critic). No result is reported unless it
was produced by a run whose configuration and raw outputs are stored with it.
