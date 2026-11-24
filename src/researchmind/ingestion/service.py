"""Library service: projects, PDF ingestion, imports from discovery, indexing."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from researchmind.config import Settings
from researchmind.db.models import (
    Chunk,
    ChunkEmbedding,
    Paper,
    PaperStatus,
    Project,
    ProjectPaper,
    Section,
    SectionKind,
)
from researchmind.discovery.http import DiscoveryError, RateLimitedClient
from researchmind.discovery.models import PaperRecord
from researchmind.embeddings import Embedder
from researchmind.ingestion.chunker import chunk_document
from researchmind.ingestion.dedup import Candidate, find_duplicate
from researchmind.ingestion.metadata import (
    extract_identifiers,
    guess_title,
    normalize_title,
)
from researchmind.ingestion.pdf_parser import PDFParseError, join_lines, parse_pdf
from researchmind.ingestion.sections import detect_sections
from researchmind.log import get_logger
from researchmind.retrieval.lexical import rebuild_index

log = get_logger(__name__)


@dataclass
class IngestResult:
    paper_id: uuid.UUID | None
    status: str  # ingested | duplicate | metadata_only | failed
    title: str = ""
    duplicate_rule: str | None = None
    n_chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Library:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        settings: Settings,
        http: RateLimitedClient | None = None,
    ) -> None:
        self.sf = session_factory
        self.embedder = embedder
        self.settings = settings
        self.http = http
        settings.ensure_dirs()

    # ----------------------------------------------------------------- projects
    def create_project(self, name: str, description: str = "") -> uuid.UUID:
        with self.sf.begin() as s:
            if existing := s.scalar(select(Project).where(Project.name == name)):
                return existing.id
            p = Project(name=name.strip(), description=description)
            s.add(p)
            s.flush()
            return p.id

    def list_projects(self) -> list[Project]:
        with self.sf() as s:
            return list(s.scalars(select(Project).order_by(Project.created_at)))

    def project_papers(self, project_id: uuid.UUID) -> list[Paper]:
        with self.sf() as s:
            return list(s.scalars(
                select(Paper).join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                .where(ProjectPaper.project_id == project_id).order_by(Paper.year.desc().nulls_last(),
                                                                      Paper.title)
            ))

    def _link(self, s: Session, project_id: uuid.UUID, paper_id: uuid.UUID) -> bool:
        if s.get(ProjectPaper, (project_id, paper_id)):
            return False
        s.add(ProjectPaper(project_id=project_id, paper_id=paper_id))
        return True

    def remove_from_project(self, project_id: uuid.UUID, paper_id: uuid.UUID) -> None:
        with self.sf.begin() as s:
            s.execute(delete(ProjectPaper).where(ProjectPaper.project_id == project_id,
                                                 ProjectPaper.paper_id == paper_id))
        self.reindex(project_id)

    # ---------------------------------------------------------------- ingestion
    def ingest_pdf(
        self,
        project_id: uuid.UUID,
        path: str | Path,
        record: PaperRecord | None = None,
        source: str = "upload",
        reindex: bool = True,
    ) -> IngestResult:
        path = Path(path)
        digest = sha256_file(path)
        try:
            doc = parse_pdf(path)
        except PDFParseError as exc:
            return IngestResult(None, "failed", title=path.name, error=str(exc))

        ids = extract_identifiers(doc)
        title = (record.title if record else None) or guess_title(doc) or path.stem
        sections = detect_sections(doc)
        first_author = record.authors[0] if record and record.authors else None
        cand = Candidate(
            title=title,
            year=record.year if record else None,
            first_author=first_author,
            doi=(record.doi if record else None) or ids.get("doi"),
            arxiv_id=(record.arxiv_id if record else None) or ids.get("arxiv_id"),
            pdf_sha256=digest,
        )

        with self.sf.begin() as s:
            existing, rule = find_duplicate(s, cand)
            if existing is not None and existing.status is PaperStatus.ingested:
                self._link(s, project_id, existing.id)
                result = IngestResult(existing.id, "duplicate", existing.title, rule)
                paper = None
            else:
                paper = existing or Paper(source=source, title=title,
                                          title_norm=normalize_title(title))
                if existing is None:
                    s.add(paper)
                self._apply_record(paper, record)
                paper.doi = paper.doi or cand.doi
                paper.arxiv_id = paper.arxiv_id or cand.arxiv_id
                if not paper.abstract:
                    abstract = next((join_lines(sec.lines) for sec in sections
                                     if sec.kind is SectionKind.abstract), None)
                    paper.abstract = abstract[:5000] if abstract else None
                dest = self.settings.pdf_dir / f"{digest}.pdf"
                if not dest.exists():
                    shutil.copyfile(path, dest)
                paper.pdf_path = str(dest)
                paper.pdf_sha256 = digest
                paper.page_count = doc.page_count
                paper.parse_warnings = list(doc.warnings)
                paper.status = PaperStatus.parsing
                s.flush()
                self._link(s, project_id, paper.id)
                result = IngestResult(paper.id, "ingested", paper.title, rule,
                                      warnings=list(doc.warnings))

        if paper is not None:
            try:
                result.n_chunks = self._index_paper(paper.id, doc, sections)
            except Exception as exc:
                log.exception("ingest_failed", paper_id=str(paper.id))
                with self.sf.begin() as s:
                    p = s.get(Paper, paper.id)
                    p.status = PaperStatus.failed
                    p.parse_warnings = [*p.parse_warnings, f"indexing failed: {exc}"]
                result.status, result.error = "failed", str(exc)
        if reindex:
            self.reindex(project_id)
        return result

    def _apply_record(self, paper: Paper, record: PaperRecord | None) -> None:
        if record is None:
            return
        paper.title = record.title or paper.title
        paper.title_norm = normalize_title(paper.title)
        paper.authors = [{"name": a} for a in record.authors] or paper.authors or []
        paper.year = record.year or paper.year
        paper.venue = record.venue or paper.venue
        paper.abstract = record.abstract or paper.abstract
        paper.doi = paper.doi or record.doi
        paper.arxiv_id = paper.arxiv_id or record.arxiv_id
        paper.openalex_id = paper.openalex_id or record.openalex_id
        paper.s2_id = paper.s2_id or record.s2_id
        paper.url = paper.url or record.url
        paper.pdf_url = paper.pdf_url or record.pdf_url
        paper.license = paper.license or record.license

    def _index_paper(self, paper_id: uuid.UUID, doc, sections) -> int:
        drafts = chunk_document(doc, sections)
        with self.sf.begin() as s:
            s.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
            s.execute(delete(Section).where(Section.paper_id == paper_id))
            paper = s.get(Paper, paper_id)
            section_ids = []
            for i, sec in enumerate(sections):
                row = Section(paper_id=paper_id, ord=i, title=sec.title[:500], level=sec.level,
                              kind=sec.kind, page_start=sec.page_start, page_end=sec.page_end)
                s.add(row)
                section_ids.append(row)
            s.flush()
            rows = []
            for d in drafts:
                c = Chunk(paper_id=paper_id, section_id=section_ids[d.section_index].id, ord=d.ord,
                          text=d.text, token_count=d.token_count, section_title=d.section_title,
                          section_kind=d.section_kind, page_start=d.page_start,
                          page_end=d.page_end, char_start=d.char_start, char_end=d.char_end,
                          content_hash=d.content_hash, is_reference_list=d.is_reference_list)
                s.add(c)
                rows.append(c)
            s.flush()
            to_embed = [c for c in rows if not c.is_reference_list]
            texts = [f"{paper.title}. {c.section_title or ''}. {c.text}" for c in to_embed]
            vectors = self.embedder.embed_documents(texts) if texts else []
            for c, v in zip(to_embed, vectors, strict=True):
                s.add(ChunkEmbedding(chunk_id=c.id, model=self.embedder.model, embedding=v))
            paper.status = PaperStatus.ingested
        log.info("paper_indexed", paper_id=str(paper_id), chunks=len(drafts))
        return len(drafts)

    def import_record(self, project_id: uuid.UUID, record: PaperRecord,
                      download_pdf: bool = True) -> IngestResult:
        """Add a discovered paper. Open-access PDFs are downloaded and ingested; otherwise the
        paper is kept as metadata only (title and abstract are still searchable after reindex)."""
        if download_pdf and record.pdf_url and self.http is not None:
            try:
                data = self.http.get_bytes(record.pdf_url)
                if data[:5] == b"%PDF-":
                    tmp = self.settings.cache_dir / f"dl-{uuid.uuid4().hex}.pdf"
                    tmp.write_bytes(data)
                    try:
                        return self.ingest_pdf(project_id, tmp, record, source=record.source)
                    finally:
                        tmp.unlink(missing_ok=True)
                log.warning("pdf_url_not_pdf", url=record.pdf_url)
            except DiscoveryError as exc:
                log.warning("pdf_download_failed", url=record.pdf_url, error=str(exc))
        cand = Candidate(record.title, record.year, record.authors[0] if record.authors else None,
                         record.doi, record.arxiv_id)
        with self.sf.begin() as s:
            existing, rule = find_duplicate(s, cand)
            if existing is not None:
                self._apply_record(existing, record)
                self._link(s, project_id, existing.id)
                return IngestResult(existing.id, "duplicate", existing.title, rule)
            p = Paper(source=record.source, title=record.title,
                      title_norm=normalize_title(record.title), status=PaperStatus.metadata_only)
            self._apply_record(p, record)
            s.add(p)
            s.flush()
            self._link(s, project_id, p.id)
            pid = p.id
        self._index_abstract(pid)
        self.reindex(project_id)
        return IngestResult(pid, "metadata_only", record.title)

    def _index_abstract(self, paper_id: uuid.UUID) -> None:
        """Metadata-only papers are searchable through their abstract (page 0 = no PDF)."""
        with self.sf.begin() as s:
            p = s.get(Paper, paper_id)
            if not p.abstract:
                return
            c = Chunk(paper_id=p.id, ord=0, text=p.abstract, token_count=len(p.abstract.split()),
                      section_title="Abstract", section_kind=SectionKind.abstract, page_start=0,
                      page_end=0, char_start=0, char_end=len(p.abstract),
                      content_hash=hashlib.sha256(p.abstract.encode()).hexdigest())
            s.add(c)
            s.flush()
            v = self.embedder.embed_documents([f"{p.title}. Abstract. {p.abstract}"])[0]
            s.add(ChunkEmbedding(chunk_id=c.id, model=self.embedder.model, embedding=v))

    def reindex(self, project_id: uuid.UUID) -> int:
        with self.sf() as s:
            return rebuild_index(s, project_id, self.settings.index_dir)
