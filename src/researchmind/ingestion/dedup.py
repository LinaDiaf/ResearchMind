"""Duplicate detection across uploads and discovery sources."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from researchmind.db.models import Paper
from researchmind.ingestion.metadata import normalize_arxiv, normalize_doi, normalize_title

TITLE_THRESHOLD = 93.0


@dataclass
class Candidate:
    title: str
    year: int | None = None
    first_author: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_sha256: str | None = None


def _surname(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip()
    if "," in name:
        return name.split(",")[0].strip().casefold()
    return name.split()[-1].casefold() if name.split() else ""


def _first_author(paper: Paper) -> str | None:
    if paper.authors:
        a = paper.authors[0]
        return a.get("name") if isinstance(a, dict) else str(a)
    return None


def titles_match(a: Candidate, paper: Paper) -> bool:
    ta, tb = normalize_title(a.title), paper.title_norm
    if not ta or not tb or fuzz.ratio(ta, tb) < TITLE_THRESHOLD:
        return False
    if a.year and paper.year and abs(a.year - paper.year) > 1:
        return False
    sa, sb = _surname(a.first_author), _surname(_first_author(paper))
    return not (sa and sb and sa != sb)


def find_duplicate(session: Session, c: Candidate) -> tuple[Paper | None, str | None]:
    """Return (existing paper, matching rule) or (None, None)."""
    if doi := normalize_doi(c.doi):
        if p := session.scalar(select(Paper).where(Paper.doi == doi)):
            return p, "doi"
    if aid := normalize_arxiv(c.arxiv_id):
        if p := session.scalar(select(Paper).where(Paper.arxiv_id == aid)):
            return p, "arxiv_id"
    if c.pdf_sha256 and (p := session.scalar(select(Paper).where(Paper.pdf_sha256 == c.pdf_sha256))):
        return p, "pdf_sha256"
    norm = normalize_title(c.title) if c.title else ""
    if len(norm) >= 12:
        rows = session.execute(select(Paper.id, Paper.title_norm)).all()
        choices = {pid: t for pid, t in rows}
        for _, _, pid in process.extract(norm, choices, scorer=fuzz.ratio,
                                         score_cutoff=TITLE_THRESHOLD, limit=5):
            p = session.get(Paper, pid)
            if p is not None and titles_match(c, p):
                return p, "title"
    return None, None
