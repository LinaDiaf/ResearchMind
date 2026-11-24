"""Identifier extraction and normalization (DOI, arXiv), title normalization."""

from __future__ import annotations

import re
import unicodedata

from researchmind.ingestion.pdf_parser import ParsedDocument

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", d, flags=re.IGNORECASE)
    d = d.rstrip(".,;)]}").lower()
    return d if d.startswith("10.") else None


def normalize_arxiv(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    a = arxiv_id.strip()
    a = re.sub(r"^(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)", "", a, flags=re.IGNORECASE)
    a = re.sub(r"\.pdf$", "", a)
    a = re.sub(r"v\d+$", "", a)
    return a or None


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title)
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_identifiers(doc: ParsedDocument, pages: int = 2) -> dict[str, str]:
    end = doc.page_offsets[pages] if len(doc.page_offsets) > pages else len(doc.text)
    head = doc.text[:end]
    out: dict[str, str] = {}
    if m := DOI_RE.search(head):
        if doi := normalize_doi(m.group(1)):
            out["doi"] = doi
    if m := ARXIV_RE.search(head):
        if aid := normalize_arxiv(m.group(1)):
            out["arxiv_id"] = aid
    return out


def guess_title(doc: ParsedDocument) -> str | None:
    """The largest text on page 1 (merged across consecutive lines), else PDF metadata."""
    first = [ln for ln in doc.lines if ln.page == 1]
    if first:
        top = max(ln.size for ln in first)
        if top > doc.body_font_size * 1.2:
            parts: list[str] = []
            for ln in first:
                if ln.size == top:
                    parts.append(ln.text)
                elif parts:
                    break
            title = " ".join(parts).strip()
            if 3 <= len(title) <= 300:
                return title
    meta_title = (doc.metadata.get("title") or "").strip()
    return meta_title or None
