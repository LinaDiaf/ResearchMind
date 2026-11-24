"""Source-independent paper metadata record."""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchmind.ingestion.metadata import normalize_arxiv, normalize_doi, normalize_title


class PaperRecord(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    is_open_access: bool | None = None
    citation_count: int | None = None
    source: str

    def model_post_init(self, __context) -> None:
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = normalize_arxiv(self.arxiv_id)

    @property
    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        return f"title:{normalize_title(self.title)}:{self.year}"

    def merge(self, other: PaperRecord) -> PaperRecord:
        """Fill missing fields from another record of the same work."""
        data = self.model_dump()
        for k, v in other.model_dump().items():
            if data.get(k) in (None, [], "") and v not in (None, [], ""):
                data[k] = v
        if other.citation_count and (self.citation_count or 0) < other.citation_count:
            data["citation_count"] = other.citation_count
        data["source"] = self.source if self.source == other.source else f"{self.source}+{other.source}"
        return PaperRecord(**data)
