"""Clients for OpenAlex, Semantic Scholar, Crossref and arXiv."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from researchmind.discovery.http import DiscoveryError, RateLimitedClient
from researchmind.discovery.models import PaperRecord
from researchmind.log import get_logger

log = get_logger(__name__)


class Source(Protocol):
    name: str

    def search(self, query: str, limit: int = 10, year_min: int | None = None) -> list[PaperRecord]: ...


def _strip_tags(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip() or None


class OpenAlex:
    name = "openalex"
    URL = "https://api.openalex.org/works"

    def __init__(self, http: RateLimitedClient, email: str | None = None) -> None:
        self.http, self.email = http, email

    @staticmethod
    def _abstract(inv: dict | None) -> str | None:
        if not inv:
            return None
        pos: dict[int, str] = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        return " ".join(pos[i] for i in sorted(pos)) or None

    def parse(self, w: dict) -> PaperRecord | None:
        title = w.get("title") or w.get("display_name")
        if not title:
            return None
        arxiv = None
        for loc in w.get("locations") or []:
            url = (loc or {}).get("landing_page_url") or ""
            if m := re.search(r"arxiv\.org/abs/([\w.\-/]+)", url):
                arxiv = m.group(1)
        best = w.get("best_oa_location") or {}
        oa = w.get("open_access") or {}
        primary = w.get("primary_location") or {}
        venue = ((primary.get("source") or {}) or {}).get("display_name")
        return PaperRecord(
            title=title,
            authors=[(a.get("author") or {}).get("display_name", "")
                     for a in w.get("authorships") or [] if a.get("author")],
            year=w.get("publication_year"),
            venue=venue,
            abstract=self._abstract(w.get("abstract_inverted_index")),
            doi=w.get("doi"),
            arxiv_id=arxiv,
            openalex_id=(w.get("id") or "").rsplit("/", 1)[-1] or None,
            url=primary.get("landing_page_url") or w.get("doi"),
            pdf_url=best.get("pdf_url"),
            license=best.get("license"),
            is_open_access=oa.get("is_oa"),
            citation_count=w.get("cited_by_count"),
            source=self.name,
        )

    def search(self, query: str, limit: int = 10, year_min: int | None = None) -> list[PaperRecord]:
        params: dict = {"search": query, "per-page": min(limit, 50)}
        if year_min:
            params["filter"] = f"from_publication_date:{year_min}-01-01"
        if self.email:
            params["mailto"] = self.email
        data = self.http.get_json(self.URL, params)
        return [r for w in data.get("results", []) if (r := self.parse(w))]


class SemanticScholar:
    name = "semantic_scholar"
    URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,venue,authors,externalIds,openAccessPdf,url,citationCount,isOpenAccess"

    def __init__(self, http: RateLimitedClient, api_key: str | None = None) -> None:
        self.http, self.api_key = http, api_key

    def parse(self, p: dict) -> PaperRecord | None:
        if not p.get("title"):
            return None
        ext = p.get("externalIds") or {}
        pdf = p.get("openAccessPdf") or {}
        return PaperRecord(
            title=p["title"],
            authors=[a.get("name", "") for a in p.get("authors") or []],
            year=p.get("year"),
            venue=p.get("venue") or None,
            abstract=p.get("abstract"),
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            s2_id=p.get("paperId"),
            url=p.get("url"),
            pdf_url=pdf.get("url") or None,
            license=pdf.get("license"),
            is_open_access=p.get("isOpenAccess"),
            citation_count=p.get("citationCount"),
            source=self.name,
        )

    def search(self, query: str, limit: int = 10, year_min: int | None = None) -> list[PaperRecord]:
        params: dict = {"query": query, "limit": min(limit, 100), "fields": self.FIELDS}
        if year_min:
            params["year"] = f"{year_min}-"
        headers = {"x-api-key": self.api_key} if self.api_key else None
        data = self.http.get_json(self.URL, params, headers)
        return [r for p in data.get("data") or [] if (r := self.parse(p))]


class Crossref:
    name = "crossref"
    URL = "https://api.crossref.org/works"

    def __init__(self, http: RateLimitedClient, email: str | None = None) -> None:
        self.http, self.email = http, email

    def parse(self, it: dict) -> PaperRecord | None:
        titles = it.get("title") or []
        if not titles:
            return None
        year = None
        for key in ("published-print", "published-online", "issued", "created"):
            parts = ((it.get(key) or {}).get("date-parts") or [[None]])[0]
            if parts and parts[0]:
                year = int(parts[0])
                break
        authors = []
        for a in it.get("author") or []:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name")
            if name:
                authors.append(name)
        lic = (it.get("license") or [{}])[0].get("URL")
        pdf = next((ln.get("URL") for ln in it.get("link") or []
                    if ln.get("content-type") == "application/pdf"), None)
        return PaperRecord(
            title=titles[0],
            authors=authors,
            year=year,
            venue=(it.get("container-title") or [None])[0],
            abstract=_strip_tags(it.get("abstract")),
            doi=it.get("DOI"),
            url=it.get("URL"),
            pdf_url=pdf,
            license=lic,
            citation_count=it.get("is-referenced-by-count"),
            source=self.name,
        )

    def search(self, query: str, limit: int = 10, year_min: int | None = None) -> list[PaperRecord]:
        params: dict = {"query.bibliographic": query, "rows": min(limit, 50)}
        if year_min:
            params["filter"] = f"from-pub-date:{year_min}"
        if self.email:
            params["mailto"] = self.email
        data = self.http.get_json(self.URL, params)
        return [r for it in (data.get("message") or {}).get("items", []) if (r := self.parse(it))]

    def lookup_doi(self, doi: str) -> PaperRecord | None:
        data = self.http.get_json(f"{self.URL}/{doi}")
        return self.parse(data.get("message") or {})


class ArXiv:
    name = "arxiv"
    URL = "https://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, http: RateLimitedClient) -> None:
        self.http = http

    def parse_feed(self, xml_text: str) -> list[PaperRecord]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise DiscoveryError(f"invalid arXiv response: {exc}") from exc
        out = []
        for e in root.findall("a:entry", self.NS):
            title = re.sub(r"\s+", " ", e.findtext("a:title", "", self.NS)).strip()
            id_url = e.findtext("a:id", "", self.NS)
            if not title or "arxiv.org/abs/" not in id_url:
                continue
            aid = id_url.split("/abs/")[-1]
            pdf = next((ln.get("href") for ln in e.findall("a:link", self.NS)
                        if ln.get("title") == "pdf"), None)
            published = e.findtext("a:published", "", self.NS)
            out.append(PaperRecord(
                title=title,
                authors=[a.findtext("a:name", "", self.NS) for a in e.findall("a:author", self.NS)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue="arXiv",
                abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", self.NS)).strip() or None,
                doi=e.findtext("arxiv:doi", None, self.NS),
                arxiv_id=aid,
                url=f"https://arxiv.org/abs/{aid}",
                pdf_url=(pdf.replace("http://", "https://") if pdf else None),
                is_open_access=True,
                source=self.name,
            ))
        return out

    def search(self, query: str, limit: int = 10, year_min: int | None = None) -> list[PaperRecord]:
        params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 50)}
        records = self.parse_feed(self.http.get_text(self.URL, params))
        return [r for r in records if not year_min or (r.year or 0) >= year_min]


def merge_records(groups: list[list[PaperRecord]]) -> list[PaperRecord]:
    """Merge results of several sources, deduplicating by DOI, arXiv id, then title."""
    from researchmind.ingestion.metadata import normalize_title

    merged: list[PaperRecord] = []
    for group in groups:
        for rec in group:
            match = None
            for i, m in enumerate(merged):
                if (rec.doi and rec.doi == m.doi) or (rec.arxiv_id and rec.arxiv_id == m.arxiv_id):
                    match = i
                    break
                if normalize_title(rec.title) == normalize_title(m.title) and (
                    not rec.year or not m.year or abs(rec.year - m.year) <= 1
                ):
                    match = i
                    break
            if match is None:
                merged.append(rec)
            else:
                merged[match] = merged[match].merge(rec)
    return merged


class Discovery:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = {s.name: s for s in sources}

    def search(
        self, query: str, sources: list[str] | None = None, limit: int = 10,
        year_min: int | None = None,
    ) -> tuple[list[PaperRecord], dict[str, str]]:
        """Return merged records and per-source errors (a failing source does not fail the search)."""
        chosen = [self.sources[n] for n in (sources or list(self.sources)) if n in self.sources]
        errors: dict[str, str] = {}
        groups: list[list[PaperRecord]] = []

        def run(src: Source) -> list[PaperRecord]:
            return src.search(query, limit=limit, year_min=year_min)

        with ThreadPoolExecutor(max_workers=len(chosen) or 1) as pool:
            futures = {pool.submit(run, s): s.name for s in chosen}
            for fut, name in futures.items():
                try:
                    groups.append(fut.result())
                except Exception as exc:
                    errors[name] = str(exc)
                    log.warning("discovery_source_failed", source=name, error=str(exc))
        return merge_records(groups), errors


def build_discovery(settings=None) -> Discovery:
    from researchmind.config import get_settings

    s = settings or get_settings()
    http = RateLimitedClient(
        cache_dir=s.cache_dir,
        cache_ttl_s=s.http_cache_ttl_s,
        contact_email=s.contact_email,
        min_interval_s={
            "api.openalex.org": 0.15,
            "api.semanticscholar.org": 1.1 if not s.semantic_scholar_api_key else 0.35,
            "api.crossref.org": 0.2,
            "export.arxiv.org": 3.0,  # arXiv asks for one request every 3 seconds
        },
    )
    key = s.semantic_scholar_api_key.get_secret_value() if s.semantic_scholar_api_key else None
    return Discovery([
        OpenAlex(http, s.contact_email),
        SemanticScholar(http, key),
        Crossref(http, s.contact_email),
        ArXiv(http),
    ])
