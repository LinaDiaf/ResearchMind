"""Discovery clients, tested on hand-written responses that follow each API's documented format."""

from __future__ import annotations

import httpx
import respx

from researchmind.discovery.http import DiscoveryError, RateLimitedClient
from researchmind.discovery.models import PaperRecord
from researchmind.discovery.sources import (
    ArXiv,
    Crossref,
    Discovery,
    OpenAlex,
    SemanticScholar,
    merge_records,
)

OPENALEX_WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.1000/XYZ.1",
    "title": "A Paper",
    "publication_year": 2021,
    "authorships": [{"author": {"display_name": "Ana Silva"}}],
    "abstract_inverted_index": {"Retrieval": [0], "helps": [1], "QA.": [2]},
    "primary_location": {"landing_page_url": "https://x.org/p", "source": {"display_name": "ACL"}},
    "locations": [{"landing_page_url": "https://arxiv.org/abs/2101.00001v2"}],
    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2101.00001", "license": "cc-by"},
    "open_access": {"is_oa": True},
    "cited_by_count": 42,
}

S2_PAPER = {
    "paperId": "abc", "title": "A Paper", "abstract": None, "year": 2021, "venue": "ACL",
    "authors": [{"name": "Ana Silva"}], "externalIds": {"DOI": "10.1000/xyz.1", "ArXiv": "2101.00001"},
    "openAccessPdf": {"url": "https://arxiv.org/pdf/2101.00001", "license": None},
    "url": "https://www.semanticscholar.org/paper/abc", "citationCount": 50, "isOpenAccess": True,
}

CROSSREF_ITEM = {
    "DOI": "10.2000/other", "title": ["Another Paper"], "author": [{"given": "Bo", "family": "Li"}],
    "issued": {"date-parts": [[2019, 5]]}, "container-title": ["Journal X"],
    "abstract": "<jats:p>An <jats:italic>abstract</jats:italic>.</jats:p>",
    "license": [{"URL": "http://creativecommons.org/licenses/by/4.0/"}],
    "is-referenced-by-count": 7, "URL": "https://doi.org/10.2000/other",
}

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <published>2021-01-01T00:00:00Z</published>
    <title>A
      Paper</title>
    <summary>Retrieval helps QA.</summary>
    <author><name>Ana Silva</name></author>
    <arxiv:doi>10.1000/xyz.1</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2101.00001v2" rel="related"/>
  </entry>
</feed>"""


def http():
    return RateLimitedClient(cache_dir=None)


def test_openalex_parse():
    r = OpenAlex(http()).parse(OPENALEX_WORK)
    assert r.doi == "10.1000/xyz.1" and r.arxiv_id == "2101.00001"
    assert r.abstract == "Retrieval helps QA." and r.venue == "ACL"
    assert r.openalex_id == "W123" and r.is_open_access


def test_s2_crossref_arxiv_parse():
    s = SemanticScholar(http()).parse(S2_PAPER)
    assert s.doi == "10.1000/xyz.1" and s.citation_count == 50
    c = Crossref(http()).parse(CROSSREF_ITEM)
    assert c.year == 2019 and c.authors == ["Bo Li"] and c.abstract == "An abstract ."
    a = ArXiv(http()).parse_feed(ARXIV_FEED)[0]
    assert a.title == "A Paper" and a.arxiv_id == "2101.00001"
    assert a.pdf_url.startswith("https://")


def test_merge_records_deduplicates_across_sources():
    oa = OpenAlex(http()).parse(OPENALEX_WORK)
    s2 = SemanticScholar(http()).parse(S2_PAPER)
    ax = ArXiv(http()).parse_feed(ARXIV_FEED)[0]
    other = Crossref(http()).parse(CROSSREF_ITEM)
    merged = merge_records([[oa], [s2, other], [ax]])
    assert len(merged) == 2
    m = next(r for r in merged if r.doi == "10.1000/xyz.1")
    assert m.citation_count == 50 and "semantic_scholar" in m.source


@respx.mock
def test_discovery_tolerates_failing_source():
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [OPENALEX_WORK]}))
    respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(400))
    d = Discovery([OpenAlex(http()), Crossref(http())])
    records, errors = d.search("retrieval", limit=5)
    assert len(records) == 1 and "crossref" in errors


@respx.mock
def test_http_cache_and_retry(tmp_path):
    route = respx.get("https://api.example.org/x").mock(side_effect=[
        httpx.Response(503), httpx.Response(200, json={"ok": 1}),
    ])
    client = RateLimitedClient(cache_dir=tmp_path)
    assert client.get_json("https://api.example.org/x", {"q": 1}) == {"ok": 1}
    assert client.get_json("https://api.example.org/x", {"q": 1}) == {"ok": 1}  # cached
    assert route.call_count == 2


@respx.mock
def test_http_error_raises():
    respx.get("https://api.example.org/y").mock(return_value=httpx.Response(404))
    try:
        RateLimitedClient().get_json("https://api.example.org/y")
    except DiscoveryError:
        pass
    else:
        raise AssertionError("expected DiscoveryError")


def test_record_key():
    assert PaperRecord(title="T", doi="https://doi.org/10.1/A", source="x").key == "doi:10.1/a"
