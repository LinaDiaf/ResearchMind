from __future__ import annotations

import re

import pytest

from researchmind.db.models import SectionKind
from researchmind.ingestion.chunker import chunk_document
from researchmind.ingestion.metadata import (
    extract_identifiers,
    guess_title,
    normalize_arxiv,
    normalize_doi,
    normalize_title,
)
from researchmind.ingestion.pdf_parser import PDFParseError, parse_pdf
from researchmind.ingestion.sections import classify_heading, detect_sections
from tests.fixtures.synthetic_papers import build_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("corpus"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("-\n", "")).strip()


def test_parse_title_identifiers_and_pages(corpus):
    doc = parse_pdf(corpus["sparse"])
    assert doc.page_count == 2
    assert guess_title(doc) == "Synthetic Study A: Learned Sparse Retrieval with Term Weighting"
    assert extract_identifiers(doc) == {"arxiv_id": "2401.00001"}
    assert extract_identifiers(parse_pdf(corpus["dense"])) == {"doi": "10.5555/synthetic.0002"}


def test_sections_are_detected_and_typed(corpus):
    sections = detect_sections(parse_pdf(corpus["sparse"]))
    kinds = [s.kind for s in sections]
    for k in (SectionKind.abstract, SectionKind.introduction, SectionKind.method,
              SectionKind.experiments, SectionKind.results, SectionKind.limitations,
              SectionKind.references):
        assert k in kinds
    results = next(s for s in sections if s.kind is SectionKind.results)
    assert results.page_start == 2


def test_chunks_keep_exact_provenance(corpus):
    doc = parse_pdf(corpus["sparse"])
    chunks = chunk_document(doc, detect_sections(doc))
    assert chunks, "no chunks"
    for c in chunks:
        assert _norm(doc.text[c.char_start:c.char_end]) == _norm(c.text)
        assert doc.page_of(c.char_start) == c.page_start
    res = next(c for c in chunks if "MRR@10 of 38.2" in c.text)
    assert res.page_start == 2 and res.section_kind is SectionKind.results
    assert any(c.is_reference_list for c in chunks)


def test_chunk_windows_overlap(corpus):
    doc = parse_pdf(corpus["sparse"])
    chunks = chunk_document(doc, detect_sections(doc), target_tokens=40, overlap_tokens=25,
                            min_tokens=5)
    same = [c for c in chunks if c.section_kind is SectionKind.introduction]
    assert len(same) >= 2
    assert same[1].char_start < same[0].char_end  # overlap


def test_unreadable_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PDFParseError):
        parse_pdf(bad)


@pytest.mark.parametrize("raw,expected", [
    ("https://doi.org/10.1145/3397271.3401075", "10.1145/3397271.3401075"),
    ("doi:10.18653/V1/2020.EMNLP-MAIN.550.", "10.18653/v1/2020.emnlp-main.550"),
    ("not a doi", None),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


def test_normalize_arxiv_and_title():
    assert normalize_arxiv("https://arxiv.org/abs/2005.11401v4") == "2005.11401"
    assert normalize_arxiv("arXiv:hep-th/9901001v1") == "hep-th/9901001"
    assert normalize_title("Retrieval-Augmented Generation: A Survey!") == \
        "retrieval augmented generation a survey"


@pytest.mark.parametrize("title,kind", [
    ("4.2 Ablation Study", SectionKind.results),
    ("Related Work", SectionKind.related_work),
    ("6 Limitations and Future Work", SectionKind.limitations),
    ("3 Proposed Approach", SectionKind.method),
    ("Acknowledgments", SectionKind.other),
])
def test_classify_heading(title, kind):
    assert classify_heading(title) is kind
