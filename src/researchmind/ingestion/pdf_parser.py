"""PDF text extraction with page and character provenance (PyMuPDF).

Every line keeps its page number, font size, boldness and its character span in the
document's full text. Downstream components (sections, chunks) only work with these offsets, so
a passage can always be traced back to its page.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Line:
    text: str
    page: int  # 1-based
    size: float
    bold: bool
    char_start: int  # offsets in ParsedDocument.text
    char_end: int
    block: int


@dataclass
class ParsedDocument:
    text: str
    lines: list[Line]
    page_count: int
    page_offsets: list[int]  # char offset where each page starts
    body_font_size: float
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def page_of(self, offset: int) -> int:
        page = 1
        for i, start in enumerate(self.page_offsets):
            if offset >= start:
                page = i + 1
            else:
                break
        return page


class PDFParseError(RuntimeError):
    pass


_HYPHEN_BREAK = re.compile(r"(\w)-$")


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_pdf(path: str | Path) -> ParsedDocument:
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise PDFParseError(f"cannot open PDF: {exc}") from exc
    if doc.needs_pass:
        raise PDFParseError("PDF is encrypted")

    parts: list[str] = []
    lines: list[Line] = []
    page_offsets: list[int] = []
    sizes: Counter[float] = Counter()
    offset = 0
    block_id = 0
    warnings: list[str] = []

    for page_index, page in enumerate(doc):
        if page_index > 0:
            parts.append(PAGE_SEPARATOR)
            offset += len(PAGE_SEPARATOR)
        page_offsets.append(offset)
        data = page.get_text("dict", sort=True)
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_id += 1
            for raw_line in block.get("lines", []):
                spans = [s for s in raw_line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in raw_line["spans"]))
                if not text:
                    continue
                size = max(s["size"] for s in spans)
                # bit 4 (16) of span flags = bold; also catch "Bold" in the font name
                bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                if parts and not parts[-1].endswith(("\n", PAGE_SEPARATOR)):
                    parts.append("\n")
                    offset += 1
                start = offset
                parts.append(text)
                offset += len(text)
                lines.append(Line(text, page_index + 1, round(size, 1), bool(bold), start, offset,
                                  block_id))
                for s in spans:
                    sizes[round(s["size"], 1)] += len(s["text"])

    full = "".join(parts)
    if len(full.strip()) < 200:
        warnings.append("very little extractable text (scanned PDF without text layer?)")
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    meta = {k: v for k, v in (doc.metadata or {}).items() if isinstance(v, str) and v}
    page_count = doc.page_count
    doc.close()
    return ParsedDocument(full, lines, page_count, page_offsets, float(body), meta, warnings)


def join_lines(lines: list[Line]) -> str:
    """Join wrapped lines into running text, undoing end-of-line hyphenation."""
    out: list[str] = []
    for ln in lines:
        if out and _HYPHEN_BREAK.search(out[-1]) and ln.text[:1].islower():
            out[-1] = out[-1][:-1] + ln.text
        else:
            out.append(ln.text)
    return " ".join(out)
