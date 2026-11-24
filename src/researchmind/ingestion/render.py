"""Render a PDF page with a passage highlighted, for evidence inspection."""

from __future__ import annotations

import re

import pymupdf


def _segments(text: str, words: int = 6, limit: int = 40) -> list[str]:
    tokens = re.sub(r"\s+", " ", text).strip().split(" ")
    return [" ".join(tokens[i:i + words]) for i in range(0, len(tokens), words)][:limit]


def render_highlight(pdf_path: str, page_number: int, passage: str, zoom: float = 1.6) -> tuple[bytes, int]:
    """Return (PNG bytes, number of highlighted fragments). ``page_number`` is 1-based."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]
        found = 0
        for seg in _segments(passage):
            if len(seg) < 12:
                continue
            for quad in page.search_for(seg, quads=True):
                annot = page.add_highlight_annot(quad)
                annot.set_colors(stroke=(1.0, 0.85, 0.2))
                annot.update()
                found += 1
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return pix.tobytes("png"), found
