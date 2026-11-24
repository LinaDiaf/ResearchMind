"""Section-aware chunking that preserves page and character provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from researchmind.db.models import SectionKind
from researchmind.ingestion.pdf_parser import Line, ParsedDocument, join_lines
from researchmind.ingestion.sections import DetectedSection

_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


@dataclass
class ChunkDraft:
    ord: int
    text: str
    section_index: int
    section_title: str
    section_kind: SectionKind
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    is_reference_list: bool

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class _Sentence:
    text: str
    char_start: int
    char_end: int


def _sentences(lines: list[Line]) -> list[_Sentence]:
    """Split a section into sentences, mapping each one back to document offsets."""
    out: list[_Sentence] = []
    # Paragraph-level text with a map from joined-text offsets to document offsets.
    text = join_lines(lines)
    # Map each line's position in the joined text to its document offset (approximate when
    # hyphenation is undone: offsets stay within the right line).
    anchors: list[tuple[int, int]] = []
    pos = 0
    for ln in lines:
        idx = text.find(ln.text[: min(len(ln.text), 20)], pos)
        if idx >= 0:
            anchors.append((idx, ln.char_start))
            pos = idx
    anchors.append((len(text), lines[-1].char_end))

    def to_doc(i: int) -> int:
        prev = anchors[0]
        for a in anchors:
            if a[0] > i:
                break
            prev = a
        return prev[1] + (i - prev[0])

    start = 0
    for m in list(_SENT_END.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        s = text[start:end].strip()
        if s:
            out.append(_Sentence(s, to_doc(start), min(to_doc(end), lines[-1].char_end)))
        if m:
            start = m.end()
    return out


def chunk_document(
    doc: ParsedDocument,
    sections: list[DetectedSection],
    *,
    target_tokens: int = 350,
    overlap_tokens: int = 60,
    min_tokens: int = 40,
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for si, section in enumerate(sections):
        sents = _sentences(section.lines)
        if not sents:
            continue
        window: list[_Sentence] = []
        tokens = 0

        def flush(final: bool, si: int = si, section: DetectedSection = section) -> None:
            nonlocal window, tokens
            if not window:
                return
            text = " ".join(s.text for s in window)
            tc = approx_tokens(text)
            if tc < min_tokens and chunks and chunks[-1].section_index == si and final:
                # merge a short tail into the previous chunk of the same section
                prev = chunks[-1]
                prev.text = f"{prev.text} {text}"
                prev.char_end = window[-1].char_end
                prev.page_end = doc.page_of(prev.char_end - 1)
                prev.token_count = approx_tokens(prev.text)
            else:
                start, end = window[0].char_start, window[-1].char_end
                chunks.append(ChunkDraft(
                    ord=len(chunks), text=text, section_index=si, section_title=section.title,
                    section_kind=section.kind, page_start=doc.page_of(start),
                    page_end=doc.page_of(max(start, end - 1)), char_start=start, char_end=end,
                    token_count=tc, is_reference_list=section.kind is SectionKind.references,
                ))
            if final:
                window, tokens = [], 0
                return
            # keep the last sentences as overlap
            keep: list[_Sentence] = []
            kept = 0
            for s in reversed(window):
                t = approx_tokens(s.text)
                if kept + t > overlap_tokens:
                    break
                keep.insert(0, s)
                kept += t
            window, tokens = keep, kept

        for s in sents:
            t = approx_tokens(s.text)
            if tokens + t > target_tokens and window:
                flush(final=False)
            window.append(s)
            tokens += t
        flush(final=True)
    return chunks
