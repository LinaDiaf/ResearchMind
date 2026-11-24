"""Section detection and typing from font and numbering cues."""

from __future__ import annotations

import re
from dataclasses import dataclass

from researchmind.db.models import SectionKind
from researchmind.ingestion.pdf_parser import Line, ParsedDocument

_NUMBERED = re.compile(r"^(?:(?:\d+|[IVX]+|[A-Z])(?:\.\d+)*\.?)\s+[A-Z]")
_LEADING_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+")
_KIND_RULES: list[tuple[SectionKind, tuple[str, ...]]] = [
    (SectionKind.abstract, ("abstract",)),
    (SectionKind.references, ("references", "bibliography", "works cited")),
    (SectionKind.related_work, ("related work", "background", "prior work", "literature review")),
    (SectionKind.limitations, ("limitation", "threats to validity", "future work", "broader impact")),
    (SectionKind.introduction, ("introduction", "motivation")),
    (SectionKind.results, ("result", "findings", "evaluation", "analysis", "ablation")),
    (SectionKind.experiments, ("experiment", "experimental", "setup", "dataset", "implementation details", "benchmark")),
    (SectionKind.method, ("method", "approach", "model", "architecture", "framework", "proposed", "algorithm", "system")),
    (SectionKind.discussion, ("discussion",)),
    (SectionKind.conclusion, ("conclusion", "concluding", "summary")),
    (SectionKind.appendix, ("appendix", "supplementary")),
]


def classify_heading(title: str) -> SectionKind:
    t = _LEADING_NUMBER.sub("", title).lower().strip()
    for kind, keys in _KIND_RULES:
        if any(k in t for k in keys):
            return kind
    return SectionKind.other


@dataclass
class DetectedSection:
    title: str
    kind: SectionKind
    level: int
    lines: list[Line]

    @property
    def page_start(self) -> int:
        return self.lines[0].page if self.lines else 1

    @property
    def page_end(self) -> int:
        return self.lines[-1].page if self.lines else self.page_start


def _is_heading(line: Line, body: float) -> bool:
    text = line.text.strip()
    words = text.split()
    if not (1 <= len(words) <= 12) or len(text) > 90:
        return False
    if text.endswith((".", ",", ";", ":")) and not _NUMBERED.match(text):
        return False
    if re.fullmatch(r"[\d\s.,()%-]+", text):
        return False
    larger = line.size >= body * 1.12
    numbered = bool(_NUMBERED.match(text))
    keyword = classify_heading(text) is not SectionKind.other and len(words) <= 4
    if larger and (text[:1].isupper() or numbered):
        return True
    if line.bold and (numbered or keyword):
        return True
    return numbered and keyword


def _level(text: str) -> int:
    m = re.match(r"^(\d+(?:\.\d+)*)", text)
    return m.group(1).count(".") + 1 if m else 1


def detect_sections(doc: ParsedDocument) -> list[DetectedSection]:
    sections: list[DetectedSection] = []
    current = DetectedSection("Front matter", SectionKind.other, 1, [])
    title_page_max = max((ln.size for ln in doc.lines if ln.page == 1), default=0)
    for ln in doc.lines:
        # the paper title (largest text on page 1) is not a section heading
        is_title = ln.page == 1 and ln.size == title_page_max and ln.size > doc.body_font_size * 1.4
        if not is_title and _is_heading(ln, doc.body_font_size):
            if current.lines or current.title != "Front matter":
                sections.append(current)
            current = DetectedSection(ln.text.strip(), classify_heading(ln.text), _level(ln.text), [])
            continue
        current.lines.append(ln)
    sections.append(current)
    sections = [s for s in sections if s.lines]
    # Everything after "References" that is not an appendix belongs to the references.
    in_refs = False
    for s in sections:
        if s.kind is SectionKind.references:
            in_refs = True
        elif in_refs and s.kind is not SectionKind.appendix:
            s.kind = SectionKind.references
    # Inline "Abstract" paragraphs in front matter (common in two-column templates)
    if sections and sections[0].title == "Front matter":
        fm = sections[0]
        for i, ln in enumerate(fm.lines):
            if re.match(r"^abstract\b", ln.text.strip(), re.IGNORECASE):
                body = fm.lines[i:]
                fm.lines = fm.lines[:i]
                sections.insert(1, DetectedSection("Abstract", SectionKind.abstract, 1, body))
                if not fm.lines:
                    sections.pop(0)
                break
    return sections
