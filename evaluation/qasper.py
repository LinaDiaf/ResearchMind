"""QASPER loader (Dasigi et al., 2021; CC BY 4.0).

QASPER contains questions about NLP papers, answered by NLP practitioners who also selected the
paragraphs that support each answer. Those human-selected evidence paragraphs are the gold
labels for retrieval metrics; unanswerable questions measure insufficiency detection.
"""

from __future__ import annotations

import io
import json
import random
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

URLS = {
    "test": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
             "qasper-test-v0.3.json"),
    "dev": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
            "qasper-dev-v0.3.json"),
}


@dataclass
class QasperQuestion:
    paper_id: str
    question_id: str
    question: str
    answers: list[str]  # one normalized answer string per annotator
    evidence: list[str]  # union of annotators' evidence paragraphs (text only)
    unanswerable: bool  # majority of annotators marked it unanswerable
    answer_type: str


@dataclass
class QasperPaper:
    paper_id: str
    title: str
    abstract: str
    sections: list[tuple[str, list[str]]]
    questions: list[QasperQuestion] = field(default_factory=list)


def download(split: str, cache_dir: Path) -> Path:
    url, member = URLS[split]
    target = cache_dir / member
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith(member):
                f = tar.extractfile(m)
                target.write_bytes(f.read())
                return target
    raise FileNotFoundError(f"{member} not found in {url}")


def _answer_text(a: dict) -> tuple[str, str]:
    if a.get("unanswerable"):
        return "Unanswerable", "none"
    if a.get("extractive_spans"):
        return ", ".join(a["extractive_spans"]), "extractive"
    if a.get("free_form_answer"):
        return a["free_form_answer"], "abstractive"
    if a.get("yes_no") is not None:
        return ("Yes" if a["yes_no"] else "No"), "boolean"
    return "", "none"


def parse(data: dict) -> list[QasperPaper]:
    papers = []
    for pid, p in data.items():
        sections = []
        for sec in p.get("full_text") or []:
            name = (sec.get("section_name") or "").split(":::")[-1].strip()
            paras = [x for x in sec.get("paragraphs") or [] if x and x.strip()]
            if paras:
                sections.append((name, paras))
        paper = QasperPaper(pid, p.get("title", ""), p.get("abstract", ""), sections)
        for qa in p.get("qas") or []:
            answers, evidence, types, unans = [], [], [], 0
            for ann in qa.get("answers") or []:
                a = ann.get("answer") or {}
                text, kind = _answer_text(a)
                answers.append(text)
                types.append(kind)
                unans += bool(a.get("unanswerable"))
                for ev in a.get("evidence") or []:
                    if ev and not ev.startswith("FLOAT SELECTED") and ev not in evidence:
                        evidence.append(ev)
            if not answers:
                continue
            paper.questions.append(QasperQuestion(
                paper_id=pid, question_id=qa.get("question_id", ""), question=qa.get("question", ""),
                answers=answers, evidence=evidence, unanswerable=unans * 2 > len(answers),
                answer_type=max(set(types), key=types.count)))
        papers.append(paper)
    return papers


def load(split: str, cache_dir: Path, path: Path | None = None) -> list[QasperPaper]:
    source = path or download(split, cache_dir)
    return parse(json.loads(Path(source).read_text()))


def sample(papers: list[QasperPaper], n_papers: int, max_questions_per_paper: int, seed: int,
           min_evidence_questions: int = 1) -> list[QasperPaper]:
    """Deterministic sample. Papers need at least one answerable question with evidence."""
    rng = random.Random(seed)
    pool = [p for p in papers
            if sum(1 for q in p.questions if q.evidence and not q.unanswerable) >= min_evidence_questions]
    rng.shuffle(pool)
    out = []
    for p in pool[:n_papers]:
        qs = list(p.questions)
        rng.shuffle(qs)
        p.questions = qs[:max_questions_per_paper]
        out.append(p)
    return out


_WS = re.compile(r"\s+")


def norm_space(s: str) -> str:
    return _WS.sub(" ", s).strip()
