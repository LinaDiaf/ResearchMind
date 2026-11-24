"""BM25 keyword retrieval (bm25s), one index per project.

The index is a derived artefact: it is rebuilt from the ``chunks`` table after each ingestion
and can be deleted at any time.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from sqlalchemy import text
from sqlalchemy.orm import Session

from researchmind.retrieval.types import RetrievalFilters

_stemmer = Stemmer.Stemmer("english")
_lock = threading.Lock()
_cache: dict[str, tuple[float, LexicalIndex]] = {}


def _tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False)


@dataclass
class LexicalIndex:
    retriever: bm25s.BM25 | None
    chunk_ids: list[uuid.UUID]
    paper_ids: np.ndarray  # str per chunk
    years: np.ndarray  # float, nan when unknown
    kinds: np.ndarray  # str per chunk
    is_ref: np.ndarray  # bool per chunk

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def mask(self, f: RetrievalFilters) -> np.ndarray:
        m = np.ones(self.size, dtype=bool)
        if not f.include_references:
            m &= ~self.is_ref
        if f.paper_ids:
            m &= np.isin(self.paper_ids, [str(p) for p in f.paper_ids])
        if f.year_min is not None:
            m &= np.nan_to_num(self.years, nan=-1) >= f.year_min
        if f.year_max is not None:
            m &= np.nan_to_num(self.years, nan=1e9) <= f.year_max
        if f.section_kinds:
            m &= np.isin(self.kinds, [k.value for k in f.section_kinds])
        return m

    def search(self, query: str, filters: RetrievalFilters, limit: int) -> list[tuple[uuid.UUID, float]]:
        if self.retriever is None or self.size == 0:
            return []
        mask = self.mask(filters)
        if not mask.any():
            return []
        q = _tokenize([query])
        if not q.vocab:
            return []
        k = min(limit, int(mask.sum()))
        docs, scores = self.retriever.retrieve(
            q, k=k, show_progress=False, weight_mask=mask.astype(float), n_threads=1
        )
        out = []
        for idx, score in zip(docs[0], scores[0], strict=True):
            if score > 0:
                out.append((self.chunk_ids[int(idx)], float(score)))
        return out


def _dir(index_root: Path, project_id: uuid.UUID) -> Path:
    return index_root / f"bm25-{project_id}"


def rebuild_index(session: Session, project_id: uuid.UUID, index_root: Path) -> int:
    rows = session.execute(
        text(
            "SELECT c.id, c.text, c.section_title, c.paper_id, p.year, c.section_kind::text, "
            "c.is_reference_list, p.title "
            "FROM chunks c JOIN papers p ON p.id = c.paper_id "
            "JOIN project_papers pp ON pp.paper_id = c.paper_id AND pp.project_id = :pid "
            "ORDER BY c.paper_id, c.ord"
        ),
        {"pid": project_id},
    ).all()
    target = _dir(index_root, project_id)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    meta = {
        "chunk_ids": [str(r[0]) for r in rows],
        "paper_ids": [str(r[3]) for r in rows],
        "years": [r[4] for r in rows],
        "kinds": [r[5] for r in rows],
        "is_ref": [bool(r[6]) for r in rows],
    }
    if rows:
        # Section title and paper title are indexed with the passage: they carry key terms.
        corpus = [f"{r[7]}. {r[2] or ''}. {r[1]}" for r in rows]
        retriever = bm25s.BM25()
        retriever.index(_tokenize(corpus), show_progress=False)
        retriever.save(str(tmp), show_progress=False)
    (tmp / "meta.json").write_text(json.dumps(meta))
    with _lock:
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        _cache.pop(str(target), None)
    return len(rows)


def load_index(index_root: Path, project_id: uuid.UUID) -> LexicalIndex | None:
    target = _dir(index_root, project_id)
    meta_path = target / "meta.json"
    if not meta_path.exists():
        return None
    mtime = meta_path.stat().st_mtime
    with _lock:
        cached = _cache.get(str(target))
        if cached and cached[0] == mtime:
            return cached[1]
    meta = json.loads(meta_path.read_text())
    retriever = bm25s.BM25.load(str(target)) if meta["chunk_ids"] else None
    idx = LexicalIndex(
        retriever=retriever,
        chunk_ids=[uuid.UUID(c) for c in meta["chunk_ids"]],
        paper_ids=np.array(meta["paper_ids"], dtype=object),
        years=np.array([np.nan if y is None else y for y in meta["years"]], dtype=float),
        kinds=np.array(meta["kinds"], dtype=object),
        is_ref=np.array(meta["is_ref"], dtype=bool),
    )
    with _lock:
        _cache[str(target)] = (mtime, idx)
    return idx
