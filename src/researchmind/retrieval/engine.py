"""Hybrid retrieval engine: dense + BM25 → weighted RRF → diversity → cross-encoder rerank."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchmind.db.models import Chunk, Paper
from researchmind.embeddings import Embedder
from researchmind.log import get_logger
from researchmind.retrieval.dense import dense_search
from researchmind.retrieval.fusion import weighted_rrf
from researchmind.retrieval.lexical import load_index
from researchmind.retrieval.reranker import NoReranker, Reranker
from researchmind.retrieval.types import RetrievalRequest, RetrievalResult, RetrievedPassage

log = get_logger(__name__)


class RetrievalEngine:
    def __init__(
        self,
        session_factory,
        embedder: Embedder,
        reranker: Reranker | None,
        index_root: Path,
    ) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.reranker = reranker or NoReranker()
        self.index_root = index_root

    def retrieve(self, req: RetrievalRequest) -> RetrievalResult:
        start = time.perf_counter()
        rankings: list[tuple[float, list[tuple[uuid.UUID, float]]]] = []
        dense_scores: dict[uuid.UUID, float] = {}
        lexical_scores: dict[uuid.UUID, float] = {}
        n_dense = n_lexical = 0
        with self.session_factory() as session:
            if req.strategy in ("dense", "hybrid") and req.dense_weight > 0:
                for q in req.queries:
                    vec = self.embedder.embed_query(q)
                    hits = dense_search(session, req.project_id, vec, self.embedder.model,
                                        req.filters, req.candidates)
                    n_dense += len(hits)
                    rankings.append((req.dense_weight, hits))
                    for cid, s in hits:
                        dense_scores[cid] = max(dense_scores.get(cid, 0.0), s)
            if req.strategy in ("lexical", "hybrid") and req.lexical_weight > 0:
                index = load_index(self.index_root, req.project_id)
                if index is not None:
                    for q in req.queries:
                        hits = index.search(q, req.filters, req.candidates)
                        n_lexical += len(hits)
                        rankings.append((req.lexical_weight, hits))
                        for cid, s in hits:
                            lexical_scores[cid] = max(lexical_scores.get(cid, 0.0), s)
                else:
                    log.warning("lexical_index_missing", project_id=str(req.project_id))

            excluded = set(req.exclude_chunk_ids)
            fused = [(c, s) for c, s in weighted_rrf(rankings) if c not in excluded]
            pool_ids = [c for c, _ in fused[: max(req.rerank_candidates * 2, req.k)]]
            passages = self._load(session, pool_ids)

        fused_map = dict(fused)
        for p in passages:
            p.scores["fused"] = fused_map.get(p.chunk_id, 0.0)
            if p.chunk_id in dense_scores:
                p.scores["dense"] = dense_scores[p.chunk_id]
            if p.chunk_id in lexical_scores:
                p.scores["bm25"] = lexical_scores[p.chunk_id]
        passages.sort(key=lambda p: -p.scores["fused"])
        passages = self._diversify(passages, req.max_per_paper)[: req.rerank_candidates]

        if passages and not isinstance(self.reranker, NoReranker):
            query = req.queries[0]
            scores = self.reranker.score(query, [p.text for p in passages])
            for p, s in zip(passages, scores, strict=True):
                p.scores["rerank"] = float(s)
            passages.sort(key=lambda p: -p.scores["rerank"])

        return RetrievalResult(
            passages=passages[: req.k],
            request=req,
            n_dense=n_dense,
            n_lexical=n_lexical,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    @staticmethod
    def _diversify(passages: list[RetrievedPassage], max_per_paper: int) -> list[RetrievedPassage]:
        seen_text: set[str] = set()
        per_paper: dict[uuid.UUID, int] = {}
        out = []
        for p in passages:
            key = " ".join(p.text.lower().split())[:300]
            if key in seen_text:
                continue
            if per_paper.get(p.paper_id, 0) >= max_per_paper:
                continue
            seen_text.add(key)
            per_paper[p.paper_id] = per_paper.get(p.paper_id, 0) + 1
            out.append(p)
        return out

    @staticmethod
    def _load(session: Session, ids: list[uuid.UUID]) -> list[RetrievedPassage]:
        if not ids:
            return []
        rows = session.execute(
            select(Chunk, Paper).join(Paper, Paper.id == Chunk.paper_id).where(Chunk.id.in_(ids))
        ).all()
        by_id = {c.id: (c, p) for c, p in rows}
        out = []
        for cid in ids:
            if cid not in by_id:
                continue
            c, p = by_id[cid]
            out.append(RetrievedPassage(
                chunk_id=c.id, paper_id=p.id, text=c.text, paper_title=p.title, year=p.year,
                doi=p.doi, arxiv_id=p.arxiv_id, section_title=c.section_title,
                section_kind=c.section_kind, page_start=c.page_start, page_end=c.page_end,
                char_start=c.char_start, char_end=c.char_end,
            ))
        return out
