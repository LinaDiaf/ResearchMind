"""Embedding models behind one interface."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from researchmind.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    pass


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# Task prefixes required by some retrieval-trained models.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "BAAI/bge-base-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "BAAI/bge-small-en-v1.5": ("", "Represent this sentence for searching relevant passages: "),
    "intfloat/e5-base-v2": ("passage: ", "query: "),
}


def _prefixes(model: str) -> tuple[str, str]:
    base = model.split(":")[0]
    return _PREFIXES.get(model) or _PREFIXES.get(base) or ("", "")


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        base_url: str = "http://localhost:11434",
        *,
        batch_size: int = 32,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._doc_prefix, self._query_prefix = _prefixes(model)
        self._client = client or httpx.Client(timeout=timeout_s)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                r = self._client.post(
                    f"{self.base_url}/api/embed", json={"model": self.model, "input": batch}
                )
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc
            vectors = r.json().get("embeddings") or []
            if len(vectors) != len(batch):
                raise EmbeddingError("Ollama returned a different number of embeddings")
            for v in vectors:
                if len(v) != self.dim:
                    raise EmbeddingError(
                        f"{self.model} returned dimension {len(v)}, EMBEDDING_DIM={self.dim}"
                    )
                out.append(_normalize(list(v)))
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model: str, dim: int, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EmbeddingError("install the 'ml' extra: pip install researchmind[ml]") from exc
        self.model = model
        self._st = SentenceTransformer(model, device=device, trust_remote_code=False)
        actual = self._st.get_sentence_embedding_dimension()
        if actual != dim:
            raise EmbeddingError(f"{model} has dimension {actual}, EMBEDDING_DIM={dim}")
        self.dim = dim
        self.batch_size = batch_size
        self._doc_prefix, self._query_prefix = _prefixes(model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._st.encode(
            [self._doc_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self._st.encode(
            [self._query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Feature-hashing bag of words + character trigrams.

    Not a semantic model. Used for tests and for running the pipeline without any model
    download; results obtained with it must not be reported as dense-retrieval results.
    """

    def __init__(self, dim: int = 768, model: str = "hashing-v1") -> None:
        self.model = model
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        feats = tokens + [t[i : i + 3] for t in tokens if len(t) > 3 for i in range(len(t) - 2)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedding_provider == "ollama":
        return OllamaEmbedder(s.embedding_model, s.embedding_dim, s.ollama_url)
    if s.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim, s.reranker_device)
    return HashingEmbedder(s.embedding_dim)
