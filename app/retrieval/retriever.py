"""The retrieval pipeline: scope -> candidates -> fuse -> rerank -> top-k."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from . import fusion, lexical
from .embedder import Embedder
from .vector_store import Hit, VectorStore


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "text": self.text,
            "score": round(self.score, 4),
        }


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder):
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, doc_ids: list[str], titles: dict[str, str]) -> list[Passage]:
        s = self.settings
        if not doc_ids:
            return []

        dense = self.store.search(self.embedder.embed_query(query), doc_ids, s.candidate_k)

        if s.use_hybrid:
            sparse = lexical.search(query, doc_ids, s.candidate_k)
            candidates = fusion.rrf([dense, sparse], k=s.rrf_k)
        else:
            candidates = dense

        if s.use_reranker and candidates:
            candidates = rerank_hits(query, candidates[: s.candidate_k], s)
        else:
            candidates = candidates[: s.top_k]

        return [
            Passage(h.chunk_id, h.doc_id, titles.get(h.doc_id, h.doc_id), h.page, h.text, h.score)
            for h in candidates[: s.top_k]
        ]


def rerank_hits(query: str, hits: list[Hit], settings: Settings) -> list[Hit]:
    from .reranker import rerank

    return rerank(query, hits, settings.rerank_model, settings.top_k)
