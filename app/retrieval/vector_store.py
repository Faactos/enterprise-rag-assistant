"""Qdrant wrapper -- embedded file mode by default, server mode if QDRANT_URL.

Embedded mode holds an exclusive file lock, so exactly one process may open it.
That is why the API owns the store and the UI and eval scripts go through HTTP
rather than importing this module.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..config import Settings


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float


class VectorStore:
    def __init__(self, settings: Settings, dimension: int):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        self.collection = settings.collection
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self._ensure(dimension)

    def _ensure(self, dimension: int) -> None:
        qm = self._qm
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
            )

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        qm = self._qm
        points = [
            qm.PointStruct(
                # Qdrant ids must be a UUID or an int; derive one deterministically
                # from the chunk id so re-indexing replaces rather than duplicates.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                vector=vector,
                payload=chunk,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], doc_ids: list[str], limit: int) -> list[Hit]:
        if not doc_ids:
            return []
        qm = self._qm
        flt = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(doc_ids)))]
        )
        found = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
        return [
            Hit(
                chunk_id=p.payload["chunk_id"],
                doc_id=p.payload["doc_id"],
                page=int(p.payload["page"]),
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in found
        ]

    def delete_document(self, doc_id: str) -> None:
        qm = self._qm
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )
