"""The composition root: one object that owns the models, store, and database.

Built once per process. The embedded Qdrant store takes an exclusive file lock,
so a second instance in the same process is harmless but a second *process* is
not -- which is why the API builds this at startup and nothing else imports it.
"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .config import Settings, settings as default_settings
from .generation.answerer import Answer, Answerer
from .generation.llm import build_llm
from .ingestion import chunker, pdf_loader
from .retrieval.embedder import Embedder
from .retrieval.retriever import Passage, Retriever
from .retrieval.vector_store import VectorStore


class AccessDenied(PermissionError):
    """The caller may not read or modify this document."""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    title: str
    pages: int
    chunks: int

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "pages": self.pages, "chunks": self.chunks}


class RagService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        db.init()
        Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder(self.settings.embed_model)
        self.store = VectorStore(self.settings, self.embedder.dimension)
        self.retriever = Retriever(self.settings, self.store, self.embedder)
        self._llm = None

    @property
    def llm(self):
        # Lazy so retrieval-only work (and the eval's --retrieval-only mode)
        # never requires an API key.
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # -- documents ---------------------------------------------------------

    def ingest(self, user_id: str, path: str | Path, title: str | None = None) -> IngestResult:
        path = Path(path)
        pages = pdf_loader.load(path)
        if not pages:
            raise ValueError("No extractable text found -- is this a scanned PDF?")

        chunks = chunker.chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no chunks.")

        doc_id = str(uuid.uuid4())
        title = title or path.name
        stored = Path(self.settings.upload_dir) / f"{doc_id}{path.suffix.lower()}"
        shutil.copyfile(path, stored)

        payloads = [
            {
                "chunk_id": f"{doc_id}:{c.ordinal}",
                "doc_id": doc_id,
                "page": c.page,
                "ordinal": c.ordinal,
                "text": c.text,
                "title": title,
            }
            for c in chunks
        ]
        self.store.upsert(payloads, self.embedder.embed_passages([c.text for c in chunks]))

        with db.connect() as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, owner_id, title, filename, pages, chunks, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    doc_id,
                    user_id,
                    title,
                    stored.name,
                    len(pages),
                    len(chunks),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, page, ordinal, text) VALUES (?,?,?,?,?)",
                [(p["chunk_id"], doc_id, p["page"], p["ordinal"], p["text"]) for p in payloads],
            )

        return IngestResult(doc_id, title, len(pages), len(chunks))

    def list_documents(self, user_id: str) -> list[dict]:
        ids = db.readable_doc_ids(user_id)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT doc_id, owner_id, title, pages, chunks, created_at FROM documents"
                f" WHERE doc_id IN ({placeholders}) ORDER BY created_at DESC",
                tuple(ids),
            ).fetchall()
        return [{**dict(r), "owned": r["owner_id"] == user_id} for r in rows]

    def delete_document(self, user_id: str, doc_id: str) -> None:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT owner_id, filename FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                raise KeyError(doc_id)
            if row["owner_id"] != user_id:
                # Being able to *read* a shared document does not imply being
                # able to destroy it for its owner.
                raise AccessDenied("Only the owner may delete this document.")
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM shares WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        self.store.delete_document(doc_id)
        stored = Path(self.settings.upload_dir) / row["filename"]
        stored.unlink(missing_ok=True)

    def share_document(self, user_id: str, doc_id: str, email: str) -> dict:
        email = email.strip().lower()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT owner_id FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                raise KeyError(doc_id)
            if row["owner_id"] != user_id:
                raise AccessDenied("Only the owner may share this document.")
            target = conn.execute(
                "SELECT user_id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if target is None:
                raise KeyError(email)
            conn.execute(
                "INSERT OR IGNORE INTO shares (doc_id, user_id) VALUES (?,?)",
                (doc_id, target["user_id"]),
            )
        return {"doc_id": doc_id, "shared_with": email}

    # -- retrieval and answering -------------------------------------------

    def _scope(self, user_id: str, doc_ids: list[str] | None) -> tuple[list[str], dict[str, str]]:
        """Resolve the document scope, intersected with what the user may read.

        A caller-supplied doc_ids list narrows the scope; it can never widen it.
        """
        allowed = db.readable_doc_ids(user_id)
        if doc_ids:
            requested = set(doc_ids)
            allowed = [d for d in allowed if d in requested]
        if not allowed:
            return [], {}
        placeholders = ",".join("?" * len(allowed))
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT doc_id, title FROM documents WHERE doc_id IN ({placeholders})",
                tuple(allowed),
            ).fetchall()
        return allowed, {r["doc_id"]: r["title"] for r in rows}

    def retrieve(self, user_id: str, question: str, doc_ids: list[str] | None = None) -> list[Passage]:
        scope, titles = self._scope(user_id, doc_ids)
        return self.retriever.retrieve(question, scope, titles)

    def ask(self, user_id: str, question: str, doc_ids: list[str] | None = None) -> Answer:
        scope, titles = self._scope(user_id, doc_ids)
        passages = self.retriever.retrieve(question, scope, titles)
        return Answerer(self.settings, self.llm).answer(question, passages, titles)
