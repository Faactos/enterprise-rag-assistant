"""SQLite metadata: users, documents, the sharing ACL, and chunk text.

Chunk text lives here as well as in Qdrant's payload so BM25 can be built
without a vector round-trip -- and so the lexical half of hybrid search still
works if the vector store is rebuilt.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    email      TEXT UNIQUE NOT NULL,
    token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    title      TEXT NOT NULL,
    filename   TEXT NOT NULL,
    pages      INTEGER NOT NULL,
    chunks     INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shares (
    doc_id  TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (doc_id, user_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    page     INTEGER NOT NULL,
    ordinal  INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_owner ON documents(owner_id);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = Path(settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def readable_doc_ids(user_id: str) -> list[str]:
    """Documents the user owns or has been given read access to.

    Retrieval is scoped with this rather than filtering results afterwards --
    a passage the caller may not read never enters the candidate pool, so it
    can never leak through a citation or a reranker score.
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT doc_id FROM documents WHERE owner_id = ?
            UNION
            SELECT doc_id FROM shares    WHERE user_id = ?
            """,
            (user_id, user_id),
        ).fetchall()
    return [r["doc_id"] for r in rows]
