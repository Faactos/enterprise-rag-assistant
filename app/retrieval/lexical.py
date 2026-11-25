"""BM25 over the chunk text in SQLite.

Lexical search is here because embeddings reliably miss exact tokens -- an
invoice number, a statute reference, a product SKU. Those are precisely the
queries where a confident wrong answer is most expensive.
"""
from __future__ import annotations

import re

from .. import db
from .vector_store import Hit

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def search(query: str, doc_ids: list[str], limit: int) -> list[Hit]:
    if not doc_ids:
        return []
    from rank_bm25 import BM25Okapi

    placeholders = ",".join("?" * len(doc_ids))
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT chunk_id, doc_id, page, text FROM chunks WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    if not rows:
        return []

    corpus = [tokenize(r["text"]) for r in rows]
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = BM25Okapi(corpus).get_scores(tokens)
    ranked = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)[:limit]
    return [
        Hit(r["chunk_id"], r["doc_id"], int(r["page"]), r["text"], float(s))
        for r, s in ranked
        if s > 0
    ]
