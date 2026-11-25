"""Reciprocal Rank Fusion.

RRF combines rankings, not scores. That matters because a cosine similarity and
a BM25 score are not on comparable scales, and any attempt to normalise them
into one number is a tuning parameter in disguise.
"""
from __future__ import annotations

from .vector_store import Hit


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)

    if not scores:
        return []
    top = max(scores.values())
    return [
        # Rescale to 0..1 so MIN_SCORE means the same thing whether or not
        # fusion ran. An RRF sum is otherwise ~0.03 and would refuse everything.
        Hit(h.chunk_id, h.doc_id, h.page, h.text, scores[cid] / top)
        for cid, h in sorted(best.items(), key=lambda p: scores[p[0]], reverse=True)
    ]
