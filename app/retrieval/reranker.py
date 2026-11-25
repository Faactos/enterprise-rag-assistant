"""Cross-encoder reranking.

The bi-encoder scores a query and a passage independently; the cross-encoder
reads them together, which is why it can tell "the policy does not cover X"
apart from "the policy covers X". It is also ~1GB and slow, hence the hard cap
on how many candidates reach it.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .vector_store import Hit


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def _sigmoid(x: float) -> float:
    # The model emits a logit. MIN_SCORE is expressed as a probability, so the
    # squashing has to happen here rather than in the caller's threshold.
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def rerank(query: str, hits: list[Hit], model_name: str, top_k: int) -> list[Hit]:
    if not hits:
        return []
    raw = _cross_encoder(model_name).predict([(query, h.text) for h in hits])
    scored = [
        Hit(h.chunk_id, h.doc_id, h.page, h.text, _sigmoid(float(s)))
        for h, s in zip(hits, raw)
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:top_k]
