"""Local sentence-transformer embeddings, loaded once per process.

bge models want an instruction prefix on the *query* side only; embedding a
query like a passage measurably degrades recall, so the two are separate calls.
"""
from __future__ import annotations

from functools import lru_cache

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=4)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return int(_model(self.model_name).get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = _model(self.model_name).encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = _model(self.model_name).encode(
            QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()

    def warm(self) -> None:
        _model(self.model_name)
