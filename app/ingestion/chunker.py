"""Page-aware chunking.

Chunks never span a page boundary. A chunk built across pages cannot be cited
honestly -- half of it is on page 3 and half on page 4 -- so the small loss in
context is worth the traceability.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pdf_loader import Page


@dataclass(frozen=True)
class Chunk:
    page: int
    ordinal: int
    text: str


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    ordinal = 0
    step = size - overlap
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            window = words[start : start + size]
            text = " ".join(window).strip()
            if text:
                chunks.append(Chunk(page.number, ordinal, text))
                ordinal += 1
            if start + size >= len(words):
                break
            start += step
    return chunks
