"""Text extraction that preserves page numbers.

A citation is only useful if it points at a page, so extraction is per-page
from the start -- recovering page boundaries after concatenating is guesswork.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a reader would cite it
    text: str


def load(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")


def _load_pdf(path: Path) -> list[Page]:
    import fitz  # PyMuPDF

    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(i, text))
    return pages


def _load_text(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Respect explicit form feeds; otherwise a text file is a single page.
    parts = [p.strip() for p in raw.split("\f")]
    return [Page(i, p) for i, p in enumerate(parts, start=1) if p]
