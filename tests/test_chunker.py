"""Chunking: page fidelity, overlap, and the invariants the citations rest on."""
from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import Page


def words(n: int, token: str = "w") -> str:
    return " ".join(f"{token}{i}" for i in range(n))


def test_a_chunk_never_spans_two_pages():
    # The whole citation contract depends on this: a chunk belongs to one page.
    pages = [Page(1, words(400)), Page(2, words(400))]
    chunks = chunk_pages(pages, size=512, overlap=64)
    assert {c.page for c in chunks} == {1, 2}
    assert len(chunks) == 2  # neither page was merged into the other


def test_long_page_is_split_with_overlap():
    pages = [Page(7, words(1000))]
    chunks = chunk_pages(pages, size=400, overlap=100)
    assert len(chunks) > 1
    assert all(c.page == 7 for c in chunks)

    first = chunks[0].text.split()
    second = chunks[1].text.split()
    # The tail of one chunk is the head of the next, or a sentence split across
    # the boundary becomes unretrievable from either side.
    assert first[-100:] == second[:100]


def test_ordinals_are_unique_and_sequential():
    pages = [Page(1, words(900)), Page(2, words(900))]
    chunks = chunk_pages(pages, size=300, overlap=50)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_and_whitespace_pages_are_dropped():
    pages = [Page(1, "   "), Page(2, "real content here"), Page(3, "")]
    chunks = chunk_pages(pages, size=100, overlap=10)
    assert [c.page for c in chunks] == [2]


def test_short_page_produces_exactly_one_chunk():
    chunks = chunk_pages([Page(3, words(10))], size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_overlap_must_be_smaller_than_size():
    # Equal values make the stride zero, which loops forever rather than failing.
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages([Page(1, words(50))], size=100, overlap=100)


def test_size_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        chunk_pages([Page(1, words(50))], size=0, overlap=0)


def test_no_words_are_lost_across_the_split():
    pages = [Page(1, words(730))]
    chunks = chunk_pages(pages, size=200, overlap=0)
    rebuilt = " ".join(c.text for c in chunks).split()
    assert rebuilt == pages[0].text.split()
