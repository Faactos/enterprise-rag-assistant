"""Deterministic grounding checks.

These are the ground truth of the eval. A judge model is evidence; a string
check against the retrieved text is a fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class CitationReport:
    markers: list[int]
    valid: list[int]
    invalid: list[int]

    @property
    def has_citations(self) -> bool:
        return bool(self.valid)

    @property
    def validity(self) -> float:
        """Share of emitted markers that point at a real passage."""
        if not self.markers:
            return 0.0
        return len(self.valid) / len(self.markers)


def check_citations(answer: str, passage_count: int) -> CitationReport:
    markers = [int(m) for m in MARKER.findall(answer)]
    valid = [m for m in markers if 1 <= m <= passage_count]
    invalid = [m for m in markers if not 1 <= m <= passage_count]
    return CitationReport(markers, valid, invalid)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def groundedness(answer: str, passages: Sequence[str], stopwords: set[str] | None = None) -> float:
    """Fraction of the answer's content words that appear in the context.

    A lexical proxy, deliberately: it cannot be gamed by a fluent paraphrase of
    something the documents never said, which is exactly the failure mode an
    LLM judge is worst at catching.
    """
    stop = stopwords or STOPWORDS
    answer_tokens = {t for t in _tokens(answer) if t not in stop and len(t) > 2}
    if not answer_tokens:
        return 1.0
    context = set()
    for p in passages:
        context |= _tokens(p)
    return len(answer_tokens & context) / len(answer_tokens)


def refusal_accuracy(refused: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Did the system decline exactly when it should have?

    Scored over answerable and unanswerable questions together -- a system that
    answers everything and one that refuses everything both score poorly, while
    recall alone would rank the first one perfect.
    """
    if not refused:
        return 0.0
    correct = sum(1 for r, a in zip(refused, answerable) if r != a)
    return correct / len(refused)


STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "has",
    "had", "was", "were", "with", "that", "this", "from", "they", "their", "have",
    "will", "would", "there", "been", "when", "which", "who", "whom", "what",
    "may", "must", "shall", "should", "per", "its", "his", "her", "them", "then",
    "than", "also", "into", "such", "each", "upon", "under", "over", "does", "did",
}
