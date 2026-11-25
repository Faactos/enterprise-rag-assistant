"""Prompt construction, citation resolution, and the refusal gates.

Three gates keep an answer traceable; all three can only *reduce* what is
returned, never add to it:

1. score floor  -- refuse before spending a token on weak context
2. citations    -- an answer with no resolvable [n] is not an answer
3. marker range -- a [9] against five passages is dropped, not linked
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Settings
from ..retrieval.retriever import Passage
from .llm import Completion

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM = (
    "You answer strictly from the numbered passages supplied by the user. "
    "Cite every claim with the passage number in square brackets, like [1] or [2][3]. "
    "Never cite a number that is not in the list. "
    f"If the passages do not contain the answer, reply with exactly {INSUFFICIENT} "
    "and nothing else. Do not use outside knowledge."
)

RETRY_SUFFIX = (
    "\n\nYour previous reply contained no passage citation. Answer again using only "
    "the passages above, and put the passage number in square brackets after each "
    "claim, for example [1]. If the passages genuinely do not contain the answer, "
    f"reply with exactly {INSUFFICIENT}."
)

_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Citation:
    marker: int
    doc_id: str
    title: str
    page: int
    chunk_id: str

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    refused: bool = False
    reason: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "refused": self.refused,
            "reason": self.reason,
            "citations": [c.to_dict() for c in self.citations],
            "passages": [p.to_dict() for p in self.passages],
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = [
        f"[{i}] (source: {p.title}, page {p.page})\n{p.text}"
        for i, p in enumerate(passages, start=1)
    ]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def _resolve_citations(
    text: str, passages: list[Passage], titles: dict[str, str]
) -> list[Citation]:
    """Markers that point at a real passage, deduplicated, in first-use order.

    Gate 3 lives here: a marker outside 1..len(passages) is silently dropped
    rather than rendered as a broken source link.
    """
    seen: dict[int, Citation] = {}
    for raw in _MARKER.findall(text):
        marker = int(raw)
        if not 1 <= marker <= len(passages):
            continue
        if marker in seen:
            continue
        p = passages[marker - 1]
        seen[marker] = Citation(
            marker=marker,
            doc_id=p.doc_id,
            title=titles.get(p.doc_id, p.title),
            page=p.page,
            chunk_id=p.chunk_id,
        )
    return [seen[m] for m in sorted(seen)]


class Answerer:
    def __init__(self, settings: Settings, llm):
        self.settings = settings
        self.llm = llm

    def answer(
        self, question: str, passages: list[Passage], titles: dict[str, str] | None = None
    ) -> Answer:
        titles = titles or {}

        if not passages:
            return Answer(
                answer="I could not find anything relevant in the documents you have access to.",
                refused=True,
                reason="no_passages",
            )

        # Gate 1: refuse on weak context before paying for a completion.
        usable = [p for p in passages if p.score >= self.settings.min_score]
        if not usable:
            best = max(p.score for p in passages)
            return Answer(
                answer="The documents do not contain enough relevant information to answer that.",
                passages=passages,
                refused=True,
                reason=(
                    f"below_min_score (best {best:.2f} < {self.settings.min_score:.2f})"
                ),
            )

        prompt = build_prompt(question, usable)
        completion: Completion = self.llm.complete(SYSTEM, prompt)
        text = completion.text.strip()

        if text.upper().startswith(INSUFFICIENT):
            return Answer(
                answer="The documents do not contain enough information to answer that.",
                passages=usable,
                refused=True,
                reason="model_declined",
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        citations = _resolve_citations(text, usable, titles)

        # Gate 2, with one retry. The model intermittently writes a correct
        # answer and simply forgets the markers; discarding it outright turned
        # good retrieval into a refusal. The retry cannot invent evidence -- it
        # sees the same passages and may still decline.
        if not citations:
            retry = self.llm.complete(SYSTEM, f"{prompt}{RETRY_SUFFIX}")
            retry_text = retry.text.strip()
            if not retry_text.upper().startswith(INSUFFICIENT):
                retry_citations = _resolve_citations(retry_text, usable, titles)
                if retry_citations:
                    text, citations, completion = retry_text, retry_citations, retry

        if not citations:
            return Answer(
                answer=(
                    "I could not produce an answer that cites the retrieved passages, "
                    "so I am declining rather than risking an ungrounded one."
                ),
                passages=usable,
                refused=True,
                reason="no_valid_citations",
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        return Answer(
            answer=text,
            citations=citations,
            passages=usable,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )
