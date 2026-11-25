"""LLM-as-judge scorers -- supplementary evidence, never the ground truth.

A judge shares the generator's blind spots and costs quota, so these are opt-in
and the deterministic checks in grounding.py always outrank them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

FAITHFULNESS_PROMPT = """You are grading one answer against the passages it was
built from. Answer ONLY with JSON: {"score": 0-1 float, "reason": "..."}.

score 1.0 = every claim is supported by the passages
score 0.0 = the answer asserts something the passages do not contain

Passages:
{passages}

Answer:
{answer}
"""

RELEVANCE_PROMPT = """Grade how well the answer addresses the question. Answer
ONLY with JSON: {"score": 0-1 float, "reason": "..."}.

Question: {question}

Answer: {answer}
"""


class Completer(Protocol):
    def complete(self, system: str, user: str): ...


@dataclass(frozen=True)
class Verdict:
    score: float
    reason: str


def _parse(text: str) -> Verdict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Verdict(0.0, f"unparseable judge output: {text[:120]!r}")
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Verdict(0.0, f"invalid judge JSON: {exc}")
    try:
        score = float(body.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return Verdict(max(0.0, min(1.0, score)), str(body.get("reason", "")))


def judge_faithfulness(llm: Completer, answer: str, passages: list[str]) -> Verdict:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    prompt = FAITHFULNESS_PROMPT.replace("{passages}", numbered).replace("{answer}", answer)
    return _parse(llm.complete("You are a strict grader. Output JSON only.", prompt).text)


def judge_relevance(llm: Completer, question: str, answer: str) -> Verdict:
    prompt = RELEVANCE_PROMPT.replace("{question}", question).replace("{answer}", answer)
    return _parse(llm.complete("You are a strict grader. Output JSON only.", prompt).text)
