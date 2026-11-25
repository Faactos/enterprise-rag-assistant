"""The three grounding gates, with a stubbed LLM.

What is under test is the *gate logic*: which model outputs become answers,
which become refusals, and what survives into the citation list.
"""
from __future__ import annotations

from dataclasses import replace

from app.config import settings as base_settings
from app.generation.answerer import INSUFFICIENT, Answerer, build_prompt
from app.generation.llm import Completion
from app.retrieval.retriever import Passage

SETTINGS = replace(base_settings, min_score=0.25, top_k=5)


class FixedLLM:
    """Returns the same completion however many times it is called."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        self.last_prompt = user
        return Completion(text=self.text, model="stub", prompt_tokens=10, completion_tokens=5)


class SequenceLLM:
    """Replays a list of replies -- for exercising the citation retry."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        self.prompts.append(user)
        text = self.texts.pop(0) if self.texts else ""
        return Completion(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


def passage(score: float, page: int = 1, text: str = "Full-time employees get 25 days.") -> Passage:
    return Passage(
        chunk_id=f"c{page}",
        doc_id="d1",
        title="leave-policy.md",
        page=page,
        text=text,
        score=score,
    )


def ask(llm, passages, question="How many leave days?"):
    return Answerer(SETTINGS, llm).answer(question, passages, {"d1": "leave-policy.md"})


# -- gate 1: score floor ---------------------------------------------------


def test_no_passages_refuses_without_calling_the_model():
    llm = FixedLLM("25 days [1].")
    out = ask(llm, [])
    assert out.refused and out.reason == "no_passages"
    assert llm.calls == 0


def test_weak_context_refuses_before_spending_a_completion():
    llm = FixedLLM("25 days [1].")
    out = ask(llm, [passage(0.10), passage(0.05, page=2)])
    assert out.refused
    assert out.reason.startswith("below_min_score")
    assert llm.calls == 0  # the point of the gate is to refuse *cheaply*


def test_passages_below_the_floor_are_excluded_from_the_prompt():
    llm = FixedLLM("25 days [1].")
    ask(llm, [passage(0.9, page=1), passage(0.01, page=2)])
    assert "page 1" in llm.last_prompt
    assert "page 2" not in llm.last_prompt


# -- gate 2: citations required --------------------------------------------


def test_an_answer_with_a_valid_citation_is_returned():
    out = ask(FixedLLM("Full-time employees receive 25 days [1]."), [passage(0.9)])
    assert not out.refused
    assert [c.marker for c in out.citations] == [1]
    assert out.citations[0].page == 1
    assert out.citations[0].title == "leave-policy.md"


def test_the_model_declining_is_passed_through_as_a_refusal():
    out = ask(FixedLLM(INSUFFICIENT), [passage(0.9)])
    assert out.refused and out.reason == "model_declined"


def test_an_uncited_answer_triggers_exactly_one_retry():
    # The model intermittently writes a correct answer and forgets the markers.
    llm = SequenceLLM("Full-time employees receive 25 days.", "25 days [1].")
    out = ask(llm, [passage(0.9)])
    assert llm.calls == 2
    assert not out.refused
    assert out.answer == "25 days [1]."


def test_the_retry_prompt_asks_for_markers_and_keeps_the_same_passages():
    llm = SequenceLLM("25 days.", "25 days [1].")
    ask(llm, [passage(0.9)])
    retry_prompt = llm.prompts[1]
    assert "square brackets" in retry_prompt
    # Same evidence -- the retry must not be able to widen the context.
    assert llm.prompts[0] in retry_prompt


def test_a_retry_that_also_omits_citations_refuses():
    llm = SequenceLLM("25 days.", "Still 25 days.")
    out = ask(llm, [passage(0.9)])
    assert llm.calls == 2
    assert out.refused and out.reason == "no_valid_citations"


def test_a_retry_may_legitimately_decline():
    # The retry must never be able to manufacture evidence to satisfy the gate.
    llm = SequenceLLM("Probably 25 days.", INSUFFICIENT)
    out = ask(llm, [passage(0.9)])
    assert out.refused and out.reason == "no_valid_citations"


def test_a_cited_first_answer_never_retries():
    llm = SequenceLLM("25 days [1].", "should not be used")
    out = ask(llm, [passage(0.9)])
    assert llm.calls == 1
    assert out.answer == "25 days [1]."


# -- gate 3: marker range validation ---------------------------------------


def test_a_marker_beyond_the_passage_count_is_dropped():
    out = ask(FixedLLM("Per the handbook [1], and also [9]."), [passage(0.9)])
    assert [c.marker for c in out.citations] == [1]


def test_an_answer_citing_only_an_invented_marker_is_refused():
    llm = SequenceLLM("See [9].", "Still [9].")
    out = ask(llm, [passage(0.9)])
    assert out.refused and out.reason == "no_valid_citations"


def test_repeated_markers_produce_one_citation_each():
    passages = [passage(0.9, page=1), passage(0.8, page=2)]
    out = ask(FixedLLM("First [1], second [2], first again [1]."), passages)
    assert [c.marker for c in out.citations] == [1, 2]
    assert [c.page for c in out.citations] == [1, 2]


def test_marker_zero_is_not_a_passage():
    llm = SequenceLLM("See [0].", "See [0] again.")
    out = ask(llm, [passage(0.9)])
    assert out.refused


# -- prompt construction ---------------------------------------------------


def test_the_prompt_numbers_passages_from_one_and_names_the_page():
    prompt = build_prompt("q?", [passage(0.9, page=4), passage(0.8, page=7)])
    assert "[1] (source: leave-policy.md, page 4)" in prompt
    assert "[2] (source: leave-policy.md, page 7)" in prompt
    assert "Question: q?" in prompt


def test_usage_is_reported_on_the_answer():
    out = ask(FixedLLM("25 days [1]."), [passage(0.9)])
    assert (out.prompt_tokens, out.completion_tokens) == (10, 5)
    assert out.model == "stub"
