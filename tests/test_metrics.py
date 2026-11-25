"""The shared eval toolkit: ranking metrics and deterministic grounding checks."""
from __future__ import annotations

import pytest

from llm_eval_toolkit import (
    check_citations,
    estimate_cost,
    groundedness,
    markdown_table,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    refusal_accuracy,
)

# -- ranking ---------------------------------------------------------------


def test_recall_counts_unique_relevant_items_found():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k(["a", "x", "y"], ["a", "c"], 3) == 0.5


def test_recall_respects_the_cutoff():
    # The relevant item is retrieved, but not within the top 2.
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_recall_of_an_unanswerable_question_is_one():
    # Nothing to find, so retrieving nothing is correct rather than a zero.
    assert recall_at_k([], [], 5) == 1.0


def test_precision_divides_by_the_retrieved_count():
    assert precision_at_k(["a", "x"], ["a"], 2) == 0.5
    assert precision_at_k([], ["a"], 5) == 0.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["a", "x"], ["a"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_rewards_ranking_the_right_passage_first():
    # This is what recall cannot see: both retrieve the answer, one buries it.
    first = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    last = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert first == 1.0
    assert last < first


def test_ndcg_is_bounded_and_ideal_when_perfectly_ordered():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert 0.0 <= ndcg_at_k(["x"], ["a"], 1) <= 1.0


def test_percentile_uses_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 100) == 40.0
    assert percentile([], 50) == 0.0


# -- grounding -------------------------------------------------------------


def test_citation_markers_are_split_into_valid_and_invalid():
    report = check_citations("As stated [1] and [9].", passage_count=3)
    assert report.valid == [1]
    assert report.invalid == [9]
    assert report.validity == 0.5


def test_an_answer_with_no_markers_has_no_citations():
    report = check_citations("25 days.", passage_count=3)
    assert not report.has_citations
    assert report.validity == 0.0


def test_groundedness_is_high_when_the_answer_reuses_the_context():
    score = groundedness(
        "Full-time employees receive twenty-five days of annual leave.",
        ["Full-time employees accrue twenty-five days of paid annual leave per year."],
    )
    # Not 1.0: "receive" paraphrases "accrue", and a lexical check is meant to
    # notice that. The threshold tolerates paraphrase, not invention.
    assert score > 0.85


def test_groundedness_falls_when_the_answer_introduces_new_claims():
    score = groundedness(
        "Employees receive unlimited sabbatical funding through the Zurich office.",
        ["Full-time employees accrue twenty-five days of paid annual leave."],
    )
    assert score < 0.5


def test_refusal_accuracy_scores_declining_exactly_when_it_should():
    # refused=True must line up with answerable=False.
    assert refusal_accuracy([False, True], [True, False]) == 1.0
    assert refusal_accuracy([True, True], [True, False]) == 0.5
    # A system that answers everything confidently scores zero here even if its
    # recall is perfect -- which is the point of keeping this metric.
    assert refusal_accuracy([False, False], [True, False]) == 0.5


# -- ops -------------------------------------------------------------------


def test_cost_scales_with_tokens_and_is_zero_for_a_local_model():
    assert estimate_cost("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("llama3.1:8b", 1_000_000, 1_000_000) == 0.0


def test_markdown_table_renders_a_header_and_rounds_floats():
    table = markdown_table([{"recall": 0.666666, "hybrid": True}])
    assert "| recall | hybrid |" in table
    assert "0.667" in table
    assert "yes" in table


def test_markdown_table_handles_no_rows():
    assert markdown_table([]) == "_no results_"
