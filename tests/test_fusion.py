"""Reciprocal Rank Fusion: ranking behaviour and score rescaling."""
from __future__ import annotations

from app.retrieval.fusion import rrf
from app.retrieval.vector_store import Hit


def hit(cid: str, score: float = 0.5) -> Hit:
    return Hit(chunk_id=cid, doc_id="d1", page=1, text=cid, score=score)


def test_a_passage_found_by_both_retrievers_outranks_one_found_by_either():
    dense = [hit("a"), hit("b")]
    sparse = [hit("c"), hit("a")]
    fused = rrf([dense, sparse], k=60)
    assert fused[0].chunk_id == "a"


def test_scores_are_rescaled_so_the_top_hit_is_one():
    # An RRF sum is ~0.03; left unscaled, MIN_SCORE would refuse every answer.
    fused = rrf([[hit("a"), hit("b")], [hit("a")]], k=60)
    assert fused[0].score == 1.0
    assert all(0.0 < h.score <= 1.0 for h in fused)


def test_scores_are_monotonically_non_increasing():
    fused = rrf([[hit("a"), hit("b"), hit("c")], [hit("c"), hit("a")]], k=60)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_every_input_passage_survives_fusion():
    fused = rrf([[hit("a"), hit("b")], [hit("c")]], k=60)
    assert {h.chunk_id for h in fused} == {"a", "b", "c"}


def test_duplicates_are_collapsed_to_one_entry():
    fused = rrf([[hit("a")], [hit("a")], [hit("a")]], k=60)
    assert len(fused) == 1


def test_empty_rankings_fuse_to_nothing():
    assert rrf([[], []], k=60) == []
    assert rrf([], k=60) == []


def test_one_empty_ranking_does_not_discard_the_other():
    # Lexical search legitimately returns nothing for a query with no shared
    # tokens; that must not wipe out the dense results.
    fused = rrf([[hit("a"), hit("b")], []], k=60)
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_a_smaller_k_sharpens_the_advantage_of_rank_one():
    sharp = rrf([[hit("a"), hit("b")]], k=1)
    flat = rrf([[hit("a"), hit("b")]], k=1000)
    assert sharp[1].score < flat[1].score
