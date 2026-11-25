"""Rank-aware retrieval metrics.

All of these take *retrieved* and *relevant* as opaque hashable ids -- the
toolkit never knows what a "page" is, which is what lets it be shared across
projects.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence

Id = Hashable


def recall_at_k(retrieved: Sequence[Id], relevant: Sequence[Id], k: int) -> float:
    if not relevant:
        return 1.0  # nothing to find; retrieving nothing is correct
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def precision_at_k(retrieved: Sequence[Id], relevant: Sequence[Id], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / len(top)


def mrr(retrieved: Sequence[Id], relevant: Sequence[Id]) -> float:
    """Reciprocal rank of the first relevant hit -- 0 if there is none."""
    wanted = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[Id], relevant: Sequence[Id], k: int) -> float:
    """Binary-gain nDCG: rewards putting the right passage *first*, not merely
    somewhere in the top k, which is what recall alone cannot distinguish."""
    wanted = set(relevant)
    if not wanted:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in wanted
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. Small eval sets make interpolation misleading."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[idx]
