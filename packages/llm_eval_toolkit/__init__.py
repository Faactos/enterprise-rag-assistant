"""Reusable LLM evaluation primitives.

Deliberately dependency-free and ignorant of this project: it takes ids, strings
and numbers, so the same module can score a different retrieval system.
"""
from .grounding import CitationReport, check_citations, groundedness, refusal_accuracy
from .ops import Accumulator, estimate_cost, markdown_table, timed, write_report
from .retrieval_metrics import mean, mrr, ndcg_at_k, percentile, precision_at_k, recall_at_k

__all__ = [
    "CitationReport",
    "check_citations",
    "groundedness",
    "refusal_accuracy",
    "Accumulator",
    "estimate_cost",
    "markdown_table",
    "timed",
    "write_report",
    "mean",
    "mrr",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
]
