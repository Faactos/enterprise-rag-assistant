"""Operational measurements: latency, token cost, and report rendering.

Quality metrics alone cannot choose a configuration. Reranking buys recall and
costs half a second per question; the sweep table only answers "is it worth it?"
if latency and cost sit in the same table as nDCG.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

# Groq free tier is unmetered, but the whole point of tracking cost is to make
# the comparison portable, so published per-million-token rates are used.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama3.1:8b": (0.0, 0.0),  # local
}
DEFAULT_PRICE = (0.15, 0.75)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = PRICES_PER_MTOK.get(model, DEFAULT_PRICE)
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000


@contextmanager
def timed() -> Iterator[list[float]]:
    """Yield a one-element list that receives the elapsed milliseconds.

    A list rather than a return value because the duration is only known after
    the block has run, and the caller wants it inside the same scope.
    """
    holder: list[float] = []
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder.append((time.perf_counter() - start) * 1000.0)


@dataclass
class Accumulator:
    """Running totals for one configuration under evaluation."""

    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def record(self, latency_ms: float, model: str, prompt: int, completion: int) -> None:
        self.latencies_ms.append(latency_ms)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += estimate_cost(model, prompt, completion)

    def fail(self, message: str) -> None:
        self.errors.append(message)


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Render rows as a GitHub markdown table, floats to three places."""
    if not rows:
        return "_no results_"
    columns = list(columns or rows[0].keys())

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(path: str | Path, title: str, rows: Sequence[dict[str, Any]], notes: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", "", markdown_table(rows)]
    if notes:
        parts += ["", "## Notes", "", notes]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    # The JSON sibling is what a later run diffs against; markdown is for humans.
    path.with_suffix(".json").write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    return path
