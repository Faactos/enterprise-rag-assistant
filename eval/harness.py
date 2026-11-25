"""Shared evaluation machinery: index a corpus, score a dataset.

The harness builds an isolated service per configuration -- its own SQLite file
and its own embedded Qdrant directory -- because the sweep varies CHUNK_SIZE,
and chunk size changes what is *in* the index. Reusing one index across chunk
sizes would measure nothing.
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from app.config import Settings  # noqa: E402
from app.generation.answerer import Answerer  # noqa: E402
from app.service import RagService  # noqa: E402
from llm_eval_toolkit import (  # noqa: E402
    Accumulator,
    check_citations,
    groundedness,
    mean,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    refusal_accuracy,
    timed,
)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    relevant: frozenset


def load_dataset(path: str | Path) -> list[Question]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Question(
            id=q["id"],
            question=q["question"],
            answerable=bool(q.get("answerable", True)),
            relevant=frozenset((r["doc"], int(r["page"])) for r in q.get("relevant", [])),
        )
        for q in body["questions"]
    ]


def isolated_settings(base: Settings, workspace: Path, **overrides) -> Settings:
    workspace.mkdir(parents=True, exist_ok=True)
    return replace(
        base,
        db_path=str(workspace / "meta.db"),
        qdrant_path=str(workspace / "qdrant"),
        upload_dir=str(workspace / "uploads"),
        **overrides,
    )


def index_corpus(service: RagService, corpus: str | Path) -> str:
    """Index every supported file under `corpus` for one throwaway user."""
    from app import auth

    user, _ = auth.register(f"eval-{uuid.uuid4().hex[:8]}@localhost")
    files = sorted(
        p
        for p in Path(corpus).iterdir()
        if p.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}
    )
    if not files:
        raise SystemExit(f"No documents found in {corpus}")
    for path in files:
        service.ingest(user.user_id, path, title=path.name)
    return user.user_id


def evaluate(
    service: RagService,
    user_id: str,
    questions: list[Question],
    *,
    retrieval_only: bool = False,
) -> dict:
    """Score one configuration. Returns a flat row ready for the report table."""
    s = service.settings
    k = s.top_k
    recalls, precisions, ndcgs, rrs = [], [], [], []
    groundedness_scores, citation_validity = [], []
    refused_flags, answerable_flags = [], []
    acc = Accumulator()

    answerer = None if retrieval_only else Answerer(s, service.llm)

    for q in questions:
        with timed() as elapsed:
            passages = service.retrieve(user_id, q.question)
            answer = None
            if answerer is not None:
                titles = {p.doc_id: p.title for p in passages}
                answer = answerer.answer(q.question, passages, titles)
        latency_ms = elapsed[0]

        retrieved = [(p.title, p.page) for p in passages]
        relevant = list(q.relevant)
        # Unanswerable questions have no relevant pages; including them in the
        # ranking averages would reward retrieving nothing, so they are scored
        # only by refusal accuracy.
        if q.answerable:
            recalls.append(recall_at_k(retrieved, relevant, k))
            precisions.append(precision_at_k(retrieved, relevant, k))
            ndcgs.append(ndcg_at_k(retrieved, relevant, k))
            rrs.append(mrr(retrieved, relevant))

        if answer is not None:
            refused_flags.append(answer.refused)
            answerable_flags.append(q.answerable)
            acc.record(
                latency_ms,
                answer.model or s.llm_model,
                answer.prompt_tokens,
                answer.completion_tokens,
            )
            if not answer.refused:
                groundedness_scores.append(
                    groundedness(answer.answer, [p.text for p in answer.passages])
                )
                citation_validity.append(
                    check_citations(answer.answer, len(answer.passages)).validity
                )
        else:
            acc.record(latency_ms, s.llm_model, 0, 0)

    row = {
        "chunk": s.chunk_size,
        "top_k": k,
        "hybrid": s.use_hybrid,
        "rerank": s.use_reranker,
        "recall@k": mean(recalls),
        "precision@k": mean(precisions),
        "nDCG@k": mean(ndcgs),
        "MRR": mean(rrs),
        "p50_ms": percentile(acc.latencies_ms, 50),
        "p95_ms": percentile(acc.latencies_ms, 95),
    }
    if not retrieval_only:
        row.update(
            {
                "groundedness": mean(groundedness_scores),
                "citation_validity": mean(citation_validity),
                "refusal_acc": refusal_accuracy(refused_flags, answerable_flags),
                "cost_usd": acc.cost_usd,
            }
        )
    return row
