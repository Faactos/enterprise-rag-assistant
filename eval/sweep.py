"""Sweep retrieval configurations and emit a comparison table.

    python -m eval.sweep --dataset eval/datasets/sample.json --corpus data/corpus

Twenty-four configurations: chunk size x top-k x hybrid x reranking. Each chunk
size gets a freshly built index, because chunk size changes what is indexed.
"""
from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

from app.config import settings as base_settings
from app.service import RagService
from eval.harness import evaluate, index_corpus, isolated_settings, load_dataset
from llm_eval_toolkit import markdown_table, write_report

CHUNK_SIZES = (256, 512, 1024)
TOP_KS = (3, 5)
HYBRID = (True, False)
RERANK = (True, False)

NOTES = """Each row is a full pass over the gold set.

`refusal_acc` is scored over the answerable and unanswerable questions together:
a configuration that answers everything confidently scores badly here even when
its recall is perfect, which is the whole reason the unanswerable questions are
in the set.

Latency is end-to-end per question, including generation. Cost is estimated from
published per-token rates rather than billed, so configurations stay comparable
across providers.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/datasets/sample.json")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--workspace", default="data/sweep-workspace")
    parser.add_argument("--out", default="eval/results/sweep.md")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--chunk-sizes", type=int, nargs="*", default=list(CHUNK_SIZES))
    parser.add_argument("--top-ks", type=int, nargs="*", default=list(TOP_KS))
    args = parser.parse_args()

    questions = load_dataset(args.dataset)
    workspace = Path(args.workspace)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)

    combos = list(itertools.product(args.chunk_sizes, args.top_ks, HYBRID, RERANK))
    rows = []

    for i, (chunk, top_k, hybrid, rerank) in enumerate(combos, start=1):
        label = f"chunk={chunk} top_k={top_k} hybrid={hybrid} rerank={rerank}"
        print(f"[{i}/{len(combos)}] {label}")
        settings = isolated_settings(
            base_settings,
            workspace / f"c{chunk}-k{top_k}-h{int(hybrid)}-r{int(rerank)}",
            chunk_size=chunk,
            top_k=top_k,
            candidate_k=max(base_settings.candidate_k, top_k * 4),
            use_hybrid=hybrid,
            use_reranker=rerank,
        )
        service = RagService(settings)
        try:
            user_id = index_corpus(service, args.corpus)
            rows.append(
                evaluate(service, user_id, questions, retrieval_only=args.retrieval_only)
            )
        finally:
            # Release the embedded Qdrant file lock before the next iteration
            # opens its own store, or Windows refuses to clean the directory.
            service.store.client.close()

    print()
    print(markdown_table(rows))
    path = write_report(args.out, f"Retrieval sweep: {Path(args.dataset).name}", rows, NOTES)
    print(f"\nWrote {path} and its .json sibling")


if __name__ == "__main__":
    main()
