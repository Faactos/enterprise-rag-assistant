"""Evaluate one configuration against a gold dataset.

    python -m eval.run_eval --dataset eval/datasets/sample.json
    python -m eval.run_eval --dataset ... --retrieval-only   # no LLM, no key
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings as base_settings
from app.service import RagService
from eval.harness import evaluate, index_corpus, isolated_settings, load_dataset
from llm_eval_toolkit import markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/datasets/sample.json")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--workspace", default="data/eval-workspace")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retrieval metrics only -- no LLM call, no API key, no quota spent.",
    )
    parser.add_argument("--out", default="eval/results/run.md")
    args = parser.parse_args()

    questions = load_dataset(args.dataset)
    settings = isolated_settings(base_settings, Path(args.workspace))
    service = RagService(settings)

    print(f"Indexing {args.corpus} ...")
    user_id = index_corpus(service, args.corpus)

    mode = "retrieval only" if args.retrieval_only else "full pipeline"
    print(f"Scoring {len(questions)} questions ({mode}) ...")
    row = evaluate(service, user_id, questions, retrieval_only=args.retrieval_only)

    print()
    print(markdown_table([row]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"# Evaluation: {Path(args.dataset).name}"
    out.write_text(f"{title}\n\n{markdown_table([row])}\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
