"""CLI for evidence-aware evaluation of a retrieval JSONL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .baseline import load_jsonl
from .metrics import evaluate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval run")
    parser.add_argument("--qa", required=True, help="QA dataset JSONL")
    parser.add_argument("--run", required=True, help="retrieval run JSONL")
    parser.add_argument("--qrels", help="optional qrels JSONL")
    parser.add_argument("--output", required=True, help="evaluation report JSON")
    parser.add_argument(
        "--cutoffs",
        default="1,5,10,20",
        help="comma-separated positive ranks (default: 1,5,10,20)",
    )
    parser.add_argument(
        "--min-span-coverage",
        type=float,
        default=0.5,
        help="fraction of each gold span that a returned span must cover",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cutoffs = tuple(int(value.strip()) for value in args.cutoffs.split(","))
    except ValueError as error:
        raise SystemExit("--cutoffs must contain comma-separated integers") from error
    report = evaluate_dataset(
        load_jsonl(args.qa),
        load_jsonl(args.run),
        qrels=load_jsonl(args.qrels) if args.qrels else (),
        cutoffs=cutoffs,
        min_span_coverage=args.min_span_coverage,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
