"""CLI for deterministic answer/citation evaluation of prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from qegs.io import read_jsonl

from .answer_citation import evaluate_answer_citation_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers, citations, and reviewed claim labels"
    )
    parser.add_argument("--qa", required=True, help="gold QA dataset JSONL")
    parser.add_argument(
        "--predictions", required=True, help="prediction JSONL (qursor-prediction-v1)"
    )
    parser.add_argument(
        "--retrieval-report",
        help="optional existing retrieval metrics JSON used only for Joint Success",
    )
    parser.add_argument(
        "--joint-cutoff",
        type=int,
        default=10,
        help="read complete_evidence_all_hops@K from the retrieval report (default: 10)",
    )
    parser.add_argument("--output", required=True, help="answer/citation report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        retrieval_report = None
        if args.retrieval_report:
            retrieval_report = json.loads(
                Path(args.retrieval_report).read_text(encoding="utf-8")
            )
        report = evaluate_answer_citation_dataset(
            read_jsonl(args.qa),
            read_jsonl(args.predictions),
            retrieval_report=retrieval_report,
            joint_cutoff=args.joint_cutoff,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        print(
            json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
