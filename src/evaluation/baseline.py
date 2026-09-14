"""Command-line runner for three dependency-free local retrieval baselines.

Example::

    python -m evaluation.baseline --method all \
      --corpus data/processed/chunks.jsonl \
      --queries data/dataset/dev.jsonl \
      --qrels data/dataset/qrels-dev.jsonl \
      --output artifacts/evaluation/runs \
      --report-dir artifacts/evaluation/reports \
      --top-k 20

Methods are ``bm25``, deterministic ``dense-hash``, and
``hybrid-rerank`` (RRF plus a non-neural lexical pair scorer).  The latter two
are smoke baselines, not substitutes for a versioned embedding model and neural
reranker.  Every emitted item includes ``source_spans`` and follows the project
retrieval-run schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import evaluate_dataset
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    Document,
    HashingDenseEncoder,
    HybridRetriever,
    LexicalRerankingRetriever,
    Retriever,
)


BASELINE_METHODS = ("bm25", "dense-hash", "hybrid-rerank")


def build_bm25_run(
    corpus_records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 20,
    chunk_id_key: str = "chunk_id",
    doc_id_key: str = "doc_id",
    text_key: str = "text",
    qid_key: str = "qid",
    question_key: str = "question",
    k1: float = 1.2,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Build an in-memory BM25 index and return a deterministic run."""

    return build_baseline_run(
        corpus_records,
        query_records,
        method="bm25",
        top_k=top_k,
        chunk_id_key=chunk_id_key,
        doc_id_key=doc_id_key,
        text_key=text_key,
        qid_key=qid_key,
        question_key=question_key,
        k1=k1,
        b=b,
    )


def build_baseline_run(
    corpus_records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    top_k: int = 20,
    chunk_id_key: str = "chunk_id",
    doc_id_key: str = "doc_id",
    text_key: str = "text",
    qid_key: str = "qid",
    question_key: str = "question",
    k1: float = 1.2,
    b: float = 0.75,
    hash_dimensions: int = 512,
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
    rrf_k: int = 60,
    candidate_multiplier: int = 4,
) -> list[dict[str, Any]]:
    """Run one named deterministic baseline against project-schema records."""

    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline method {method!r}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    documents = _build_documents(
        corpus_records,
        chunk_id_key=chunk_id_key,
        doc_id_key=doc_id_key,
        text_key=text_key,
    )
    retriever = _build_retriever(
        documents,
        method=method,
        k1=k1,
        b=b,
        hash_dimensions=hash_dimensions,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
        candidate_multiplier=candidate_multiplier,
    )
    return _run_queries(
        retriever,
        query_records,
        top_k=top_k,
        qid_key=qid_key,
        question_key=question_key,
    )


def _build_documents(
    corpus_records: Sequence[Mapping[str, Any]],
    *,
    chunk_id_key: str,
    doc_id_key: str,
    text_key: str,
) -> list[Document]:
    documents: list[Document] = []

    for record in corpus_records:
        if chunk_id_key not in record:
            raise ValueError(f"corpus record is missing {chunk_id_key!r}")
        if text_key not in record:
            raise ValueError(f"corpus record is missing {text_key!r}")
        reserved = {chunk_id_key, doc_id_key, text_key}
        documents.append(
            Document(
                chunk_id=str(record[chunk_id_key]),
                doc_id=(
                    str(record[doc_id_key])
                    if record.get(doc_id_key) is not None
                    else None
                ),
                text=str(record[text_key]),
                metadata={key: value for key, value in record.items() if key not in reserved},
            )
        )
    return documents


def _build_retriever(
    documents: Sequence[Document],
    *,
    method: str,
    k1: float,
    b: float,
    hash_dimensions: int,
    lexical_weight: float,
    dense_weight: float,
    rrf_k: int,
    candidate_multiplier: int,
) -> Retriever:
    bm25 = BM25Retriever(documents, k1=k1, b=b)
    if method == "bm25":
        return bm25
    dense = DenseRetriever(documents, HashingDenseEncoder(hash_dimensions))
    if method == "dense-hash":
        return dense
    hybrid = HybridRetriever(
        bm25,
        dense,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
        # The outer pair reranker expands again; keep the inner pool bounded.
        candidate_multiplier=2,
    )
    return LexicalRerankingRetriever(
        hybrid, candidate_multiplier=candidate_multiplier
    )


def _run_queries(
    retriever: Retriever,
    query_records: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    qid_key: str,
    question_key: str,
) -> list[dict[str, Any]]:
    run = []
    seen_qids: set[str] = set()
    for record in query_records:
        if qid_key not in record or question_key not in record:
            raise ValueError(
                f"query records require {qid_key!r} and {question_key!r}"
            )
        qid = str(record[qid_key])
        if qid in seen_qids:
            raise ValueError(f"duplicate query id {qid!r}")
        seen_qids.add(qid)
        results = retriever.search(str(record[question_key]), k=top_k)
        run.append({"qid": qid, "items": [item.as_run_item() for item in results]})
    return run


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON objects from JSONL with useful line-number errors."""

    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            result.append(value)
    return result


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write a run atomically so an interrupted baseline does not look complete."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(output)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write an indented UTF-8 JSON report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dependency-free BM25, dense-hash, and hybrid-rerank baselines"
    )
    parser.add_argument(
        "--method",
        choices=(*BASELINE_METHODS, "all"),
        default="bm25",
        help="baseline method; 'all' writes one run per method",
    )
    parser.add_argument("--corpus", required=True, help="chunk corpus JSONL")
    parser.add_argument("--queries", required=True, help="QA/query JSONL")
    parser.add_argument(
        "--output",
        required=True,
        help="run JSONL for one method, or output directory with --method all",
    )
    parser.add_argument("--qrels", help="qrels JSONL used for optional reports")
    parser.add_argument(
        "--report-dir",
        help="if set, evaluate each run and write METHOD.metrics.json here",
    )
    parser.add_argument("--cutoffs", default="1,5,10,20")
    parser.add_argument("--min-span-coverage", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-id-key", default="chunk_id")
    parser.add_argument("--doc-id-key", default="doc_id")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--qid-key", default="qid")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--hash-dimensions", type=int, default=512)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)
    qrels = load_jsonl(args.qrels) if args.qrels else []
    try:
        cutoffs = tuple(int(value.strip()) for value in args.cutoffs.split(","))
    except ValueError as error:
        raise SystemExit("--cutoffs must contain comma-separated integers") from error
    methods = BASELINE_METHODS if args.method == "all" else (args.method,)
    if args.method == "all":
        output_directory = Path(args.output)
        if output_directory.suffix:
            raise SystemExit("--output must be a directory when --method all is used")
        output_directory.mkdir(parents=True, exist_ok=True)

    for method in methods:
        run = build_baseline_run(
            corpus,
            queries,
            method=method,
            top_k=args.top_k,
            chunk_id_key=args.chunk_id_key,
            doc_id_key=args.doc_id_key,
            text_key=args.text_key,
            qid_key=args.qid_key,
            question_key=args.question_key,
            k1=args.k1,
            b=args.b,
            hash_dimensions=args.hash_dimensions,
            lexical_weight=args.lexical_weight,
            dense_weight=args.dense_weight,
            rrf_k=args.rrf_k,
            candidate_multiplier=args.candidate_multiplier,
        )
        run_path = (
            Path(args.output) / f"{method}.jsonl"
            if args.method == "all"
            else Path(args.output)
        )
        write_jsonl(run_path, run)
        if args.report_dir:
            report = evaluate_dataset(
                queries,
                run,
                qrels=qrels,
                cutoffs=cutoffs,
                min_span_coverage=args.min_span_coverage,
            )
            report["baseline"] = _baseline_metadata(method, args)
            write_json(Path(args.report_dir) / f"{method}.metrics.json", report)
    return 0


def _baseline_metadata(method: str, args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": method,
        "non_neural_local_baseline": method != "bm25",
        "top_k": args.top_k,
        "min_source_span_coverage": args.min_span_coverage,
    }
    if method in {"bm25", "hybrid-rerank"}:
        metadata["bm25"] = {"k1": args.k1, "b": args.b}
    if method in {"dense-hash", "hybrid-rerank"}:
        metadata["dense_hash"] = {
            "dimensions": args.hash_dimensions,
            "features": "unicode tokens + character 3-5 grams",
            "learned_semantics": False,
        }
    if method == "hybrid-rerank":
        metadata["hybrid"] = {
            "fusion": "weighted_rrf",
            "lexical_weight": args.lexical_weight,
            "dense_weight": args.dense_weight,
            "rrf_k": args.rrf_k,
        }
        metadata["reranker"] = {
            "type": "non-neural lexical query-document pair scorer",
            "candidate_multiplier": args.candidate_multiplier,
        }
    return metadata


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "BASELINE_METHODS",
    "build_baseline_run",
    "build_bm25_run",
    "load_jsonl",
    "main",
    "write_json",
    "write_jsonl",
]
