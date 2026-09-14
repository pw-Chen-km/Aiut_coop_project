"""Command-line entry point for deterministic QEGS corpus preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .chunking import chunk_pages
from .evidence import build_evidence_candidates
from .ids import make_document_id, slugify
from .io import canonical_json, read_jsonl, write_json, write_jsonl
from .manifest import build_manifest, sha256_path, verify_manifest
from .sources import SourceExtractionError, read_pages
from .validation import DatasetQuota, SchemaValidationError, validate_dataset, validate_record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qegs",
        description="Build and validate QURSOR evidence-grounded RAG datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build documents, chunks, and evidence candidates")
    build.add_argument("--input", required=True, type=Path, help="PDF, page JSON(L), text, or page directory")
    build.add_argument("--out-dir", required=True, type=Path)
    build.add_argument("--family", required=True, help="stable document family name")
    build.add_argument("--document-version", required=True)
    build.add_argument("--title", required=True)
    build.add_argument("--product", required=True)
    build.add_argument("--doc-type", default="manual")
    build.add_argument("--language", default="en")
    build.add_argument("--dataset-version", default="qursor-rag-p1.0")
    build.add_argument("--effective-date")
    build.add_argument("--access-scope", default="internal")
    build.add_argument("--product-surface", action="append", dest="product_surfaces")
    build.add_argument(
        "--rights-classification",
        default="confidential",
        choices=["public", "internal", "confidential", "restricted"],
    )
    build.add_argument("--rights-owner", default="Aiut")
    build.add_argument("--source-provider", default="Aiut")
    build.add_argument("--source-message-id")
    build.add_argument("--allowed-role", action="append", dest="allowed_roles")
    build.add_argument("--canonical-lineage")
    build.add_argument("--duplicate-lineage", action="append", default=[])
    build.add_argument(
        "--allow-public-upload",
        action="store_true",
        help="explicitly mark source rights as allowing public upload (default: false)",
    )
    build.add_argument("--chunk-schema-version", default="ckv1")
    build.add_argument("--max-chars", type=int, default=2400)
    build.add_argument("--overlap-chars", type=int, default=240)
    build.add_argument("--min-evidence-chars", type=int, default=24)
    build.set_defaults(handler=_command_build)

    validate = subparsers.add_parser("validate", help="validate JSONL records and cross-record references")
    validate.add_argument("--documents", required=True, type=Path)
    validate.add_argument("--chunks", required=True, type=Path)
    validate.add_argument("--qa", type=Path)
    validate.add_argument("--qrels", type=Path)
    validate.add_argument("--source-atoms", type=Path)
    validate.add_argument("--atom-chunk-map", type=Path)
    validate.add_argument("--atom-qrels", type=Path)
    validate.add_argument("--quota", type=Path, help="quota JSON matching schemas/quota.schema.json")
    validate.set_defaults(handler=_command_validate)

    record = subparsers.add_parser("validate-record", help="validate one JSON object")
    record.add_argument("--kind", required=True, choices=["document", "chunk", "qa", "qrel", "evidence"])
    record.add_argument("path", type=Path)
    record.set_defaults(handler=_command_validate_record)

    manifest = subparsers.add_parser("verify-manifest", help="verify a content-addressed manifest")
    manifest.add_argument("--root", required=True, type=Path)
    manifest.add_argument("--manifest", required=True, type=Path)
    manifest.set_defaults(handler=_command_verify_manifest)
    return parser


def _command_build(args: argparse.Namespace) -> int:
    pages, extraction_method = read_pages(args.input)
    if not pages:
        raise SourceExtractionError("input produced no non-empty pages")
    source_hash = sha256_path(args.input)
    doc_id = make_document_id(args.family, args.document_version, source_hash)
    document = {
        "dataset_version": args.dataset_version,
        "doc_id": doc_id,
        "doc_family_id": f"qursor:{slugify(args.family)}",
        "title": args.title,
        "product": args.product,
        "doc_type": args.doc_type,
        "language": args.language,
        "document_version": args.document_version,
        "source_filename": args.input.name,
        "source_sha256": source_hash,
        "page_count": len(pages),
        "extraction_method": extraction_method,
        "access_scope": args.access_scope,
        "product_surfaces": args.product_surfaces or [args.product],
        "rights": {
            "classification": args.rights_classification,
            "owner": args.rights_owner,
            "public_upload_allowed": bool(args.allow_public_upload),
        },
        "canonical_lineage": args.canonical_lineage or f"qursor:{slugify(args.family)}",
        "duplicate_lineage": args.duplicate_lineage,
        "source_provider": args.source_provider,
        "permissions": {
            "internal_processing_allowed": True,
            "public_upload_allowed": bool(args.allow_public_upload),
            "allowed_roles": args.allowed_roles or [],
        },
    }
    if args.effective_date is not None:
        document["effective_date"] = args.effective_date
    if args.source_message_id is not None:
        document["source_message_id"] = args.source_message_id
    chunks, blocks = chunk_pages(
        pages,
        doc_id,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        schema_version=args.chunk_schema_version,
    )
    if not chunks:
        raise SourceExtractionError("input produced no chunks")
    evidence = build_evidence_candidates(
        document,
        chunks,
        blocks,
        min_chars=args.min_evidence_chars,
    )
    validate_record("document", document)
    for index, chunk in enumerate(chunks):
        validate_record("chunk", chunk, path=f"$.chunks[{index}]")
    for index, item in enumerate(evidence):
        validate_record("evidence", item, path=f"$.evidence[{index}]")
    validate_dataset(documents=[document], chunks=chunks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    documents_path = args.out_dir / "documents.jsonl"
    chunks_path = args.out_dir / "chunks.jsonl"
    evidence_path = args.out_dir / "evidence_candidates.jsonl"
    write_jsonl(documents_path, [document])
    write_jsonl(chunks_path, chunks)
    write_jsonl(evidence_path, evidence)
    manifest = build_manifest(
        [documents_path, chunks_path, evidence_path],
        dataset_version=args.dataset_version,
        base_dir=args.out_dir,
    )
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        canonical_json(
            {
                "doc_id": doc_id,
                "documents": 1,
                "chunks": len(chunks),
                "evidence_candidates": len(evidence),
                "dataset_sha256": manifest["dataset_sha256"],
            }
        )
    )
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    documents = read_jsonl(args.documents)
    chunks = read_jsonl(args.chunks)
    qa = read_jsonl(args.qa) if args.qa else []
    qrels = read_jsonl(args.qrels) if args.qrels else []
    source_atoms = read_jsonl(args.source_atoms) if args.source_atoms else []
    atom_chunk_map = read_jsonl(args.atom_chunk_map) if args.atom_chunk_map else []
    atom_qrels = read_jsonl(args.atom_qrels) if args.atom_qrels else []
    quota = None
    if args.quota:
        value = json.loads(args.quota.read_text(encoding="utf-8"))
        quota = DatasetQuota.from_mapping(value)
    issues = validate_dataset(
        documents=documents,
        chunks=chunks,
        qa=qa,
        qrels=qrels,
        source_atoms=source_atoms,
        atom_chunk_map=atom_chunk_map,
        atom_qrels=atom_qrels,
        quota=quota,
        raise_on_error=False,
    )
    if issues:
        print(
            canonical_json(
                {
                    "valid": False,
                    "issue_count": len(issues),
                    "issues": [
                        {"code": issue.code, "path": issue.path, "message": issue.message}
                        for issue in issues
                    ],
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(
        canonical_json(
            {
                "valid": True,
                "documents": len(documents),
                "chunks": len(chunks),
                "qa": len(qa),
                "qrels": len(qrels),
                "source_atoms": len(source_atoms),
                "atom_chunk_mappings": len(atom_chunk_map),
                "atom_qrels": len(atom_qrels),
            }
        )
    )
    return 0


def _command_validate_record(args: argparse.Namespace) -> int:
    value = json.loads(args.path.read_text(encoding="utf-8"))
    issues = validate_record(args.kind, value, raise_on_error=False)
    if issues:
        print(canonical_json({"valid": False, "issues": [str(issue) for issue in issues]}), file=sys.stderr)
        return 2
    print(canonical_json({"valid": True, "kind": args.kind}))
    return 0


def _command_verify_manifest(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify_manifest(args.root, manifest)
    if errors:
        print(canonical_json({"valid": False, "errors": errors}), file=sys.stderr)
        return 2
    print(canonical_json({"valid": True, "dataset_sha256": manifest.get("dataset_sha256")}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (SchemaValidationError, SourceExtractionError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(canonical_json({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
