"""Build an immutable Docling + E5 corpus without a generative LLM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pypdfium2 as pdfium


REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "offline" / "src"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from doc2skill.config import sha256  # noqa: E402
from doc2skill.parsing import parse_pdf  # noqa: E402
from doc2skill.storage import Store  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(sha256(child).encode())
    return digest.hexdigest()


def pdf_info(path: Path) -> dict[str, Any]:
    pdf = pdfium.PdfDocument(str(path))
    page_chars = []
    for index in range(len(pdf)):
        page = pdf[index]
        text_page = page.get_textpage()
        text = text_page.get_text_range().strip()
        page_chars.append(len(text))
        text_page.close()
        page.close()
    pages = len(pdf)
    pdf.close()
    return {"pages": pages, "page_text_chars": page_chars,
            "low_text_pages": [i + 1 for i, count in enumerate(page_chars) if count < 80]}


def source_registry(source_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(source_dir.glob("*.pdf"), key=lambda item: item.name.casefold()):
        digest = sha256(path)
        info = pdf_info(path)
        rows.append({
            "doc_id": "qursor:doc:" + digest[:24],
            "title": path.stem.replace("_", " "),
            "source_filename": path.name,
            "source_sha256": digest,
            "bytes": path.stat().st_size,
            "page_count": info["pages"],
            "expected_page_count": info["pages"],
            "document_version": path.stem,
            "page_text_chars": info["page_text_chars"],
            "low_text_pages": info["low_text_pages"],
        })
    if not rows:
        raise ValueError(f"no PDFs found in {source_dir}")
    return rows


class LocalE5:
    def __init__(self, model_dir: Path, *, threads: int = 4, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_dir = model_dir.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False).to("cpu").eval()
        self.batch_size = batch_size
        torch.set_num_threads(threads)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        import torch
        arrays = []
        for offset in range(0, len(texts), self.batch_size):
            prepared = [text if text.startswith("passage: ") else "passage: " + text
                        for text in texts[offset:offset + self.batch_size]]
            if any(len(self.tokenizer.encode(text, add_special_tokens=True)) > 512 for text in prepared):
                raise ValueError("chunk exceeds the E5 512-token limit")
            batch = self.tokenizer(prepared, padding=True, truncation=False, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                arrays.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(arrays).astype(np.float32)


def coverage(corpus: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    chunks = corpus["chunks"]
    spans_by_block: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        for span in chunk.get("source_spans", []):
            spans_by_block.setdefault(span["block_id"], []).append(
                (int(span["source_start_char"]), int(span["source_end_char"])))
    failures = []
    checked = 0
    for block in corpus["blocks"]:
        if block.get("build_role", "content") != "content" or not block.get("text", "").strip():
            continue
        checked += 1
        intervals = sorted(spans_by_block.get(block["block_id"], []))
        cursor = 0
        for start, end in intervals:
            if start > cursor and block["text"][cursor:start].strip():
                failures.append({"block_id": block["block_id"], "reason": "uncovered_text", "from": cursor, "to": start})
            cursor = max(cursor, end)
        if cursor < len(block["text"]) and block["text"][cursor:].strip():
            failures.append({"block_id": block["block_id"], "reason": "uncovered_text", "from": cursor, "to": len(block["text"])})
    return {"schema_version": "enterprise-coverage-report-v1", "content_blocks_checked": checked,
            "chunks": len(chunks), "failures": failures, "valid": not failures}


def build(source_dir: Path, output_root: Path, model_dir: Path, version: str | None = None) -> Path:
    source_dir, output_root, model_dir = source_dir.resolve(), output_root.resolve(), model_dir.resolve()
    registry = source_registry(source_dir)
    embedding_hash = tree_hash(model_dir)
    identity = {"sources": [(row["source_filename"], row["source_sha256"]) for row in registry],
                "parser": {"docling": True, "ocr": False, "heading_hierarchy": True},
                "chunking": {"max_tokens": 384, "exact_spans": True},
                "retrieval_projection": "document-title-filename-section-path-v1",
                "embedding_sha256": embedding_hash}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    corpus_version = version or f"qursor-{fingerprint[:16]}"
    target = output_root / corpus_version
    if target.exists():
        raise FileExistsError(f"immutable corpus version already exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = LocalE5(model_dir)
    config = {
        "parser": {"device": "cpu", "threads": 4, "do_ocr": False,
                   "heading_hierarchy": True, "generate_parsed_pages": True},
        "chunking": {"max_tokens": 384},
        "embedding": {"model": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3"},
    }
    corpus = {name: [] for name in ("documents", "sections", "blocks", "source_units", "chunks", "figures", "furniture")}
    parse_documents = []
    with tempfile.TemporaryDirectory(prefix="qursor-corpus-", dir=output_root) as temporary:
        work = Path(temporary)
        for index, document in enumerate(registry, 1):
            path = source_dir / document["source_filename"]
            print(f"[{index}/{len(registry)}] parsing {path.name}", flush=True)
            fragment = parse_pdf(path, document, config, work / document["doc_id"].replace(":", "-"), encoder.tokenizer)
            for name in corpus:
                corpus[name].extend(fragment.get(name, []))
            audit = dict(fragment["audit"])
            # Docling writes absolute paths into its internal audit. Those
            # paths point at a temporary build directory and must not leak into
            # a portable serving report.
            for transient in ("raw_export", "markdown_export"):
                audit.pop(transient, None)
            parse_documents.append({"doc_id": document["doc_id"], "filename": path.name,
                                    "page_count": document["page_count"], "low_text_pages": document["low_text_pages"],
                                    "audit": audit})
        report = coverage(corpus)
        if not report["valid"]:
            raise ValueError(f"source coverage validation failed: {len(report['failures'])} gaps")
        vectors = encoder.encode_documents([row["embedding_text"] for row in corpus["chunks"]])
        staging = work / "bundle"
        staging.mkdir()
        with Store.create(staging / "corpus.sqlite", {**corpus, "metadata": {
            "corpus_version": corpus_version,
            "embedding": {"model": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                          "model_tree_sha256": embedding_hash},
        }}, vectors):
            pass
        registry_public = [{key: value for key, value in row.items() if key != "page_text_chars"} for row in registry]
        parse_report = {"schema_version": "enterprise-parse-report-v1", "documents": parse_documents,
                        "document_count": len(registry), "page_count": sum(row["page_count"] for row in registry),
                        "low_text_pages": [{"filename": row["source_filename"], "pages": row["low_text_pages"]}
                                           for row in registry if row["low_text_pages"]]}
        write_json(staging / "source_registry.json", registry_public)
        write_json(staging / "parse_report.json", parse_report)
        write_json(staging / "coverage_report.json", report)
        artifacts = {name: sha256(staging / name) for name in (
            "corpus.sqlite", "corpus.vectors.npy", "source_registry.json", "parse_report.json", "coverage_report.json")}
        manifest = {
            "schema_version": "enterprise-corpus-bundle-v1",
            "corpus_version": corpus_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_fingerprint": fingerprint,
            "document_count": len(corpus["documents"]),
            "section_count": len(corpus["sections"]),
            "chunk_count": len(corpus["chunks"]),
            "embedding": {"model": "intfloat/multilingual-e5-small",
                          "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                          "model_tree_sha256": embedding_hash,
                          "local_path": "../../../../models/e5/multilingual-e5-small"},
            "retrieval": {"bm25_candidates": 20, "e5_candidates": 20, "rrf_k": 60,
                          "bm25_weight": 0.5, "e5_weight": 0.5, "final_k": 5,
                          "bm25_field": "retrieval_text"},
            "artifacts": artifacts,
        }
        write_json(staging / "bundle_manifest.json", manifest)
        os.rename(staging, target)
    write_json(source_dir.parent / "document_manifest.json", {
        "schema_version": "enterprise-source-manifest-v1", "documents": registry})
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/raw/documents")
    parser.add_argument("--output-root", default="data/corpus/versions")
    parser.add_argument("--model-dir", default="models/e5/multilingual-e5-small")
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    result = build(Path(args.source_dir), Path(args.output_root), Path(args.model_dir), args.version)
    print(json.dumps({"status": "complete", "bundle": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
