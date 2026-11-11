"""Read-only immutable corpus bundle and local E5 dense retrieval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from qursor_enterprise_qa.rrf_retrieval import (
    BM25Index,
    SearchResult,
    weighted_rrf,
    weighted_rrf_many,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalE5Encoder:
    def __init__(self, model_dir: Path, *, threads: int = 4):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False).to("cpu").eval()
        import torch
        torch.set_num_threads(threads)

    def encode_query(self, text: str) -> np.ndarray:
        import torch
        prepared = "query: " + text if not text.startswith("query: ") else text
        if len(self.tokenizer.encode(prepared, add_special_tokens=True)) > 512:
            raise ValueError("retrieval query exceeds the E5 512-token budget")
        batch = self.tokenizer([prepared], padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0].astype(np.float32)


class CorpusBundle:
    REQUIRED = ("corpus.sqlite", "corpus.vectors.npy", "bundle_manifest.json")

    def __init__(self, root: str | Path, *, encoder: Any | None = None,
                 model_dir: str | Path | None = None, load_encoder: bool = True):
        self.root = Path(root).resolve()
        manifest_path = self.root / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing corpus manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "enterprise-corpus-bundle-v1":
            raise ValueError("unsupported corpus bundle schema")
        for name in self.REQUIRED:
            if not (self.root / name).is_file():
                raise FileNotFoundError(f"missing corpus asset: {name}")
        for relative, digest in self.manifest.get("artifacts", {}).items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"corpus asset checksum mismatch: {relative}")
        self.connection = sqlite3.connect((self.root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
        self.chunks = self._records("chunks")
        self.documents = {row["doc_id"]: row for row in self._records("documents")}
        self.sections = {row["section_id"]: row for row in self._records("sections")}
        self.vectors = np.load(self.root / "corpus.vectors.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError("corpus vector count does not match chunks")
        self._bm25 = BM25Index(self.chunks)
        self.encoder = encoder
        if load_encoder and encoder is None:
            resolved_model = Path(model_dir) if model_dir is not None else Path(
                self.manifest.get("embedding", {}).get("local_path", ""))
            if not resolved_model.is_absolute():
                resolved_model = (self.root / resolved_model).resolve()
            if not resolved_model.is_dir():
                raise FileNotFoundError(f"local E5 model directory not found: {resolved_model}")
            self.encoder = LocalE5Encoder(resolved_model)

    def _records(self, kind: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(f"SELECT payload FROM {kind} ORDER BY ordinal").fetchall()
        return [json.loads(row[0]) for row in rows]

    @property
    def version(self) -> str:
        return str(self.manifest["corpus_version"])

    def dense_search(self, query: str, k: int = 20) -> list[SearchResult]:
        if self.encoder is None:
            raise RuntimeError("E5 encoder is not loaded")
        vector = np.asarray(self.encoder.encode_query(query), dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        scores = self.vectors @ vector
        ordered = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))[:k]
        return [
            SearchResult(
                chunk_id=self.chunks[index]["chunk_id"], doc_id=self.chunks[index]["doc_id"],
                text=self.chunks[index]["text"], score=float(score), rank=rank,
                metadata=dict(self.chunks[index]), component_scores={"e5": float(score)},
            )
            for rank, (index, score) in enumerate(ordered, 1)
        ]

    def search(self, query: str, *, candidate_k: int = 20, final_k: int = 5) -> list[SearchResult]:
        lexical = self._bm25.search(query, candidate_k)
        dense = self.dense_search(query, candidate_k)
        return weighted_rrf(lexical, dense, k=final_k, lexical_weight=0.5, dense_weight=0.5, rrf_k=60)

    def search_with_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
        *,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> list[SearchResult]:
        """Favor the standalone rewrite (80%) over the original query (20%)."""
        original_bm25 = self._bm25.search(original_query, candidate_k)
        original_e5 = self.dense_search(original_query, candidate_k)
        rewritten_bm25 = self._bm25.search(rewritten_query, candidate_k)
        rewritten_e5 = self.dense_search(rewritten_query, candidate_k)
        return weighted_rrf_many(
            [
                ("original_bm25", original_bm25, 0.10),
                ("original_e5", original_e5, 0.10),
                ("rewrite_bm25", rewritten_bm25, 0.40),
                ("rewrite_e5", rewritten_e5, 0.40),
            ],
            k=final_k,
            rrf_k=60,
        )

    def source(self, result: SearchResult, citation_id: str) -> dict[str, Any]:
        chunk = result.metadata
        doc = self.documents.get(result.doc_id, {})
        section = self.sections.get(chunk.get("section_id"), {})
        spans = chunk.get("source_spans", [])
        pages = [int(span["page_index"]) + 1 for span in spans if isinstance(span.get("page_index"), int)]
        # Chunks are already bounded by the E5 384-token budget. Return the
        # complete original chunk so the cited excerpt cannot omit a claim
        # that appeared after an arbitrary API-side character cutoff.
        excerpt = result.text
        return {
            "citation_id": citation_id,
            "document_id": result.doc_id,
            "document_title": doc.get("title", doc.get("source_filename", result.doc_id)),
            "filename": doc.get("source_filename"),
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "section_path": section.get("path", [section.get("title")]) if section else [],
            "chunk_id": result.chunk_id,
            "excerpt": excerpt,
            "source_spans": [
                {"block_id": span.get("block_id"), "page_index": span.get("page_index"),
                 "start_char": span.get("source_start_char"), "end_char": span.get("source_end_char")}
                for span in spans
            ],
            "rrf_score": result.score,
        }

    def close(self) -> None:
        self.connection.close()
