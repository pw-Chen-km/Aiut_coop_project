"""Versioned, read-only SQLite corpus registry and exact scoped cosine search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence


KINDS = ("documents", "sections", "blocks", "source_units", "chunks", "figures")
ID_FIELDS = {
    "documents": ("doc_id",), "sections": ("section_id",),
    "blocks": ("block_id",), "chunks": ("chunk_id",),
    "source_units": ("source_unit_id", "unit_id", "source_atom_id"),
    "figures": ("figure_id",),
}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Doc2Skill vector storage requires numpy") from exc
    return np


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Store:
    """A corpus snapshot; opening a store never modifies its database or vectors."""

    @classmethod
    def create(cls, path: str | Path, corpus: Mapping[str, Any], vectors: Any) -> "Store":
        np = _numpy()
        path = Path(path).resolve()
        vector_path = path.with_suffix(".vectors.npy")
        if path == vector_path or path.exists() or vector_path.exists():
            raise FileExistsError("Refusing to overwrite a corpus database or its vectors")
        rows = {kind: list(corpus.get(kind, ())) for kind in KINDS}
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(rows["chunks"]) or array.shape[1] < 1:
            raise ValueError("vectors must have shape (number of chunks, positive dimensions)")
        if not np.isfinite(array).all():
            raise ValueError("vectors contain non-finite values")
        lengths = np.linalg.norm(array, axis=1)
        if np.any(lengths == 0):
            raise ValueError("zero document vectors cannot be cosine-normalized")
        array = array / lengths[:, None]
        path.parent.mkdir(parents=True, exist_ok=True)
        linked_vector = False
        with tempfile.TemporaryDirectory(prefix="doc2skill-store-", dir=path.parent) as temporary:
            temp_db = Path(temporary) / "corpus.sqlite"
            temp_vectors = temp_db.with_suffix(".vectors.npy")
            np.save(temp_vectors, array, allow_pickle=False)
            connection = sqlite3.connect(temp_db)
            try:
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                metadata = dict(corpus.get("metadata") or {})
                metadata.update({"store_schema_version": "doc2skill-store-v1",
                                 "vector_sha256": _digest(temp_vectors),
                                 "vector_dimensions": int(array.shape[1])})
                connection.executemany("INSERT INTO metadata VALUES (?, ?)",
                                       [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()])
                for kind in KINDS:
                    connection.execute(f"CREATE TABLE {kind} (ordinal INTEGER PRIMARY KEY, record_id TEXT UNIQUE NOT NULL, doc_id TEXT, section_id TEXT, payload TEXT NOT NULL)")
                    for ordinal, record in enumerate(rows[kind]):
                        if not isinstance(record, Mapping):
                            raise ValueError(f"{kind} records must be objects")
                        identifier = next((record.get(key) for key in ID_FIELDS[kind] if record.get(key)), None)
                        if identifier is None:
                            if kind in {"documents", "sections", "blocks", "chunks"}:
                                raise ValueError(f"{kind} record lacks its stable ID")
                            identifier = f"{kind}:{ordinal}"
                        connection.execute(f"INSERT INTO {kind} VALUES (?, ?, ?, ?, ?)",
                                           (ordinal, str(identifier), record.get("doc_id"), record.get("section_id"),
                                            json.dumps(dict(record), ensure_ascii=False)))
                    connection.execute(f"CREATE INDEX {kind}_scope ON {kind} (doc_id, section_id)")
                connection.commit()
            finally:
                connection.close()
            with cls(temp_db):
                pass
            try:
                # Hard links publish without overwriting even if a competing build appears.
                os.link(temp_vectors, vector_path)
                linked_vector = True
                os.link(temp_db, path)
            except Exception:
                if linked_vector:
                    vector_path.unlink()
                raise
        return cls(path)

    def __init__(self, path: str | Path):
        np = _numpy()
        self.path = Path(path).resolve()
        self.vector_path = self.path.with_suffix(".vectors.npy")
        self.vectors = None
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            self.metadata = {key: json.loads(value) for key, value in self.connection.execute("SELECT key, payload FROM metadata")}
            if self.metadata.get("store_schema_version") != "doc2skill-store-v1":
                raise ValueError("unsupported store schema")
            if _digest(self.vector_path) != self.metadata.get("vector_sha256"):
                raise ValueError("vector checksum does not match the corpus snapshot")
            self.vectors = np.load(self.vector_path, allow_pickle=False, mmap_mode="r")
            self.chunks = self.records("chunks")
            self.documents = {item["doc_id"]: item for item in self.records("documents")}
            self.sections = {item["section_id"]: item for item in self.records("sections")}
            if self.vectors.ndim != 2 or self.vectors.shape != (len(self.chunks), self.metadata["vector_dimensions"]):
                raise ValueError("vector index shape disagrees with the corpus registry")
            self._validate_tree()
        except Exception:
            self.close()
            raise

    def _validate_tree(self) -> None:
        for identifier, section in self.sections.items():
            if section.get("doc_id") not in self.documents:
                raise ValueError(f"section {identifier!r} has unknown document")
            visited = {identifier}
            parent = section.get("parent_id")
            while parent:
                if parent in visited or parent not in self.sections:
                    raise ValueError("section hierarchy has a cycle or unknown parent")
                visited.add(parent)
                if self.sections[parent].get("doc_id") != section.get("doc_id"):
                    raise ValueError("section parent crosses document boundaries")
                parent = self.sections[parent].get("parent_id")
        for chunk in self.chunks:
            section = self.sections.get(chunk.get("section_id"))
            if section is None or section.get("doc_id") != chunk.get("doc_id"):
                raise ValueError("chunk has an unknown section or mismatched document")

    def records(self, kind: str) -> list[dict[str, Any]]:
        if kind not in KINDS:
            raise ValueError(f"unknown registry kind: {kind!r}")
        return [json.loads(row[0]) for row in self.connection.execute(f"SELECT payload FROM {kind} ORDER BY ordinal")]

    def _scope(self, scope: Mapping[str, Any] | None) -> tuple[set[str] | None, set[str] | None]:
        if scope is None:
            return None, None
        if not isinstance(scope, Mapping) or not scope or set(scope) - {"doc_ids", "section_ids"}:
            raise ValueError("scope requires nonempty doc_ids and/or section_ids")
        selected: dict[str, set[str]] = {}
        for field, registry in (("doc_ids", self.documents), ("section_ids", self.sections)):
            if field not in scope:
                continue
            values = scope[field]
            if not isinstance(values, (list, tuple, set)) or not values or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field} must be a nonempty collection of IDs")
            selected[field] = set(values)
            unknown = selected[field] - registry.keys()
            if unknown:
                raise ValueError(f"unknown {field}: {sorted(unknown)}")
        doc_ids = selected.get("doc_ids")
        section_ids = selected.get("section_ids")
        if section_ids is not None:
            section_ids = set(section_ids)
            changed = True
            while changed:
                additions = {identifier for identifier, section in self.sections.items()
                             if section.get("parent_id") in section_ids} - section_ids
                changed = bool(additions)
                section_ids.update(additions)
            if doc_ids is not None:
                section_ids = {identifier for identifier in section_ids if self.sections[identifier]["doc_id"] in doc_ids}
                if not section_ids:
                    raise ValueError("document and section scopes do not intersect")
        return doc_ids, section_ids

    def validate_scope(self, scope: Mapping[str, Any]) -> None:
        """Public preflight for adapters before performing expensive encoding."""
        self._scope(scope)

    def search(self, query_vector: Sequence[float], scope: Mapping[str, Any] | None = None, k: int = 20) -> list[dict[str, Any]]:
        np = _numpy()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        doc_ids, section_ids = self._scope(scope)
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1] or not np.isfinite(query).all():
            raise ValueError("query vector has invalid dimensions or non-finite values")
        length = np.linalg.norm(query)
        if length == 0:
            raise ValueError("query vector must be nonzero")
        indices = [index for index, chunk in enumerate(self.chunks)
                   if (doc_ids is None or chunk["doc_id"] in doc_ids)
                   and (section_ids is None or chunk["section_id"] in section_ids)]
        if not indices:
            return []
        scores = self.vectors[indices] @ (query / length)
        ordered = sorted(zip(indices, scores), key=lambda item: (-float(item[1]), item[0]))[:k]
        return [dict(self.chunks[index], rank=rank, score=float(score))
                for rank, (index, score) in enumerate(ordered, 1)]

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            # NumPy's mmap outlives the SQLite connection. Release it explicitly:
            # Windows prevents publishing/deleting a snapshot while it is mapped.
            mapping = getattr(self.vectors, "_mmap", None)
            if mapping is not None:
                mapping.close()
            self.vectors = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
