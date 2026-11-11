import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from qursor_enterprise_qa.corpus_runtime import CorpusBundle
from qursor_enterprise_qa.rrf_retrieval import SearchResult


class FakeEncoder:
    def encode_query(self, _text):
        return np.array([1.0, 0.0], dtype=np.float32)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bundle(root):
    documents = [{"doc_id": "d1", "title": "Restart", "source_filename": "restart.pdf"}]
    sections = [{"section_id": "s1", "doc_id": "d1", "title": "Procedure", "path": ["Restart", "Procedure"]}]
    chunks = []
    for i in range(6):
        chunks.append({"chunk_id": f"c{i}", "doc_id": "d1", "section_id": "s1",
                       "text": f"restart server procedure step {i}",
                       "source_spans": [{"block_id": f"b{i}", "page_index": i,
                                         "source_start_char": 0, "source_end_char": 7,
                                         "chunk_start_char": 0, "chunk_end_char": 7}]})
    connection = sqlite3.connect(root / "corpus.sqlite")
    for kind, rows in (("documents", documents), ("sections", sections), ("chunks", chunks)):
        connection.execute(f"CREATE TABLE {kind} (ordinal INTEGER, payload TEXT)")
        connection.executemany(f"INSERT INTO {kind} VALUES (?, ?)",
                               [(index, json.dumps(row)) for index, row in enumerate(rows)])
    connection.commit()
    connection.close()
    vectors = np.array([[1.0 - i * .01, i * .01] for i in range(6)], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    np.save(root / "corpus.vectors.npy", vectors)
    manifest = {"schema_version": "enterprise-corpus-bundle-v1", "corpus_version": "test-v1",
                "embedding": {"model": "fake", "local_path": "missing"},
                "artifacts": {"corpus.sqlite": _digest(root / "corpus.sqlite"),
                              "corpus.vectors.npy": _digest(root / "corpus.vectors.npy")}}
    (root / "bundle_manifest.json").write_text(json.dumps(manifest))


def test_bundle_search_and_source(tmp_path):
    make_bundle(tmp_path)
    bundle = CorpusBundle(tmp_path, encoder=FakeEncoder())
    try:
        results = bundle.search("restart server", candidate_k=6, final_k=5)
        assert len(results) == 5
        source = bundle.source(results[0], "C001")
        assert source["filename"] == "restart.pdf"
        assert source["page_start"] == 1
        assert source["section_path"] == ["Restart", "Procedure"]
    finally:
        bundle.close()


def test_bundle_rewrite_is_primary_retrieval_signal(tmp_path, monkeypatch):
    make_bundle(tmp_path)
    bundle = CorpusBundle(tmp_path, encoder=FakeEncoder())
    calls = []

    def results(query, k):
        calls.append((query, k))
        # Both queries rank their own distinct passage first. The rewrite
        # should win because of the runtime weights, not different ranks.
        chunk = bundle.chunks[5 if query == "restart Qursor server" else 0]
        return [SearchResult(chunk["chunk_id"], chunk["doc_id"], chunk["text"],
                             1.0, 1, chunk)]

    monkeypatch.setattr(bundle._bm25, "search", results)
    monkeypatch.setattr(bundle, "dense_search", results)
    try:
        ranked = bundle.search_with_rewrite(
            "restart it", "restart Qursor server", candidate_k=20, final_k=2
        )
        assert [item.chunk_id for item in ranked] == ["c5", "c0"]
        assert np.isclose(ranked[0].score, 0.80 / 61)
        assert np.isclose(ranked[1].score, 0.20 / 61)
        assert set(ranked[0].component_scores) == {"rewrite_bm25", "rewrite_e5"}
        assert calls == [("restart it", 20)] * 2 + [("restart Qursor server", 20)] * 2
    finally:
        bundle.close()
