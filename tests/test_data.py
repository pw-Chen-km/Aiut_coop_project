from pathlib import Path
from types import SimpleNamespace

import pytest

from topic_rag.config import load_config
from topic_rag.datasets.emanual import normalize
from topic_rag.indexing import make_chunks, validate_chunks
from topic_rag.io import read_jsonl, write_jsonl
from topic_rag.validation import validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def row(identifier, keys):
    return {"id": identifier, "question": "How?", "documents": ["First. Second."],
            "documents_sentences": [[["0a", "First."], ["0b", "Second."]]],
            "all_relevant_sentence_keys": keys, "response": identifier,
            "adherence_score": identifier != "b"}


def test_explicit_keys_deduplication_and_all_references(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [row("a", ["0a"]), row("b", ["0b"]), row("bad", ["0b."])])
    report = normalize(raw, tmp_path / "normalized")
    assert report["raw_rows"] == 3
    assert report["valid_rows"] == 2
    assert report["unique_questions"] == 1
    assert report["invalid_rows"][0]["unmapped_keys"] == ["0b."]
    questions = read_jsonl(tmp_path / "normalized/questions.jsonl")
    assert len(questions[0]["gold_evidence"]) == 2
    assert [r["response"] for r in questions[0]["reference_responses"]] == ["a", "b"]
    assert questions[0]["reference_responses"][1]["adherence_score"] is False
    assert len(read_jsonl(tmp_path / "normalized/sources.jsonl")) == 1
    first_hashes = {p.name: p.read_bytes() for p in (tmp_path / "normalized").iterdir()}
    normalize(raw, tmp_path / "normalized")
    assert first_hashes == {p.name: p.read_bytes() for p in (tmp_path / "normalized").iterdir()}


def test_no_fuzzy_span_repair(tmp_path):
    bad = row("bad", ["0a"])
    bad["documents_sentences"][0][0][1] = "FIRST."
    write_jsonl(tmp_path / "raw.jsonl", [bad])
    report = normalize(tmp_path / "raw.jsonl", tmp_path / "out")
    assert report["valid_rows"] == 0
    assert "invalid_sentence:0a" in report["invalid_rows"][0]["reasons"]


class CharacterTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(text))


def test_chunking_preserves_spaces_and_overlong_sentences():
    source = {"source_id": "s", "title": "T", "text": "First. " + "x" * 100 + " Last.",
              "sentence_spans": [{"start": 0, "end": 6}, {"start": 7, "end": 107}]}
    chunks = make_chunks([source], CharacterTokenizer(), max_tokens=35)
    validate_chunks([source], chunks)
    assert len(chunks) > 3
    assert "".join(c["text"] for c in chunks) == source["text"]
    assert max(c["embedding_tokens"] for c in chunks) <= 35
    broken = [dict(c) for c in chunks]
    broken[1]["source_start"] += 1
    with pytest.raises(ValueError, match="coverage/alignment"):
        validate_chunks([source], broken)


def test_real_dataset_is_prepared_and_fully_aligned():
    config = load_config(ROOT / "configs/emanual.yaml")
    report = validate_dataset(config)
    assert (report["sources"], report["chunks"], report["questions"], report["labels"]) == (102, 115, 66, 102)
    assert list(report["topic_options"]) == [str(i) for i in range(1, 11)]
    chunks = read_jsonl(Path(config["dataset_dir"]) / "chunks.jsonl")
    assert max(c["embedding_tokens"] for c in chunks) <= 384
    questions = read_jsonl(Path(config["dataset_dir"]) / "questions.jsonl")
    assert sum(len(q["reference_responses"]) for q in questions) == 130
