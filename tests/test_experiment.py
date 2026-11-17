from pathlib import Path
import json
import shutil

import numpy as np

from topic_rag import experiment
from topic_rag.config import load_config
from topic_rag.io import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_default_plan_never_constructs_llm_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not open a model client")
    monkeypatch.setattr(experiment, "ModelClient", forbidden)
    monkeypatch.setattr(experiment, "E5Encoder", forbidden)
    report = experiment.plan(load_config(ROOT / "configs/emanual.yaml"))
    assert report["requires_explicit_execute"] is True
    assert report["llm_requests_sent"] == 0


def test_runner_uses_shared_decision_top5_answers_and_top20_recall(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/emanual.yaml")
    original = Path(config["dataset_dir"])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("sources.jsonl", "chunks.jsonl", "validation.json"):
        shutil.copyfile(original / name, dataset / name)
    questions = read_jsonl(original / "questions.jsonl")[:2]
    write_jsonl(dataset / "questions.jsonl", questions)
    config.update(dataset_dir=str(dataset), results_dir=str(tmp_path / "results"))
    chunks = read_jsonl(dataset / "chunks.jsonl")
    vectors = np.tile([1.0, 0.0], (len(chunks), 1))
    monkeypatch.setattr(experiment, "load_index", lambda *a: (chunks, vectors, {"test": True}))
    calls = {"scorer": [], "answer_sizes": [], "closed": False}

    class Encoder:
        def __init__(self, *a):
            pass

        def query(self, query):
            return np.array([1.0, 0.0])

    class Client:
        def __init__(self, *a):
            pass

        def identity(self):
            return {"id": "mock-4b"}

        def close(self):
            calls["closed"] = True

    class Scorer:
        def __init__(self, *a):
            pass

        def score(self, query):
            calls["scorer"].append(query)
            return {"probabilities": {str(i): .1 for i in range(1, 11)}, "elapsed_ms": 7}

    class Generator:
        def __init__(self, *a):
            pass

        def generate(self, query, selected):
            calls["answer_sizes"].append(len(selected))
            return {"answer": "mock only", "response_type": "answer", "sources": selected[:1]}

    monkeypatch.setattr(experiment, "E5Encoder", Encoder)
    monkeypatch.setattr(experiment, "ModelClient", Client)
    monkeypatch.setattr(experiment, "JevScorer", Scorer)
    monkeypatch.setattr(experiment, "AnswerGenerator", Generator)
    output = experiment.run(config)
    assert calls["closed"] is True
    assert len(calls["scorer"]) == 2
    assert len(calls["answer_sizes"]) == 6 and max(calls["answer_sizes"]) == 5
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert len(read_jsonl(output / "answer_review.jsonl")) == 6
    baseline = read_jsonl(output / "baseline.jsonl")
    soft = read_jsonl(output / "soft_topic.jsonl")
    hard = read_jsonl(output / "hard_topic.jsonl")
    for b, s, h in zip(baseline, soft, hard):
        assert len(b["retrieved_chunks"]) == 20
        assert len(b["answer_context"]) == 5
        assert set(b["evidence"]) == {"5", "10", "20"}
        assert b["retrieval_trace"]["candidate_ids"] == s["retrieval_trace"]["candidate_ids"]
        assert s["topic_probabilities"] == h["topic_probabilities"]
        assert s["latency_ms"]["topic_scoring"] == 7
        assert b["latency_ms"]["topic_scoring"] == 0
