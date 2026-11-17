import json

import pytest

from topic_rag.evaluation import evidence_metrics, summarize
from topic_rag.generation import AnswerGenerator


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def render(self, messages):
        return "prompt"

    def tokenize(self, text):
        return [1, 2]

    def post(self, path, body):
        self.calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.response)}}]}


def test_only_actually_cited_sources_in_retrieval_order():
    model = FakeModel({"response_type": "answer", "answer": "Use settings.", "citation_ids": ["C002"]})
    chunks = [{"text": "first"}, {"text": "second"}]
    result = AnswerGenerator(model, "qwen4b", "sys").generate("How?", chunks)
    assert result["sources"] == [{"citation_id": "C002", "text": "second"}]
    assert model.calls[0]["temperature"] == 0


@pytest.mark.parametrize("kind,citations", [("answer", []), ("answer", ["C999"]), ("abstain", ["C001"])])
def test_invalid_citations_fail(kind, citations):
    model = FakeModel({"response_type": kind, "answer": "text", "citation_ids": citations})
    with pytest.raises(ValueError):
        AnswerGenerator(model, "4b", "s").generate("q", [{"text": "t"}])


def test_no_truncation_and_empty_hard_pool():
    model = FakeModel({})
    generator = AnswerGenerator(model, "4b", "sys", context_tokens=10, max_tokens=9)
    with pytest.raises(ValueError, match="context exceeds"):
        generator.generate("q", [{"text": "t"}])
    assert generator.generate("q", [])["response_type"] == "abstain"
    assert model.calls == []


def test_evidence_recall_requires_span_not_merely_correct_source():
    evidence = [{"evidence_id": "e", "source_id": "s", "start": 10, "end": 20}]
    chunks = [{"source_id": "s", "source_start": 0, "source_end": 15},
              {"source_id": "s", "source_start": 15, "source_end": 25}]
    metrics = evidence_metrics(chunks, evidence + evidence, cutoffs=(1, 2))
    assert metrics["1"]["source_recall"] == 1
    assert metrics["1"]["evidence_recall"] == 0
    assert metrics["2"]["evidence_recall"] == 1
    assert metrics["2"]["eligible"] == 1


def test_answer_failure_does_not_erase_retrieval_recall():
    metrics = evidence_metrics([{"source_id": "s", "source_start": 0, "source_end": 10}],
                               [{"evidence_id": "e", "source_id": "s", "start": 1, "end": 2}])
    rows = [{"variant": "soft_topic", "error": "generation failed", "evidence": metrics},
            {"variant": "soft_topic", "error": "scoring failed"}]
    result = summarize(rows)[0]
    assert result["error_count"] == 2
    assert result["retrieval_success_count"] == 1
    assert result["evidence_recall_macro_at_5"] == 1
    assert result["recall_macro_errors_as_zero_at_5"] == .5
    assert result["answer_accuracy"] is None
