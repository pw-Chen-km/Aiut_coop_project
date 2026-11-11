from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qursor_enterprise_qa.http_api.app import create_app
from qursor_enterprise_qa.rrf_retrieval import SearchResult
from qursor_enterprise_qa.settings import Settings


class FakeBundle:
    version = "test-corpus"
    manifest = {"embedding": {"model": "fake-e5"}}

    def __init__(self):
        self.queries: list[tuple[str, int, int]] = []
        self.rewrite_queries: list[tuple[str, str, int, int]] = []

    def search(self, query, *, candidate_k, final_k):
        self.queries.append((query, candidate_k, final_k))
        return [SearchResult("c1", "d1", "restart the server", 1.0, 1, {})]

    def search_with_rewrite(self, original_query, rewritten_query, *, candidate_k, final_k):
        self.rewrite_queries.append((original_query, rewritten_query, candidate_k, final_k))
        return [SearchResult("c1", "d1", "restart the server", 1.0, 1, {})]

    def source(self, _result, citation_id):
        return {
            "citation_id": citation_id,
            "document_id": "d1",
            "filename": "manual.pdf",
            "page_start": 2,
            "page_end": 2,
            "excerpt": "restart the server",
        }


class FakeGenerator:
    def __init__(self, response_type="answer", answer=None):
        self.calls: list[tuple[str, list[dict[str, str]], list[SearchResult]]] = []
        self.ping_count = 0
        self.response_type = response_type
        self.answer = answer or (
            "Which Qursor service do you want to restart?"
            if response_type == "clarify"
            else "Restart it."
        )

    def ping(self):
        self.ping_count += 1

    def generate(self, question, history, results):
        self.calls.append((question, history, results))
        return {
            "response_type": self.response_type,
            "answerable": self.response_type == "answer",
            "answer": self.answer,
            "citation_ids": ["C001"] if self.response_type == "answer" else [],
            "missing_information": [],
        }


class FakeRewriter:
    def __init__(self, result="standalone query"):
        self.result = result
        self.calls = []
        self.call_count = 0

    def rewrite(self, question, history):
        self.call_count += 1
        self.calls.append((question, history))
        return self.result


def make_client(*, settings=None, rewriter=None, generator=None):
    root = Path("/tmp/qursor-test")
    settings = settings or Settings(root, root, "http://qwen/v1", "qwen")
    bundle = FakeBundle()
    generator = generator or FakeGenerator()
    app = create_app(settings=settings, bundle=bundle, generator=generator, rewriter=rewriter)
    return TestClient(app), bundle, generator


def test_empty_context_returns_answer_sources_and_same_conversation_id():
    client, bundle, generator = make_client()
    with client:
        response = client.post(
            "/v1/qa",
            json={"conversation_id": 101, "context": [], "question": "How?"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"conversation_id", "response_type", "answer", "sources"}
    assert payload == {
        "conversation_id": 101,
        "response_type": "answer",
        "answer": "Restart it.",
        "sources": [
            {
                "document": "manual.pdf",
                "pages": [2],
                "excerpt": "restart the server",
            }
        ],
    }
    assert bundle.queries == [("How?", 20, 5)]
    assert generator.calls[0][0] == "How?"
    assert generator.calls[0][1] == []


@pytest.mark.parametrize("endpoint", ["/v1/qa", "/internal/ablation/qa?final_k=5"])
@pytest.mark.parametrize("scores", [(1.0, 0.9, 0.8, 0.7, 0.6), (1.0, 0.9, 0.8, 0.8, 0.8)])
def test_sources_sort_only_cited_passages_by_relevance(endpoint, scores, monkeypatch):
    client, bundle, generator = make_client()
    results = [
        SearchResult(f"c{i}", f"d{i}", f"evidence {i}", score, i, {})
        for i, score in enumerate(scores, 1)
    ]

    def search(query, *, candidate_k, final_k):
        return results[:final_k]

    def source(result, citation_id):
        return {
            "citation_id": citation_id,
            "document_title": f"manual-{result.doc_id}",
            "page_start": result.rank,
            "page_end": result.rank,
            "excerpt": result.text,
        }

    def generate(question, history, results):
        generated = FakeGenerator.generate(generator, question, history, results)
        generated["citation_ids"] = ["C005", "C003", "C004"]
        return generated

    monkeypatch.setattr(bundle, "search", search)
    monkeypatch.setattr(bundle, "source", source)
    monkeypatch.setattr(generator, "generate", generate)
    with client:
        response = client.post(
            endpoint,
            json={"conversation_id": 116, "context": [], "question": "How?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 116,
        "response_type": "answer",
        "answer": "Restart it.",
        "sources": [
            {"document": f"manual-d{i}", "pages": [i], "excerpt": f"evidence {i}"}
            for i in (3, 4, 5)
        ],
    }


def test_internal_ablation_endpoint_uses_requested_final_k():
    client, bundle, generator = make_client()
    with client:
        response = client.post(
            "/internal/ablation/qa?final_k=10",
            json={"conversation_id": 114, "context": [], "question": "How?"},
        )

    assert response.status_code == 200
    assert set(response.json()) == {"conversation_id", "response_type", "answer", "sources"}
    assert bundle.queries == [("How?", 20, 10)]
    assert generator.calls[0][2]


@pytest.mark.parametrize("final_k", [0, 21, "invalid"])
def test_internal_ablation_endpoint_rejects_invalid_final_k(final_k):
    client, _, _ = make_client()
    with client:
        response = client.post(
            "/internal/ablation/qa",
            params={"final_k": final_k},
            json={"conversation_id": 115, "context": [], "question": "How?"},
        )

    assert response.status_code == 422


def test_abstain_uses_fixed_public_message_and_empty_sources():
    client, _, _ = make_client(generator=FakeGenerator(response_type="abstain"))
    with client:
        response = client.post(
            "/v1/qa",
            json={"conversation_id": 106, "context": [], "question": "What is today's dinner?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 106,
        "response_type": "abstain",
        "answer": "I can't answer this",
        "sources": [],
    }


def test_clarify_returns_concrete_follow_up_and_empty_sources():
    client, bundle, generator = make_client()
    with client:
        response = client.post(
            "/v1/qa",
            json={"conversation_id": 108, "context": [], "question": "How do I restart it?"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 108,
        "response_type": "clarify",
        "answer": "Which Qursor system, document, or procedure are you referring to?",
        "sources": [],
    }
    assert bundle.queries == []
    assert generator.calls == []


@pytest.mark.parametrize("fallback", ["I can't answer this", "  I CAN'T ANSWER THIS  "])
def test_abstain_history_is_excluded_from_rewrite_and_answer(fallback):
    rewriter = FakeRewriter("How do I restart the Qursor server?")
    client, _, generator = make_client(rewriter=rewriter)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 109,
                "context": [
                    {"Q": "What is today's dinner?", "A": fallback},
                    {"Q": "What is Qursor?", "A": "An enterprise system."},
                    {"Q": "How do I restart it?", "A": "Which Qursor service?"},
                    {"Q": "What is the weather today?", "A": fallback},
                ],
                "question": "How do I restart that server?",
            },
        )
    assert response.status_code == 200
    assert rewriter.calls == [(
        "How do I restart that server?",
        "user: What is Qursor?\nassistant: An enterprise system.\n"
        "user: How do I restart it?\nassistant: Which Qursor service?",
    )]
    assert generator.calls[0][1] == [
        {"role": "user", "content": "What is Qursor?"},
        {"role": "assistant", "content": "An enterprise system."},
        {"role": "user", "content": "How do I restart it?"},
        {"role": "assistant", "content": "Which Qursor service?"},
    ]


def test_only_abstain_history_returns_deterministic_clarify():
    rewriter = FakeRewriter()
    client, bundle, generator = make_client(rewriter=rewriter)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 110,
                "context": [{"Q": "Dinner?", "A": "I can't answer this"}],
                "question": "How do I restart it?",
            },
        )
    assert response.status_code == 200
    assert rewriter.call_count == 0
    assert response.json()["response_type"] == "clarify"
    assert bundle.queries == []
    assert generator.calls == []


def test_abstain_history_is_also_filtered_when_rewrite_does_not_trigger():
    rewriter = FakeRewriter()
    client, _, generator = make_client(rewriter=rewriter)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 111,
                "context": [
                    {"Q": "What is Kanban?", "A": "It contains steps."},
                    {"Q": "Dinner?", "A": "I can't answer this"},
                ],
                "question": "How do I install SSL?",
            },
        )
    assert response.status_code == 200
    assert rewriter.call_count == 0
    assert generator.calls[0][1] == [
        {"role": "user", "content": "What is Kanban?"},
        {"role": "assistant", "content": "It contains steps."},
    ]


def test_disabled_rewrite_does_not_include_abstain_in_retrieval_query():
    root = Path("/tmp/qursor-test")
    settings = Settings(
        root, root, "http://qwen/v1", "qwen",
        query_rewrite_mode="disabled",
    )
    client, bundle, generator = make_client(settings=settings)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 112,
                "context": [{"Q": "Dinner?", "A": "I can't answer this"}],
                "question": "How do I install SSL?",
            },
        )
    assert response.status_code == 200
    assert bundle.queries == [("CURRENT QUESTION\nHow do I install SSL?", 20, 5)]
    assert generator.calls[0][1] == []


def test_clarify_can_be_sent_back_in_context_for_next_answer():
    generator = FakeGenerator(response_type="clarify")
    client, _, _ = make_client(generator=generator)
    with client:
        first = client.post(
            "/v1/qa",
            json={"conversation_id": 113, "context": [], "question": "How do I restart it?"},
        )
        generator.response_type = "answer"
        generator.answer = "Restart the Qursor server."
        second = client.post(
            "/v1/qa",
            json={
                "conversation_id": 113,
                "context": [{"Q": "How do I restart it?", "A": first.json()["answer"]}],
                "question": "The Qursor server.",
            },
        )
    assert first.json()["response_type"] == "clarify"
    assert second.json()["response_type"] == "answer"
    assert generator.calls[0][0] == "The Qursor server."
    assert generator.calls[0][1] == [
        {"role": "user", "content": "How do I restart it?"},
        {"role": "assistant", "content": "Which Qursor system, document, or procedure are you referring to?"},
    ]


def test_openapi_documents_all_three_public_response_types():
    client, _, _ = make_client()
    schema = client.get("/openapi.json").json()
    response = schema["components"]["schemas"]["QAResponse"]
    assert set(response["required"]) == {"conversation_id", "response_type", "answer", "sources"}
    assert response["properties"]["response_type"]["enum"] == ["answer", "clarify", "abstain"]


def test_context_is_converted_to_ordered_user_assistant_history():
    client, bundle, generator = make_client()
    body = {
        "conversation_id": 102,
        "context": [
            {"Q": "What is Kanban?", "A": "It contains steps."},
            {"Q": "Who triggers it?", "A": "An operator or external system."},
        ],
        "question": "How do I verify completion?",
    }
    with client:
        response = client.post("/v1/qa", json=body)
    assert response.status_code == 200
    assert generator.calls[0][0] == body["question"]
    assert generator.calls[0][1] == [
        {"role": "user", "content": "What is Kanban?"},
        {"role": "assistant", "content": "It contains steps."},
        {"role": "user", "content": "Who triggers it?"},
        {"role": "assistant", "content": "An operator or external system."},
    ]
    assert bundle.queries == [("How do I verify completion?", 20, 5)]


def test_history_dependent_question_uses_rewriter_and_four_way_search():
    rewriter = FakeRewriter("after installing SSL, what should be verified?")
    client, bundle, generator = make_client(rewriter=rewriter)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 103,
                "context": [{"Q": "How do I install SSL?", "A": "Copy the certificate files."}],
                "question": "What should I verify after that?",
            },
        )
    assert response.status_code == 200
    assert bundle.queries == []
    assert bundle.rewrite_queries == [
        ("What should I verify after that?", "after installing SSL, what should be verified?", 20, 5)
    ]
    assert rewriter.call_count == 1
    assert "How do I install SSL?" in rewriter.calls[0][1]
    assert generator.calls[0][0] == "What should I verify after that?"


def test_failed_rewrite_returns_deterministic_clarify_without_retrieval():
    rewriter = FakeRewriter(None)
    client, bundle, _ = make_client(rewriter=rewriter)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 104,
                "context": [{"Q": "Earlier", "A": "Earlier answer"}],
                "question": "What should I verify after that?",
            },
        )
    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": 104,
        "response_type": "clarify",
        "answer": "Which Qursor system, document, or procedure are you referring to?",
        "sources": [],
    }
    assert bundle.queries == []
    assert bundle.rewrite_queries == []
    assert rewriter.call_count == 1


def test_which_one_without_history_returns_deterministic_clarify():
    client, bundle, generator = make_client()
    with client:
        response = client.post(
            "/v1/qa",
            json={"conversation_id": 114, "context": [], "question": "Which one?"},
        )
    assert response.status_code == 200
    assert response.json()["response_type"] == "clarify"
    assert bundle.queries == []
    assert generator.calls == []


def test_disabled_mode_preserves_legacy_history_query():
    settings = Settings(
        Path("/tmp/qursor-test"),
        Path("/tmp/qursor-test"),
        "http://qwen/v1",
        "qwen",
        query_rewrite_mode="disabled",
    )
    client, bundle, _ = make_client(settings=settings)
    with client:
        response = client.post(
            "/v1/qa",
            json={
                "conversation_id": 105,
                "context": [{"Q": "Previous question", "A": "Previous answer"}],
                "question": "Current question",
            },
        )
    assert response.status_code == 200
    assert bundle.queries[0][0].startswith("CURRENT QUESTION\nCurrent question")
    assert "Previous question" in bundle.queries[0][0]


def test_repeated_conversation_id_does_not_create_server_side_state():
    client, _, generator = make_client()
    with client:
        first = client.post(
            "/v1/qa",
            json={
                "conversation_id": 777,
                "context": [{"Q": "First Q", "A": "First A"}],
                "question": "First current question",
            },
        )
        second = client.post(
            "/v1/qa",
            json={
                "conversation_id": 777,
                "context": [{"Q": "Second Q", "A": "Second A"}],
                "question": "Second current question",
            },
        )
    assert first.status_code == second.status_code == 200
    assert generator.calls[0][1][0]["content"] == "First Q"
    assert generator.calls[1][1][0]["content"] == "Second Q"
    assert all("First Q" not in message["content"] for message in generator.calls[1][1])


@pytest.mark.parametrize(
    "body",
    [
        {"context": [], "question": "Question"},
        {"conversation_id": "123", "context": [], "question": "Question"},
        {"conversation_id": 1.5, "context": [], "question": "Question"},
        {"conversation_id": True, "context": [], "question": "Question"},
        {"conversation_id": 123, "question": "Question"},
        {"conversation_id": 123, "context": [{}], "question": "Question"},
        {"conversation_id": 123, "context": [{"Q": "Q"}], "question": "Question"},
        {"conversation_id": 123, "context": [{"Q": " ", "A": "A"}], "question": "Question"},
        {"conversation_id": 123, "context": [{"Q": "Q", "A": " "}], "question": "Question"},
        {"conversation_id": 123, "context": [], "question": " "},
        {"conversation_id": 123, "context": [], "question": "Question", "message": "old"},
    ],
)
def test_invalid_request_returns_422(body):
    client, _, _ = make_client()
    with client:
        response = client.post("/v1/qa", json=body)
    assert response.status_code == 422


def test_readyz_checks_qwen_without_database():
    client, _, generator = make_client()
    with client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "corpus_version": "test-corpus"}
    assert generator.ping_count == 1


def test_old_conversation_endpoints_are_removed():
    client, _, _ = make_client()
    with client:
        post_response = client.post(
            "/v1/conversations/chat-1/messages", json={"message": "old contract"}
        )
        get_response = client.get("/v1/conversations/chat-1/messages")
        delete_response = client.delete("/v1/conversations/chat-1")
    assert post_response.status_code == 404
    assert get_response.status_code == 404
    assert delete_response.status_code == 404
