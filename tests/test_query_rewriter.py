from qursor_enterprise_qa.answer_generation import QueryRewriter


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeClient:
    response_content = '{"query":"standalone SSL query"}'

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, *_args, **_kwargs):
        return FakeResponse(self.response_content)


def test_query_rewriter_returns_standalone_query(monkeypatch):
    monkeypatch.setattr("qursor_enterprise_qa.answer_generation.qwen.httpx.Client", FakeClient)
    rewriter = QueryRewriter("http://qwen/v1", "qwen")

    assert rewriter.rewrite("What should I check after that?", "user: Install SSL") == (
        "standalone SSL query"
    )
    assert rewriter.call_count == 1


def test_query_rewriter_falls_back_without_retry_on_invalid_json(monkeypatch):
    FakeClient.response_content = "not json"
    monkeypatch.setattr("qursor_enterprise_qa.answer_generation.qwen.httpx.Client", FakeClient)
    rewriter = QueryRewriter("http://qwen/v1", "qwen")

    assert rewriter.rewrite("What should I check after that?", "user: Install SSL") is None
    assert rewriter.call_count == 1
    FakeClient.response_content = '{"query":"standalone SSL query"}'
