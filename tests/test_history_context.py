from qursor_enterprise_qa.history_context import (
    HistoryPair,
    build_answer_history,
    build_retrieval_query,
    build_rewrite_history,
    has_query_rewrite_trigger,
    needs_query_rewrite,
)


def test_retrieval_budget_keeps_current_question_and_newest_complete_pair():
    pairs = [
        HistoryPair("old question", "old answer"),
        HistoryPair("new question", "new answer"),
    ]
    current = "current question"
    newest = "user: new question\nassistant: new answer"
    budget = len(f"CURRENT QUESTION\n{current}") + len(newest) + 1
    query = build_retrieval_query(pairs, current, budget)
    assert "current question" in query
    assert "new question" in query
    assert "new answer" in query
    assert "old question" not in query
    assert "old answer" not in query


def test_answer_budget_never_splits_question_answer_pair():
    pairs = [
        HistoryPair("old question", "old answer"),
        HistoryPair("new question", "new answer"),
    ]
    budget = len("new question") + len("new answer")
    history = build_answer_history(pairs, budget)
    assert history == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]


def test_current_question_is_preserved_when_retrieval_budget_is_tiny():
    query = build_retrieval_query([HistoryPair("old", "answer")], "latest", 1)
    assert query == "CURRENT QUESTION\nlatest"


def test_rewrite_trigger_requires_history_and_detects_ellipsis():
    pair = [HistoryPair("How do I install SSL?", "Copy the certificate files.")]
    assert needs_query_rewrite([], "What should I check after that?") is False
    assert needs_query_rewrite(pair, "What should I check after that?") is True
    assert needs_query_rewrite(pair, "How do I configure MQTT?") is False


def test_rewrite_trigger_can_be_detected_before_history_check():
    assert has_query_rewrite_trigger("How do I restart it?") is True
    assert has_query_rewrite_trigger("Which one?") is True
    assert has_query_rewrite_trigger("How do I configure MQTT?") is False


def test_rewrite_history_keeps_complete_recent_pairs_within_budget():
    pairs = [HistoryPair("old", "old answer"), HistoryPair("new", "new answer")]
    assert build_rewrite_history(pairs, len("user: new\nassistant: new answer") + 1) == (
        "user: new\nassistant: new answer"
    )
