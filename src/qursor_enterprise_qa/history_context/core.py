"""Build bounded retrieval and answer history from backend-supplied Q/A pairs."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence, TypeVar


@dataclass(frozen=True)
class HistoryPair:
    question: str
    answer: str


T = TypeVar("T")


_REWRITE_TRIGGER_PATTERNS = (
    re.compile(r"\b(?:it|its|this|that|these|those|they|them)\b", re.IGNORECASE),
    re.compile(r"\b(?:the previous|the earlier|the above|mentioned|same|next step)\b", re.IGNORECASE),
    re.compile(r"\b(?:after that|before that|what about|how about)\b", re.IGNORECASE),
    re.compile(r"\bwhich one\b", re.IGNORECASE),
    re.compile(r"(?:它|這個|那個|這些|那些|前面|剛才|之前|上述|同樣|下一步|接著|然後|之後)"),
)


def _recent_complete_pairs(
    pairs: Sequence[HistoryPair],
    budget: int,
    render: Callable[[HistoryPair], T],
    cost: Callable[[T], int],
) -> list[T]:
    """Keep the newest complete pairs that fit; never split a Q/A pair."""
    if budget <= 0:
        return []
    selected: list[T] = []
    used = 0
    for pair in reversed(pairs):
        rendered = render(pair)
        item_cost = cost(rendered)
        if used + item_cost > budget:
            break
        selected.append(rendered)
        used += item_cost
    selected.reverse()
    return selected


def build_retrieval_query(
    pairs: Sequence[HistoryPair], current_question: str, limit: int
) -> str:
    """Combine the current question with recent complete pairs for retrieval."""
    current = f"CURRENT QUESTION\n{current_question}"
    remaining = max(0, limit - len(current))

    def render(pair: HistoryPair) -> str:
        return f"user: {pair.question}\nassistant: {pair.answer}"

    dialogue = _recent_complete_pairs(pairs, remaining, render, lambda item: len(item) + 1)
    if not dialogue:
        return current
    return f"{current}\nPRIOR DIALOGUE\n" + "\n".join(dialogue)


def build_rewrite_history(pairs: Sequence[HistoryPair], limit: int) -> str:
    """Render bounded complete Q/A pairs for the rewrite model."""
    def render(pair: HistoryPair) -> str:
        return f"user: {pair.question}\nassistant: {pair.answer}"

    dialogue = _recent_complete_pairs(pairs, limit, render, lambda item: len(item) + 1)
    return "\n".join(dialogue)


def has_query_rewrite_trigger(current_question: str) -> bool:
    """Return whether the current question contains a history reference."""
    return any(pattern.search(current_question) for pattern in _REWRITE_TRIGGER_PATTERNS)


def needs_query_rewrite(pairs: Sequence[HistoryPair], current_question: str) -> bool:
    """Return whether rewrite can run because a trigger and history exist."""
    return bool(pairs) and has_query_rewrite_trigger(current_question)


def build_answer_history(pairs: Sequence[HistoryPair], limit: int) -> list[dict[str, str]]:
    """Convert recent complete pairs to ordered OpenAI-compatible messages."""

    def render(pair: HistoryPair) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.answer},
        )

    selected = _recent_complete_pairs(
        pairs,
        limit,
        render,
        lambda messages: len(messages[0]["content"]) + len(messages[1]["content"]),
    )
    return [message for pair in selected for message in pair]
