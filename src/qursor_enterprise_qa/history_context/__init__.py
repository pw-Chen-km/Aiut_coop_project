"""Stateless conversation-history parsing and budget selection."""

from .core import (
    HistoryPair,
    build_answer_history,
    build_retrieval_query,
    build_rewrite_history,
    has_query_rewrite_trigger,
    needs_query_rewrite,
)

__all__ = [
    "HistoryPair",
    "build_answer_history",
    "build_retrieval_query",
    "build_rewrite_history",
    "has_query_rewrite_trigger",
    "needs_query_rewrite",
]
