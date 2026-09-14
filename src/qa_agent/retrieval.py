"""Dense retrieval adapter. No LLM, Markdown, parsing, or answer logic."""
from __future__ import annotations


class GlobalDenseRetriever:
    """Explicit full-corpus baseline; never used as a failed-route fallback."""
    def __init__(self, store, encoder):
        self.store, self.encoder = store, encoder

    def search(self, question: str, *, k: int = 20) -> list[dict]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be non-empty")
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        return self.store.search(self.encoder.encode_queries([question])[0], scope=None, k=k)


class ScopedDenseRetriever:
    def __init__(self, store, encoder):
        self.store, self.encoder = store, encoder

    def search(self, question: str, scope: dict, *, k: int = 20) -> list[dict]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be non-empty")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be positive")
        # A failed/empty route must never silently mean a full-corpus search.
        if not isinstance(scope, dict) or not scope.get("doc_ids") or not scope.get("section_ids"):
            raise ValueError("Scoped dense retrieval requires document AND section IDs")
        self.store.validate_scope(scope)  # Reject invalid IDs before encoding.
        if any(self.store.sections[s]["doc_id"] not in scope["doc_ids"] for s in scope["section_ids"]):
            raise ValueError("Selected section does not belong to selected documents")
        vector = self.encoder.encode_queries([question])[0]
        return self.store.search(vector, scope=scope, k=k)
