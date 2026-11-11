"""Dependency-light BM25 and weighted reciprocal-rank fusion."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any, Iterable, Sequence


def default_tokenize(text: str) -> list[str]:
    """Tokenize technical English and individual CJK characters deterministically."""
    tokens: list[str] = []
    for part in re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE):
        if any("CJK" in unicodedata.name(char, "") for char in part):
            tokens.extend(char.casefold() for char in part if not char.isspace())
        else:
            tokens.append(part.casefold())
    return tokens


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)


class BM25Index:
    def __init__(self, documents: Iterable[dict[str, Any]], *, k1: float = 1.2, b: float = 0.75):
        self.documents = tuple(documents)
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = float(k1), float(b)
        self._tf: list[Counter[str]] = []
        self._lengths: list[int] = []
        df: Counter[str] = Counter()
        for row in self.documents:
            # ``retrieval_text`` is an offline lexical projection. Keep the
            # fallback so older immutable corpus bundles remain readable until
            # a new version is built.
            terms = default_tokenize(row.get("retrieval_text") or row["text"])
            frequencies = Counter(terms)
            self._tf.append(frequencies)
            self._lengths.append(len(terms))
            df.update(frequencies.keys())
        self._avg = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        n = len(self.documents)
        self._idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, k: int = 20) -> list[SearchResult]:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be positive")
        query_terms = Counter(default_tokenize(query))
        scored = []
        average = self._avg or 1.0
        for index, row in enumerate(self.documents):
            frequencies = self._tf[index]
            length = self._lengths[index]
            normalizer = self.k1 * (1 - self.b + self.b * length / average)
            score = 0.0
            for term, qtf in query_terms.items():
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += qtf * self._idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1) / (frequency + normalizer)
                    )
            scored.append((score, index, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"], text=row["text"],
                score=float(score), rank=rank, metadata=dict(row),
                component_scores={"bm25": float(score)},
            )
            for rank, (score, _, row) in enumerate(scored[:k], 1)
        ]


def weighted_rrf(
    lexical: list[SearchResult], dense: list[SearchResult], *, k: int = 5,
    lexical_weight: float = 0.5, dense_weight: float = 0.5, rrf_k: int = 60,
) -> list[SearchResult]:
    if k < 1 or rrf_k < 1 or lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
        raise ValueError("invalid RRF parameters")
    total = lexical_weight + dense_weight
    weights = (("bm25", lexical_weight / total, lexical), ("e5", dense_weight / total, dense))
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    exemplars: dict[str, SearchResult] = {}
    order: dict[str, int] = {}
    for name, weight, results in weights:
        for position, result in enumerate(results, 1):
            contribution = weight / (rrf_k + position)
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + contribution
            components.setdefault(result.chunk_id, {})[name] = contribution
            exemplars.setdefault(result.chunk_id, result)
            order.setdefault(result.chunk_id, len(order))
    ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], order[chunk_id]))[:k]
    return [
        SearchResult(
            chunk_id=chunk_id, doc_id=exemplars[chunk_id].doc_id, text=exemplars[chunk_id].text,
            score=fused[chunk_id], rank=rank, metadata=dict(exemplars[chunk_id].metadata),
            component_scores=components[chunk_id],
        )
        for rank, chunk_id in enumerate(ranked, 1)
    ]


def weighted_rrf_many(
    ranked_lists: Sequence[tuple[str, list[SearchResult], float]],
    *,
    k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse any number of ranked result lists with deterministic weights.

    The weights are normalized so callers can use either proportions or
    percentages. A result keeps the metadata from its first occurrence and
    receives a stable tie-break based on first-seen order.
    """
    if k < 1 or rrf_k < 1 or not ranked_lists:
        raise ValueError("invalid weighted RRF arguments")
    if any(weight < 0 for _, _, weight in ranked_lists):
        raise ValueError("RRF weights must be non-negative")
    total_weight = sum(weight for _, _, weight in ranked_lists)
    if total_weight <= 0:
        raise ValueError("at least one RRF weight must be positive")

    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    exemplars: dict[str, SearchResult] = {}
    first_seen: dict[str, int] = {}
    for name, results, weight in ranked_lists:
        normalized_weight = weight / total_weight
        for position, result in enumerate(results, 1):
            contribution = normalized_weight / (rrf_k + position)
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + contribution
            components.setdefault(result.chunk_id, {})[name] = contribution
            exemplars.setdefault(result.chunk_id, result)
            first_seen.setdefault(result.chunk_id, len(first_seen))

    ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], first_seen[chunk_id]))[:k]
    return [
        SearchResult(
            chunk_id=chunk_id,
            doc_id=exemplars[chunk_id].doc_id,
            text=exemplars[chunk_id].text,
            score=fused[chunk_id],
            rank=rank,
            metadata=dict(exemplars[chunk_id].metadata),
            component_scores=components[chunk_id],
        )
        for rank, chunk_id in enumerate(ranked, 1)
    ]
