"""Dependency-free retrieval baselines and pluggable retrieval interfaces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Document:
    """A retrieval unit; normally one immutable corpus chunk."""

    chunk_id: str
    text: str
    doc_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A normalized result emitted by every retriever in this package."""

    chunk_id: str
    score: float
    rank: int
    doc_id: str | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    component_scores: Mapping[str, float] = field(default_factory=dict)

    def as_run_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "rank": self.rank,
            "score": self.score,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_spans": _normalise_run_source_spans(
                self.metadata.get("source_spans", ())
            ),
        }
        if self.component_scores:
            item["component_scores"] = dict(self.component_scores)
        return item


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface for local indexes and remote retrieval services."""

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[SearchResult]: ...


@runtime_checkable
class DenseEncoder(Protocol):
    """Adapter boundary for sentence-transformers, APIs, or local models."""

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class CallableDenseEncoder:
    """Turn two ordinary callables into a :class:`DenseEncoder`."""

    def __init__(
        self,
        document_encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        query_encoder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self._document_encoder = document_encoder
        self._query_encoder = query_encoder or document_encoder

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._document_encoder(texts)

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._query_encoder(texts)


class CallableRetrieverAdapter:
    """Normalize an existing vector DB/search service callable.

    The callable may return :class:`SearchResult` objects or mappings with at
    least ``chunk_id`` and ``score``.  This keeps external SDK dependencies out
    of the core evaluation environment.
    """

    def __init__(
        self,
        search_function: Callable[
            [str, int, Mapping[str, Any] | None],
            Sequence[SearchResult | Mapping[str, Any]],
        ],
    ) -> None:
        self._search_function = search_function

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        _validate_k(k)
        raw = self._search_function(query, k, filters)
        results = [_coerce_result(item, rank) for rank, item in enumerate(raw, 1)]
        return _deduplicate_and_rank(results, k)


class HashingDenseEncoder:
    """Deterministic non-neural feature-hashing encoder.

    This adapter provides a runnable dense smoke baseline without downloading a
    model.  It hashes Unicode word/CJK features and character n-grams into a
    fixed-size vector with signed hashing.  It measures lexical/form similarity,
    not learned semantic equivalence, and must be labelled ``dense-hash`` in
    reports rather than presented as an embedding-model result.
    """

    def __init__(
        self,
        dimensions: int = 512,
        *,
        min_char_ngram: int = 3,
        max_char_ngram: int = 5,
    ) -> None:
        if dimensions < 16:
            raise ValueError("hash dimensions must be at least 16")
        if min_char_ngram < 1 or max_char_ngram < min_char_ngram:
            raise ValueError("invalid character n-gram range")
        self.dimensions = dimensions
        self.min_char_ngram = min_char_ngram
        self.max_char_ngram = max_char_ngram

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        features: Counter[str] = Counter(f"w:{token}" for token in default_tokenize(text))
        compact = " ".join(default_tokenize(text))
        padded = f"^{compact}$"
        for width in range(self.min_char_ngram, self.max_char_ngram + 1):
            for index in range(max(0, len(padded) - width + 1)):
                features[f"c{width}:{padded[index:index + width]}"] += 1
        for feature, frequency in features.items():
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=9, person=b"qegs-hash"
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[bucket] += sign * (1.0 + math.log(frequency))
        return vector


class BM25Retriever:
    """A small, deterministic Okapi BM25 implementation.

    IDF uses ``log(1 + (N - df + 0.5) / (df + 0.5))`` so terms never receive
    negative IDF.  The default tokenizer is Unicode-aware and emits individual
    CJK characters, making the no-dependency baseline usable on mixed-language
    technical documents.  It is intentionally a baseline, not a replacement
    for a language-specific tokenizer.
    """

    def __init__(
        self,
        documents: Sequence[Document | Mapping[str, Any]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or default_tokenize
        self.documents = tuple(_coerce_document(item) for item in documents)
        _ensure_unique_document_ids(self.documents)

        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            terms = list(self.tokenizer(document.text))
            frequencies = Counter(terms)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(terms))
            document_frequency.update(frequencies.keys())

        count = len(self.documents)
        self._average_document_length = (
            sum(self._document_lengths) / count if count else 0.0
        )
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        _validate_k(k)
        query_terms = Counter(self.tokenizer(query))
        scored: list[tuple[float, int, Document]] = []
        for index, document in enumerate(self.documents):
            if not _matches_filters(document, filters):
                continue
            score = self._score(index, query_terms)
            scored.append((score, index, document))
        # Corpus order is the stable tie breaker.  Zero-score results are kept
        # so every query has a deterministic top-k run for recall accounting.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                chunk_id=document.chunk_id,
                doc_id=document.doc_id,
                text=document.text,
                score=score,
                rank=rank,
                metadata=document.metadata,
                component_scores={"bm25": score},
            )
            for rank, (score, _, document) in enumerate(scored[:k], start=1)
        ]

    def _score(self, index: int, query_terms: Counter[str]) -> float:
        if not self.documents or not query_terms:
            return 0.0
        frequencies = self._term_frequencies[index]
        length = self._document_lengths[index]
        average = self._average_document_length or 1.0
        normalizer = self.k1 * (1.0 - self.b + self.b * length / average)
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            term_score = self._idf[term] * (
                frequency * (self.k1 + 1.0) / (frequency + normalizer)
            )
            # Basic BM25 commonly ignores qtf; multiplying provides sensible
            # deterministic behaviour for repeated identifiers in tech queries.
            score += query_frequency * term_score
        return score


class DenseRetriever:
    """Exact cosine-search baseline over embeddings from any encoder adapter."""

    def __init__(
        self,
        documents: Sequence[Document | Mapping[str, Any]],
        encoder: DenseEncoder,
    ) -> None:
        self.documents = tuple(_coerce_document(item) for item in documents)
        _ensure_unique_document_ids(self.documents)
        self.encoder = encoder
        vectors = encoder.encode_documents([item.text for item in self.documents])
        if len(vectors) != len(self.documents):
            raise ValueError("dense encoder returned the wrong number of document vectors")
        self._vectors = tuple(_normalise_vector(vector) for vector in vectors)
        dimensions = {len(vector) for vector in self._vectors}
        if len(dimensions) > 1:
            raise ValueError("all dense vectors must have the same dimension")

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        _validate_k(k)
        encoded = self.encoder.encode_queries([query])
        if len(encoded) != 1:
            raise ValueError("dense encoder must return exactly one query vector")
        query_vector = _normalise_vector(encoded[0])
        if self._vectors and len(query_vector) != len(self._vectors[0]):
            raise ValueError("query and document vectors have different dimensions")
        scored: list[tuple[float, int, Document]] = []
        for index, (document, vector) in enumerate(zip(self.documents, self._vectors)):
            if not _matches_filters(document, filters):
                continue
            score = sum(left * right for left, right in zip(query_vector, vector))
            scored.append((score, index, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                chunk_id=document.chunk_id,
                doc_id=document.doc_id,
                text=document.text,
                score=score,
                rank=rank,
                metadata=document.metadata,
                component_scores={"dense": score},
            )
            for rank, (score, _, document) in enumerate(scored[:k], start=1)
        ]


class HybridRetriever:
    """Weighted reciprocal-rank fusion over lexical and dense retrievers."""

    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        *,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if lexical_weight < 0 or dense_weight < 0:
            raise ValueError("hybrid weights must be non-negative")
        if lexical_weight + dense_weight == 0:
            raise ValueError("at least one hybrid weight must be positive")
        if rrf_k < 1 or candidate_multiplier < 1:
            raise ValueError("rrf_k and candidate_multiplier must be positive")
        self.lexical = lexical
        self.dense = dense
        total = lexical_weight + dense_weight
        self.lexical_weight = lexical_weight / total
        self.dense_weight = dense_weight / total
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        _validate_k(k)
        candidate_k = max(k, k * self.candidate_multiplier)
        lexical = self.lexical.search(query, candidate_k, filters)
        dense = self.dense.search(query, candidate_k, filters)

        fused: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        exemplars: dict[str, SearchResult] = {}
        order: dict[str, int] = {}
        for name, weight, results in (
            ("lexical_rrf", self.lexical_weight, lexical),
            ("dense_rrf", self.dense_weight, dense),
        ):
            for position, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                contribution = weight / (self.rrf_k + position)
                fused[chunk_id] = fused.get(chunk_id, 0.0) + contribution
                components.setdefault(chunk_id, {})[name] = contribution
                exemplars.setdefault(chunk_id, result)
                order.setdefault(chunk_id, len(order))
        ranked_ids = sorted(fused, key=lambda item: (-fused[item], order[item]))[:k]
        return [
            SearchResult(
                chunk_id=chunk_id,
                doc_id=exemplars[chunk_id].doc_id,
                text=exemplars[chunk_id].text,
                score=fused[chunk_id],
                rank=rank,
                metadata=exemplars[chunk_id].metadata,
                component_scores=components[chunk_id],
            )
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ]


class LexicalRerankingRetriever:
    """Non-neural query-document pair reranker over retrieved candidates.

    The score combines query-token coverage, token F1, ordered-bigram coverage,
    exact-phrase presence, and a small reciprocal-rank prior.  It is a useful
    deterministic integration baseline, but it is **not** a cross-encoder and
    must be reported as ``hybrid-rerank (non-neural lexical pair scorer)``.
    """

    def __init__(
        self,
        candidate_retriever: Retriever,
        *,
        candidate_multiplier: int = 4,
        rank_prior_weight: float = 0.15,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if not 0 <= rank_prior_weight <= 1:
            raise ValueError("rank_prior_weight must be in [0, 1]")
        self.candidate_retriever = candidate_retriever
        self.candidate_multiplier = candidate_multiplier
        self.rank_prior_weight = rank_prior_weight
        self.tokenizer = tokenizer or default_tokenize

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        _validate_k(k)
        candidates = self.candidate_retriever.search(
            query, max(k, k * self.candidate_multiplier), filters
        )
        scored: list[tuple[float, int, SearchResult, float]] = []
        for candidate in candidates:
            pair_score = self._pair_score(query, candidate.text or "")
            rank_prior = 1.0 / candidate.rank
            score = (
                (1.0 - self.rank_prior_weight) * pair_score
                + self.rank_prior_weight * rank_prior
            )
            scored.append((score, candidate.rank, candidate, pair_score))
        scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        return [
            SearchResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                text=candidate.text,
                score=score,
                rank=rank,
                metadata=candidate.metadata,
                component_scores={
                    **candidate.component_scores,
                    "lexical_pair": pair_score,
                    "candidate_rank_prior": 1.0 / candidate_rank,
                },
            )
            for rank, (score, candidate_rank, candidate, pair_score) in enumerate(
                scored[:k], start=1
            )
        ]

    def _pair_score(self, query: str, document: str) -> float:
        query_tokens = list(self.tokenizer(query))
        document_tokens = list(self.tokenizer(document))
        if not query_tokens or not document_tokens:
            return 0.0
        query_counts = Counter(query_tokens)
        document_counts = Counter(document_tokens)
        overlap = sum(
            min(count, document_counts.get(token, 0))
            for token, count in query_counts.items()
        )
        query_coverage = overlap / sum(query_counts.values())
        precision = overlap / sum(document_counts.values())
        token_f1 = (
            2.0 * query_coverage * precision / (query_coverage + precision)
            if query_coverage + precision
            else 0.0
        )
        query_bigrams = list(zip(query_tokens, query_tokens[1:]))
        document_bigrams = set(zip(document_tokens, document_tokens[1:]))
        bigram_coverage = (
            sum(item in document_bigrams for item in query_bigrams) / len(query_bigrams)
            if query_bigrams
            else query_coverage
        )
        normalised_query = " ".join(query_tokens)
        normalised_document = " ".join(document_tokens)
        exact_phrase = float(normalised_query in normalised_document)
        return (
            0.55 * query_coverage
            + 0.20 * token_f1
            + 0.20 * bigram_coverage
            + 0.05 * exact_phrase
        )


def default_tokenize(text: str) -> list[str]:
    """Case-fold Unicode words and split CJK runs into individual characters."""

    tokens: list[str] = []
    latin_buffer: list[str] = []

    def flush() -> None:
        if latin_buffer:
            tokens.append("".join(latin_buffer))
            latin_buffer.clear()

    for character in text.casefold():
        if _is_cjk(character):
            flush()
            tokens.append(character)
        elif character.isalnum():
            latin_buffer.append(character)
        else:
            flush()
    flush()
    return tokens


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _coerce_document(value: Document | Mapping[str, Any]) -> Document:
    if isinstance(value, Document):
        return value
    chunk_id = value.get("chunk_id")
    if chunk_id is None:
        raise ValueError("every document must have chunk_id")
    text = value.get("text", value.get("content"))
    if text is None:
        raise ValueError(f"document {chunk_id!r} has neither text nor content")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("document metadata must be a mapping")
    return Document(
        chunk_id=str(chunk_id),
        doc_id=str(value["doc_id"]) if value.get("doc_id") is not None else None,
        text=str(text),
        metadata=metadata,
    )


def _coerce_result(value: SearchResult | Mapping[str, Any], rank: int) -> SearchResult:
    if isinstance(value, SearchResult):
        return value
    chunk_id = value.get("chunk_id")
    if chunk_id is None:
        raise ValueError("external search result is missing chunk_id")
    component_scores = value.get("component_scores") or {}
    metadata = dict(value.get("metadata") or {})
    if "source_spans" in value:
        metadata["source_spans"] = value.get("source_spans") or []
    return SearchResult(
        chunk_id=str(chunk_id),
        doc_id=str(value["doc_id"]) if value.get("doc_id") is not None else None,
        text=str(value["text"]) if value.get("text") is not None else None,
        score=float(value.get("score", 0.0)),
        rank=int(value.get("rank", rank)),
        metadata=metadata,
        component_scores=component_scores,
    )


def _deduplicate_and_rank(results: Iterable[SearchResult], k: int) -> list[SearchResult]:
    by_chunk: dict[str, SearchResult] = {}
    for result in results:
        previous = by_chunk.get(result.chunk_id)
        if previous is None or result.score > previous.score:
            by_chunk[result.chunk_id] = result
    ordered = sorted(by_chunk.values(), key=lambda item: (-item.score, item.rank))[:k]
    return [
        SearchResult(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            text=item.text,
            score=item.score,
            rank=rank,
            metadata=item.metadata,
            component_scores=item.component_scores,
        )
        for rank, item in enumerate(ordered, 1)
    ]


def _normalise_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("dense vectors must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("dense vectors must contain only finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _normalise_run_source_spans(values: Any) -> list[dict[str, Any]]:
    """Translate corpus source spans to the compact retrieval-run contract."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        start = value.get("start_char", value.get("source_start_char"))
        end = value.get("end_char", value.get("source_end_char"))
        block_id = value.get("block_id")
        if block_id is None or start is None or end is None:
            continue
        item: dict[str, Any] = {
            "block_id": str(block_id),
            "start_char": int(start),
            "end_char": int(end),
        }
        result.append(item)
    return result


def _matches_filters(
    document: Document, filters: Mapping[str, Any] | None
) -> bool:
    if not filters:
        return True
    fields = {"chunk_id": document.chunk_id, "doc_id": document.doc_id}
    fields.update(document.metadata)
    return all(fields.get(key) == expected for key, expected in filters.items())


def _ensure_unique_document_ids(documents: Sequence[Document]) -> None:
    ids = [item.chunk_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id values must be unique in a retrieval corpus")


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")


__all__ = [
    "BM25Retriever",
    "CallableDenseEncoder",
    "CallableRetrieverAdapter",
    "DenseEncoder",
    "DenseRetriever",
    "Document",
    "HybridRetriever",
    "HashingDenseEncoder",
    "LexicalRerankingRetriever",
    "Retriever",
    "SearchResult",
    "default_tokenize",
]
