"""Evidence-aware retrieval metrics for the QURSOR RAG evaluation set.

The public functions in this module are deliberately dependency-free.  The
high-level :func:`evaluate_query` adapter understands the dataset schema used
by this project, while the small metric functions can also be used on their
own in notebooks or another evaluation runner.

An important design choice is that gold evidence is identified by a canonical
``evidence_id`` (normally a source span), not by a mutable retrieval chunk.
Retrieved chunks are mapped back to canonical evidence in this order:

1. explicit evidence ids on the retrieved item;
2. overlap with an explicitly returned source span;
3. qrels evidence ids/equivalence groups;
4. chunk-id fallback, only when no more precise signal is available.

The final fallback is useful for a first baseline but can over-credit a large
chunk.  Production reports should therefore include the matching diagnostics
returned by :func:`evaluate_query`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


MetricValue = float | int | None


@dataclass(frozen=True)
class SourceSpan:
    """A half-open character span within a stable source block."""

    block_id: str | None
    start_char: int | None
    end_char: int | None
    coordinate_space: str = "block_text"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceSpan":
        return cls(
            block_id=_optional_str(value.get("block_id")),
            start_char=_optional_int(value.get("start_char")),
            end_char=_optional_int(value.get("end_char")),
            coordinate_space=str(value.get("coordinate_space", "block_text")),
        )


@dataclass(frozen=True)
class EvidenceRef:
    """Canonical gold evidence and its provenance."""

    evidence_id: str
    source_atom_id: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    block_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    coordinate_space: str = "chunk_text"
    equivalence_group: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        # ``source_span`` is stable block-text provenance used for retrieval
        # scoring. ``span`` is the chunk-text coordinate retained for exact
        # quote resolution and is only comparable to a chunk-text run span.
        source_span = value.get("source_span")
        span = source_span or value.get("span") or {}
        if not isinstance(span, Mapping):
            span = {}
        evidence_id = value.get("evidence_id")
        if evidence_id is None:
            raise ValueError("every evidence record must have evidence_id")
        return cls(
            evidence_id=str(evidence_id),
            source_atom_id=_optional_str(value.get("source_atom_id")),
            doc_id=_optional_str(value.get("doc_id")),
            chunk_id=_optional_str(value.get("chunk_id")),
            block_id=_optional_str(value.get("block_id")),
            start_char=_optional_int(span.get("start_char")),
            end_char=_optional_int(span.get("end_char")),
            coordinate_space=str(
                span.get(
                    "coordinate_space",
                    "block_text" if source_span else "chunk_text",
                )
            ),
            equivalence_group=_optional_str(value.get("equivalence_group")),
        )


@dataclass(frozen=True)
class RetrievedRef:
    """A normalized item from a ranked retrieval run."""

    rank: int
    chunk_id: str
    score: float | None = None
    doc_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    equivalence_groups: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], fallback_rank: int
    ) -> "RetrievedRef":
        chunk_id = value.get("chunk_id", value.get("item_id"))
        if chunk_id is None:
            raise ValueError("every retrieved item must have chunk_id")
        spans_value = value.get("source_spans") or ()
        spans = tuple(
            SourceSpan.from_mapping(span)
            for span in spans_value
            if isinstance(span, Mapping)
        )
        parsed_rank = _optional_int(value.get("rank"))
        rank = parsed_rank if parsed_rank is not None else fallback_rank
        if rank < 1:
            raise ValueError("retrieval ranks are one-based and must be positive")
        score = value.get("score")
        return cls(
            rank=rank,
            chunk_id=str(chunk_id),
            score=float(score) if score is not None else None,
            doc_id=_optional_str(value.get("doc_id")),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", ())),
            equivalence_groups=tuple(
                str(item) for item in value.get("equivalence_groups", ())
            ),
            source_spans=spans,
        )


class EvidenceMatcher:
    """Map mutable retrieval chunks to stable canonical evidence spans."""

    def __init__(
        self,
        evidence: Sequence[EvidenceRef],
        qrels: Sequence[Mapping[str, Any]] = (),
        *,
        qid: str | None = None,
        min_span_coverage: float = 0.5,
    ) -> None:
        if not 0 < min_span_coverage <= 1:
            raise ValueError("min_span_coverage must be in (0, 1]")
        self.evidence = tuple(evidence)
        self.min_span_coverage = min_span_coverage
        self.by_id = {item.evidence_id: item for item in evidence}
        if len(self.by_id) != len(self.evidence):
            raise ValueError("evidence_id values must be unique within a QA record")

        self.by_chunk: dict[str, set[str]] = {}
        self.by_group: dict[str, set[str]] = {}
        for item in evidence:
            if item.chunk_id:
                self.by_chunk.setdefault(item.chunk_id, set()).add(item.evidence_id)
            if item.equivalence_group:
                self.by_group.setdefault(item.equivalence_group, set()).add(
                    item.evidence_id
                )

        self.qrels_by_chunk: dict[str, list[Mapping[str, Any]]] = {}
        for qrel in qrels:
            qrel_qid = qrel.get("qid")
            if qid is not None and qrel_qid is not None and str(qrel_qid) != qid:
                continue
            chunk_id = qrel.get("chunk_id")
            if chunk_id is not None:
                self.qrels_by_chunk.setdefault(str(chunk_id), []).append(qrel)

    def match(self, item: RetrievedRef) -> tuple[frozenset[str], str]:
        """Return matched evidence ids and the most precise method used."""

        explicit = {eid for eid in item.evidence_ids if eid in self.by_id}
        explicit.update(self._ids_for_groups(item.equivalence_groups))
        if explicit:
            return frozenset(self._expand_equivalents(explicit)), "explicit_evidence"

        # A returned source span is treated as authoritative.  If it misses the
        # gold span, we intentionally do not fall back to the containing chunk.
        if item.source_spans:
            span_matches: set[str] = set()
            comparable_span_exists = False
            for candidate in self.evidence:
                if item.doc_id and candidate.doc_id and item.doc_id != candidate.doc_id:
                    continue
                for span in item.source_spans:
                    if not self._spans_are_comparable(candidate, span, item):
                        continue
                    comparable_span_exists = True
                    if self._span_matches(candidate, span):
                        span_matches.add(candidate.evidence_id)
            if span_matches or comparable_span_exists:
                return (
                    frozenset(self._expand_equivalents(span_matches)),
                    "source_span" if span_matches else "source_span_miss",
                )

        qrel_matches: set[str] = set()
        for qrel in self.qrels_by_chunk.get(item.chunk_id, ()):
            # Only relevance=2 qrels are exact evidence mappings.  Relevance=1
            # denotes helpful-but-insufficient context and contributes to
            # graded ranking metrics, never evidence-set completion.
            if float(qrel.get("relevance", 0)) < 2:
                continue
            qrel_matches.update(
                eid for eid in qrel.get("evidence_ids", ()) if eid in self.by_id
            )
            group = qrel.get("equivalence_group")
            if group is not None:
                qrel_matches.update(self.by_group.get(str(group), ()))
        if qrel_matches:
            return frozenset(self._expand_equivalents(qrel_matches)), "qrels"

        chunk_matches = self.by_chunk.get(item.chunk_id, set())
        return (
            frozenset(self._expand_equivalents(chunk_matches)),
            "chunk_fallback" if chunk_matches else "unmatched",
        )

    def match_ranked(
        self, items: Sequence[RetrievedRef]
    ) -> list[tuple[frozenset[str], str]]:
        """Match a ranked list, unioning partial source spans across chunks.

        This is the granularity-independent path: if a future preprocessor
        divides one gold source atom into several retrieval chunks, the atom is
        credited at the first rank where their block-coordinate interval union
        reaches ``min_span_coverage``.  Overlapping chunks never double count.
        """

        intervals_by_evidence: dict[str, list[tuple[int, int]]] = {}
        union_satisfied: set[str] = set()
        result: list[tuple[frozenset[str], str]] = []
        for item in items:
            matched, method = self.match(item)
            newly_satisfied: set[str] = set()
            if item.source_spans:
                for evidence in self.evidence:
                    if item.doc_id and evidence.doc_id and item.doc_id != evidence.doc_id:
                        continue
                    for span in item.source_spans:
                        if not self._spans_are_comparable(evidence, span, item):
                            continue
                        overlap = self._span_overlap(evidence, span)
                        if overlap is not None:
                            intervals_by_evidence.setdefault(
                                evidence.evidence_id, []
                            ).append(overlap)
                    if (
                        evidence.evidence_id not in union_satisfied
                        and self._interval_union_coverage(
                            evidence, intervals_by_evidence.get(evidence.evidence_id, ())
                        )
                        >= self.min_span_coverage
                    ):
                        union_satisfied.add(evidence.evidence_id)
                        newly_satisfied.add(evidence.evidence_id)
            union_only = newly_satisfied - set(matched)
            if union_only:
                matched = frozenset(
                    self._expand_equivalents(set(matched) | union_only)
                )
                method = "source_span_union"
            result.append((matched, method))
        return result

    def _span_matches(self, evidence: EvidenceRef, retrieved: SourceSpan) -> bool:
        if (
            evidence.start_char is None
            or evidence.end_char is None
            or retrieved.start_char is None
            or retrieved.end_char is None
        ):
            return False
        evidence_length = evidence.end_char - evidence.start_char
        if evidence_length <= 0 or retrieved.end_char <= retrieved.start_char:
            return False
        overlap = max(
            0,
            min(evidence.end_char, retrieved.end_char)
            - max(evidence.start_char, retrieved.start_char),
        )
        return overlap / evidence_length >= self.min_span_coverage

    @staticmethod
    def _span_overlap(
        evidence: EvidenceRef, retrieved: SourceSpan
    ) -> tuple[int, int] | None:
        if (
            evidence.start_char is None
            or evidence.end_char is None
            or retrieved.start_char is None
            or retrieved.end_char is None
        ):
            return None
        start = max(evidence.start_char, retrieved.start_char)
        end = min(evidence.end_char, retrieved.end_char)
        return (start, end) if end > start else None

    @staticmethod
    def _interval_union_coverage(
        evidence: EvidenceRef, intervals: Sequence[tuple[int, int]]
    ) -> float:
        if evidence.start_char is None or evidence.end_char is None or not intervals:
            return 0.0
        length = evidence.end_char - evidence.start_char
        if length <= 0:
            return 0.0
        ordered = sorted(intervals)
        covered = 0
        current_start, current_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end
        covered += current_end - current_start
        return covered / length

    @staticmethod
    def _spans_are_comparable(
        evidence: EvidenceRef, retrieved: SourceSpan, item: RetrievedRef
    ) -> bool:
        if evidence.coordinate_space != retrieved.coordinate_space:
            return False
        if evidence.coordinate_space == "chunk_text":
            return evidence.chunk_id == item.chunk_id
        if evidence.coordinate_space == "block_text":
            return bool(
                evidence.block_id
                and retrieved.block_id
                and evidence.block_id == retrieved.block_id
            )
        return False

    def _ids_for_groups(self, groups: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for group in groups:
            result.update(self.by_group.get(str(group), ()))
        return result

    def _expand_equivalents(self, evidence_ids: Iterable[str]) -> set[str]:
        result = set(evidence_ids)
        for evidence_id in tuple(result):
            evidence = self.by_id.get(evidence_id)
            if evidence and evidence.equivalence_group:
                result.update(self.by_group.get(evidence.equivalence_group, ()))
        return result


def evidence_set_recall_at_k(
    gold_evidence_sets: Sequence[Iterable[str]],
    ranked_retrieved_evidence: Sequence[Iterable[str] | str],
    k: int,
    equivalence_groups: Mapping[str, str] | None = None,
) -> float | None:
    """Best recall among explicitly allowed alternative gold evidence sets.

    ``None`` is returned when there is no positive evidence set.  This avoids
    inflating retrieval averages with unanswerable questions.
    """

    _validate_k(k)
    gold_sets = _normalise_sets(gold_evidence_sets)
    if not gold_sets:
        return None
    retrieved = _flatten_ranked(ranked_retrieved_evidence, k)
    return max(
        _set_recall(gold, retrieved, equivalence_groups) for gold in gold_sets
    )


def complete_evidence_all_hops_at_k(
    gold_evidence_sets: Sequence[Iterable[str]],
    ranked_retrieved_evidence: Sequence[Iterable[str] | str],
    k: int,
    *,
    hop_requirements: Mapping[str, Iterable[str]] | None = None,
    equivalence_groups: Mapping[str, str] | None = None,
) -> int | None:
    """Whether one complete gold set and every declared hop were retrieved."""

    recall = evidence_set_recall_at_k(
        gold_evidence_sets,
        ranked_retrieved_evidence,
        k,
        equivalence_groups,
    )
    if recall is None:
        return None
    if recall < 1.0:
        return 0
    if hop_requirements:
        per_hop = per_hop_recall_at_k(
            hop_requirements,
            ranked_retrieved_evidence,
            k,
            equivalence_groups,
        )
        if any(value < 1.0 for value in per_hop.values()):
            return 0
    return 1


def source_recall_at_k(
    gold_source_sets: Sequence[Iterable[str]],
    ranked_source_ids: Sequence[str | None],
    k: int,
) -> float | None:
    """Recall over unique source/document ids, respecting alternative sets."""

    return _categorical_recall_at_k(gold_source_sets, ranked_source_ids, k)


def chunk_recall_at_k(
    gold_chunk_sets: Sequence[Iterable[str]],
    ranked_chunk_ids: Sequence[str | None],
    k: int,
) -> float | None:
    """Recall over unique gold retrieval chunks."""

    return _categorical_recall_at_k(gold_chunk_sets, ranked_chunk_ids, k)


def per_hop_recall_at_k(
    hop_requirements: Mapping[str, Iterable[str]],
    ranked_retrieved_evidence: Sequence[Iterable[str] | str],
    k: int,
    equivalence_groups: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Evidence recall for every reasoning step/hop."""

    _validate_k(k)
    retrieved = _flatten_ranked(ranked_retrieved_evidence, k)
    result: dict[str, float] = {}
    for hop_id, required in hop_requirements.items():
        required_set = frozenset(str(item) for item in required)
        if not required_set:
            continue
        result[str(hop_id)] = _set_recall(
            required_set, retrieved, equivalence_groups
        )
    return result


def mrr_at_k(ranked_relevances: Sequence[float | int], k: int = 10) -> float:
    """Reciprocal rank of the first item with relevance greater than zero."""

    _validate_k(k)
    for rank, relevance in enumerate(ranked_relevances[:k], start=1):
        if relevance > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_relevances: Sequence[float | int],
    ideal_relevances: Sequence[float | int] | None = None,
    k: int = 10,
) -> float:
    """Graded nDCG using the standard exponential gain formulation."""

    _validate_k(k)
    observed = [max(0.0, float(value)) for value in ranked_relevances[:k]]
    ideal_source = ideal_relevances if ideal_relevances is not None else observed
    ideal = sorted(
        (max(0.0, float(value)) for value in ideal_source), reverse=True
    )[:k]
    denominator = _dcg(ideal)
    if denominator == 0:
        return 0.0
    return _dcg(observed) / denominator


def hard_negative_intrusion_at_k(
    hard_negative_chunk_ids: Iterable[str],
    ranked_chunk_ids: Sequence[str | None],
    k: int,
) -> float:
    """Fraction of returned top-k slots occupied by labelled hard negatives."""

    _validate_k(k)
    top_k = [str(item) for item in ranked_chunk_ids[:k] if item is not None]
    if not top_k:
        return 0.0
    hard_negatives = {str(item) for item in hard_negative_chunk_ids}
    return sum(item in hard_negatives for item in top_k) / len(top_k)


def evaluate_query(
    qa_record: Mapping[str, Any],
    retrieved_items: Sequence[Mapping[str, Any]],
    *,
    qrels: Sequence[Mapping[str, Any]] = (),
    cutoffs: Sequence[int] = (1, 5, 10, 20),
    min_span_coverage: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one project-schema QA record against a ranked retrieval run."""

    qid = str(qa_record.get("qid", ""))
    evidence = tuple(
        EvidenceRef.from_mapping(item) for item in qa_record.get("evidence", ())
    )
    ranked = sorted(
        (
            RetrievedRef.from_mapping(item, fallback_rank=index)
            for index, item in enumerate(retrieved_items, start=1)
        ),
        key=lambda item: item.rank,
    )
    _ensure_unique_ranks_and_chunks(ranked)

    matcher = EvidenceMatcher(
        evidence,
        qrels,
        qid=qid or None,
        min_span_coverage=min_span_coverage,
    )
    matched = matcher.match_ranked(ranked)
    ranked_evidence = [ids for ids, _ in matched]
    match_methods = [method for _, method in matched]

    raw_gold_sets = qa_record.get("gold_evidence_sets") or ()
    representatives_by_group: dict[str, str] = {}
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        if item.equivalence_group:
            representatives_by_group.setdefault(
                item.equivalence_group, item.evidence_id
            )
    gold_sets = []
    for value in raw_gold_sets:
        if not isinstance(value, Mapping):
            continue
        required = [str(item) for item in value.get("required_evidence_ids", ())]
        for group in value.get("required_equivalence_groups", ()):
            representative = representatives_by_group.get(str(group))
            if representative is None:
                raise ValueError(
                    f"gold evidence set references unknown equivalence group {group!r}"
                )
            required.append(representative)
        if required:
            gold_sets.append(tuple(dict.fromkeys(required)))
    if not gold_sets and evidence and _is_answerable(qa_record):
        gold_sets = [tuple(item.evidence_id for item in evidence)]

    equivalence_groups = {
        item.evidence_id: item.equivalence_group
        for item in evidence
        if item.equivalence_group
    }
    hop_requirements = _hop_requirements(qa_record)

    evidence_by_id = {item.evidence_id: item for item in evidence}
    projected_gold_sets = _expand_equivalent_gold_sets(gold_sets, evidence_by_id)
    gold_source_sets = _project_gold_sets(projected_gold_sets, evidence_by_id, "doc_id")
    gold_chunk_sets = _project_gold_sets(projected_gold_sets, evidence_by_id, "chunk_id")
    ranked_sources = [item.doc_id for item in ranked]
    ranked_chunks = [item.chunk_id for item in ranked]

    hard_negative_records = qa_record.get("hard_negatives") or ()
    hard_negative_ids = {
        str(value.get("chunk_id"))
        for value in hard_negative_records
        if isinstance(value, Mapping) and value.get("chunk_id") is not None
    }
    hard_negative_types: dict[str, str] = {
        str(value["chunk_id"]): str(value.get("type", "unspecified"))
        for value in hard_negative_records
        if isinstance(value, Mapping) and value.get("chunk_id") is not None
    }

    metrics: dict[str, Any] = {"qid": qid}
    has_source_atoms = bool(evidence) and all(item.source_atom_id for item in evidence)
    for k in sorted(set(cutoffs)):
        _validate_k(k)
        per_hop = per_hop_recall_at_k(
            hop_requirements, ranked_evidence, k, equivalence_groups
        )
        metrics[f"evidence_set_recall@{k}"] = evidence_set_recall_at_k(
            gold_sets, ranked_evidence, k, equivalence_groups
        )
        metrics[f"complete_evidence_all_hops@{k}"] = (
            complete_evidence_all_hops_at_k(
                gold_sets,
                ranked_evidence,
                k,
                hop_requirements=hop_requirements,
                equivalence_groups=equivalence_groups,
            )
        )
        metrics[f"source_atom_recall@{k}"] = (
            metrics[f"evidence_set_recall@{k}"] if has_source_atoms else None
        )
        metrics[f"complete_source_atoms_all_hops@{k}"] = (
            metrics[f"complete_evidence_all_hops@{k}"] if has_source_atoms else None
        )
        metrics[f"source_recall@{k}"] = source_recall_at_k(
            gold_source_sets, ranked_sources, k
        )
        metrics[f"chunk_recall@{k}"] = chunk_recall_at_k(
            gold_chunk_sets, ranked_chunks, k
        )
        metrics[f"per_hop_recall@{k}"] = per_hop
        metrics[f"macro_hop_recall@{k}"] = (
            fmean(per_hop.values()) if per_hop else None
        )
        metrics[f"weakest_hop_recall@{k}"] = min(per_hop.values()) if per_hop else None
        metrics[f"hard_negative_intrusion_rate@{k}"] = (
            hard_negative_intrusion_at_k(hard_negative_ids, ranked_chunks, k)
        )
        top_k_chunks = ranked_chunks[:k]
        metrics[f"hard_negative_intrusion_count@{k}"] = sum(
            chunk_id in hard_negative_ids for chunk_id in top_k_chunks
        )

    qrel_grades = _qrel_grades(qrels, qid)
    positive_qrel_grades = [value for value in qrel_grades.values() if value > 0]
    if positive_qrel_grades:
        ranked_relevances = [qrel_grades.get(item.chunk_id, 0.0) for item in ranked]
        ideal_relevances = positive_qrel_grades
    elif gold_sets:
        ranked_relevances = [float(bool(ids)) for ids in ranked_evidence]
        ideal_relevances = [1.0] * min(
            (len(gold) for gold in gold_chunk_sets), default=1
        )
    else:
        ranked_relevances = []
        ideal_relevances = []
    metrics["mrr@10"] = (
        mrr_at_k(ranked_relevances, 10) if ideal_relevances else None
    )
    metrics["ndcg@10"] = (
        ndcg_at_k(ranked_relevances, ideal_relevances, 10)
        if ideal_relevances
        else None
    )

    intruding = [
        {
            "rank": item.rank,
            "chunk_id": item.chunk_id,
            "type": hard_negative_types.get(item.chunk_id, "unspecified"),
        }
        for item in ranked
        if item.chunk_id in hard_negative_ids
    ]
    metrics["hard_negative_first_rank"] = (
        intruding[0]["rank"] if intruding else None
    )
    metrics["diagnostics"] = {
        "matching_method_by_rank": match_methods,
        "matched_evidence_ids_by_rank": [sorted(ids) for ids in ranked_evidence],
        "matched_source_atom_ids_by_rank": [
            sorted(
                {
                    evidence_by_id[evidence_id].source_atom_id
                    for evidence_id in ids
                    if evidence_id in evidence_by_id
                    and evidence_by_id[evidence_id].source_atom_id is not None
                }
            )
            for ids in ranked_evidence
        ],
        "hard_negative_intrusions": intruding,
        "gold_evidence_set_count": len(gold_sets),
        "source_atom_ground_truth": has_source_atoms,
        "min_source_atom_coverage": min_span_coverage,
    }
    return metrics


def evaluate_dataset(
    qa_records: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]],
    *,
    qrels: Sequence[Mapping[str, Any]] = (),
    cutoffs: Sequence[int] = (1, 5, 10, 20),
    min_span_coverage: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a complete run and return per-query plus macro metrics."""

    runs_by_qid: dict[str, Sequence[Mapping[str, Any]]] = {}
    for run in run_records:
        qid = str(run.get("qid", ""))
        if not qid:
            raise ValueError("every run record must have qid")
        if qid in runs_by_qid:
            raise ValueError(f"duplicate run record for qid={qid!r}")
        items = run.get("items") or ()
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError(f"run items must be a sequence for qid={qid!r}")
        runs_by_qid[qid] = items

    query_metrics = []
    for qa in qa_records:
        qid = str(qa.get("qid", ""))
        if not qid:
            raise ValueError("every QA record must have qid")
        query_metrics.append(
            evaluate_query(
                qa,
                runs_by_qid.get(qid, ()),
                qrels=qrels,
                cutoffs=cutoffs,
                min_span_coverage=min_span_coverage,
            )
        )
    return {
        "aggregate": aggregate_query_metrics(query_metrics),
        "queries": query_metrics,
        "query_count": len(query_metrics),
        "missing_run_qids": [
            str(qa["qid"])
            for qa in qa_records
            if str(qa["qid"]) not in runs_by_qid
        ],
    }


def aggregate_query_metrics(query_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average scalar metrics and expose the denominator for each one."""

    values: dict[str, list[float]] = {}
    for record in query_metrics:
        for key, value in record.items():
            if key == "qid" or isinstance(value, (Mapping, Sequence)) and not isinstance(
                value, (str, bytes)
            ):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    values.setdefault(key, []).append(float(value))
    return {
        key: {"mean": fmean(metric_values), "count": len(metric_values)}
        for key, metric_values in sorted(values.items())
    }


def _categorical_recall_at_k(
    gold_sets: Sequence[Iterable[str]],
    ranked_values: Sequence[str | None],
    k: int,
) -> float | None:
    _validate_k(k)
    normalised = _normalise_sets(gold_sets)
    if not normalised:
        return None
    retrieved = {str(value) for value in ranked_values[:k] if value is not None}
    return max(len(gold & retrieved) / len(gold) for gold in normalised)


def _normalise_sets(values: Sequence[Iterable[str]]) -> list[frozenset[str]]:
    result = []
    for value in values:
        normalised = frozenset(str(item) for item in value)
        if normalised:
            result.append(normalised)
    return result


def _flatten_ranked(
    ranked: Sequence[Iterable[str] | str], k: int
) -> frozenset[str]:
    result: set[str] = set()
    for item in ranked[:k]:
        if isinstance(item, str):
            result.add(item)
        else:
            result.update(str(value) for value in item)
    return frozenset(result)


def _set_recall(
    gold: frozenset[str],
    retrieved: frozenset[str],
    equivalence_groups: Mapping[str, str] | None,
) -> float:
    if not equivalence_groups:
        return len(gold & retrieved) / len(gold)
    retrieved_groups = {
        equivalence_groups[item]
        for item in retrieved
        if item in equivalence_groups
    }
    satisfied = 0
    for evidence_id in gold:
        group = equivalence_groups.get(evidence_id)
        if evidence_id in retrieved or (group is not None and group in retrieved_groups):
            satisfied += 1
    return satisfied / len(gold)


def _hop_requirements(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    reasoning = record.get("reasoning") or {}
    if not isinstance(reasoning, Mapping):
        return {}
    decomposition = reasoning.get("decomposition") or ()
    result: dict[str, tuple[str, ...]] = {}
    for index, step in enumerate(decomposition, start=1):
        if not isinstance(step, Mapping):
            continue
        evidence_ids = tuple(str(item) for item in step.get("evidence_ids", ()))
        if evidence_ids:
            result[str(step.get("step", index))] = evidence_ids
    return result


def _project_gold_sets(
    gold_sets: Sequence[Iterable[str]],
    evidence_by_id: Mapping[str, EvidenceRef],
    attribute: str,
) -> list[tuple[str, ...]]:
    projected: list[tuple[str, ...]] = []
    for gold_set in gold_sets:
        values = {
            str(value)
            for evidence_id in gold_set
            if (evidence := evidence_by_id.get(str(evidence_id))) is not None
            if (value := getattr(evidence, attribute)) is not None
        }
        if values:
            projected.append(tuple(sorted(values)))
    return projected


def _expand_equivalent_gold_sets(
    gold_sets: Sequence[Iterable[str]],
    evidence_by_id: Mapping[str, EvidenceRef],
) -> list[tuple[str, ...]]:
    """Expand equivalent evidence choices before source/chunk projection."""

    by_group: dict[str, list[str]] = {}
    for item in evidence_by_id.values():
        if item.equivalence_group:
            by_group.setdefault(item.equivalence_group, []).append(item.evidence_id)
    expanded: set[tuple[str, ...]] = set()
    for gold_set in gold_sets:
        choices = []
        for evidence_id in gold_set:
            item = evidence_by_id.get(str(evidence_id))
            alternatives = (
                by_group.get(item.equivalence_group, [item.evidence_id])
                if item is not None and item.equivalence_group
                else [str(evidence_id)]
            )
            choices.append(sorted(set(alternatives)))
        for combination in product(*choices):
            expanded.add(tuple(sorted(set(combination))))
            if len(expanded) > 4096:
                raise ValueError("equivalent gold evidence expansion exceeds 4096 routes")
    return sorted(expanded)


def _qrel_grades(qrels: Sequence[Mapping[str, Any]], qid: str) -> dict[str, float]:
    grades: dict[str, float] = {}
    for qrel in qrels:
        qrel_qid = qrel.get("qid")
        if qrel_qid is not None and str(qrel_qid) != qid:
            continue
        chunk_id = qrel.get("chunk_id")
        if chunk_id is None:
            continue
        relevance = max(0.0, float(qrel.get("relevance", 0.0)))
        key = str(chunk_id)
        grades[key] = max(grades.get(key, 0.0), relevance)
    return grades


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _ensure_unique_ranks_and_chunks(ranked: Sequence[RetrievedRef]) -> None:
    ranks = [item.rank for item in ranked]
    if len(ranks) != len(set(ranks)):
        raise ValueError("retrieval ranks must be unique within a query")
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("retrieval ranks must be contiguous and start at one")
    chunk_ids = [item.chunk_id for item in ranked]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("retrieved chunk_id values must be unique within a query")


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _is_answerable(record: Mapping[str, Any]) -> bool:
    value = record.get("answerability", True)
    if isinstance(value, str):
        return value.casefold() == "answerable"
    return bool(value)


__all__ = [
    "EvidenceMatcher",
    "EvidenceRef",
    "RetrievedRef",
    "SourceSpan",
    "aggregate_query_metrics",
    "chunk_recall_at_k",
    "complete_evidence_all_hops_at_k",
    "evaluate_dataset",
    "evaluate_query",
    "evidence_set_recall_at_k",
    "hard_negative_intrusion_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "per_hop_recall_at_k",
    "source_recall_at_k",
]
