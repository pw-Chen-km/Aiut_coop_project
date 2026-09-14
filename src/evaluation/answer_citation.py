"""Deterministic answer and citation evaluation for generated RAG predictions.

This module is intentionally separate from :mod:`evaluation.metrics`, which
scores retrieval runs.  It never calls a model and never treats an unreviewed
judge label as ground truth.  Human-reviewed claim labels are the only inputs
to claim correctness and faithfulness.
"""

from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any, Mapping, Sequence
import math
import re
import unicodedata


PREDICTION_SCHEMA_VERSION = "qursor-prediction-v1"
REVIEWED_STATUS = "reviewed"
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(value: str) -> str:
    """Apply deterministic SQuAD-style normalization with Unicode support."""

    if not isinstance(value, str):
        raise TypeError("answer text must be a string")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    text = _ARTICLE_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalized_exact_match(prediction: str, references: Sequence[str]) -> float:
    """Return one when the normalized prediction equals any reference."""

    normalized_references = _normalized_references(references)
    normalized_prediction = normalize_answer(prediction)
    return float(normalized_prediction in normalized_references)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Return the best bag-of-token F1 against the canonical answer or aliases."""

    normalized_references = _normalized_references(references)
    predicted_tokens = normalize_answer(prediction).split()
    return max(_token_f1_pair(predicted_tokens, reference.split()) for reference in normalized_references)


def validate_prediction_record(
    record: Mapping[str, Any], *, path: str = "$"
) -> None:
    """Validate the dependency-free prediction JSONL contract.

    A ``pending`` judgment may carry a provisional label for diagnostics, but
    that label is ignored by scoring.  A ``reviewed`` judgment must contain a
    boolean label and at least one reviewer identifier.
    """

    if not isinstance(record, Mapping):
        raise ValueError(f"{path}: prediction record must be an object")
    allowed = {"schema_version", "qid", "answer", "citations", "claims", "metadata"}
    required = {"qid", "answer", "citations", "claims"}
    _validate_fields(record, allowed, required, path)
    version = record.get("schema_version", PREDICTION_SCHEMA_VERSION)
    if version != PREDICTION_SCHEMA_VERSION:
        raise ValueError(
            f"{path}.schema_version: expected {PREDICTION_SCHEMA_VERSION!r}"
        )
    _require_nonempty_string(record.get("qid"), f"{path}.qid")
    if not isinstance(record.get("answer"), str):
        raise ValueError(f"{path}.answer: must be a string")
    if "metadata" in record and not isinstance(record["metadata"], Mapping):
        raise ValueError(f"{path}.metadata: must be an object")

    citations = _require_array(record.get("citations"), f"{path}.citations")
    evidence_ids: set[str] = set()
    citation_ids: set[str] = set()
    for index, citation in enumerate(citations):
        citation_path = f"{path}.citations[{index}]"
        if not isinstance(citation, Mapping):
            raise ValueError(f"{citation_path}: must be an object")
        _validate_fields(
            citation,
            {"citation_id", "evidence_id", "text", "metadata"},
            {"evidence_id"},
            citation_path,
        )
        evidence_id = _require_nonempty_string(
            citation.get("evidence_id"), f"{citation_path}.evidence_id"
        )
        if evidence_id in evidence_ids:
            raise ValueError(
                f"{citation_path}.evidence_id: duplicate citation for {evidence_id!r}"
            )
        evidence_ids.add(evidence_id)
        if "citation_id" in citation:
            citation_id = _require_nonempty_string(
                citation.get("citation_id"), f"{citation_path}.citation_id"
            )
            if citation_id in citation_ids:
                raise ValueError(
                    f"{citation_path}.citation_id: duplicate id {citation_id!r}"
                )
            citation_ids.add(citation_id)
        if "text" in citation and not isinstance(citation["text"], str):
            raise ValueError(f"{citation_path}.text: must be a string")
        if "metadata" in citation and not isinstance(citation["metadata"], Mapping):
            raise ValueError(f"{citation_path}.metadata: must be an object")

    claims = _require_array(record.get("claims"), f"{path}.claims")
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        claim_path = f"{path}.claims[{index}]"
        if not isinstance(claim, Mapping):
            raise ValueError(f"{claim_path}: must be an object")
        _validate_fields(
            claim,
            {"claim_id", "text", "correctness", "entailment", "metadata"},
            {"claim_id", "text"},
            claim_path,
        )
        claim_id = _require_nonempty_string(
            claim.get("claim_id"), f"{claim_path}.claim_id"
        )
        if claim_id in claim_ids:
            raise ValueError(f"{claim_path}.claim_id: duplicate id {claim_id!r}")
        claim_ids.add(claim_id)
        _require_nonempty_string(claim.get("text"), f"{claim_path}.text")
        if "metadata" in claim and not isinstance(claim["metadata"], Mapping):
            raise ValueError(f"{claim_path}.metadata: must be an object")
        for judgment_name in ("correctness", "entailment"):
            if judgment_name in claim:
                _validate_judgment(
                    claim[judgment_name], f"{claim_path}.{judgment_name}"
                )


def citation_scores(
    qa_record: Mapping[str, Any], predicted_evidence_ids: Sequence[str]
) -> dict[str, Any]:
    """Score prediction citations against alternative/equivalent gold routes."""

    routes, evidence_keys = _gold_citation_routes(qa_record)
    if not routes:
        return {
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
            "precision_denominator": 0,
            "recall_denominator": 0,
            "selected_gold_evidence_set_id": None,
            "matched_citation_ids": [],
            "unknown_citation_ids": [
                str(value)
                for value in predicted_evidence_ids
                if str(value) not in evidence_keys
            ],
        }

    predicted_ids = [str(value) for value in predicted_evidence_ids]
    predicted_keys = [
        evidence_keys.get(evidence_id, f"evidence:{evidence_id}")
        for evidence_id in predicted_ids
    ]
    scored_routes: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for route_id, requirements in routes:
        relevant_count = sum(key in requirements for key in predicted_keys)
        matched_requirements = set(predicted_keys) & requirements
        precision = relevant_count / len(predicted_keys) if predicted_keys else 0.0
        recall = len(matched_requirements) / len(requirements)
        f1 = _harmonic_mean(precision, recall)
        scored_routes.append(
            (
                (f1, recall, precision),
                {
                    "citation_precision": precision,
                    "citation_recall": recall,
                    "citation_f1": f1,
                    "precision_denominator": len(predicted_keys),
                    "recall_denominator": len(requirements),
                    "selected_gold_evidence_set_id": route_id,
                    "matched_citation_ids": [
                        evidence_id
                        for evidence_id, key in zip(predicted_ids, predicted_keys)
                        if key in requirements
                    ],
                    "unknown_citation_ids": [
                        evidence_id
                        for evidence_id in predicted_ids
                        if evidence_id not in evidence_keys
                    ],
                },
            )
        )
    return max(scored_routes, key=lambda item: item[0])[1]


def evaluate_answer_citation_query(
    qa_record: Mapping[str, Any],
    prediction_record: Mapping[str, Any] | None,
    *,
    retrieval_metrics: Mapping[str, Any] | None = None,
    joint_cutoff: int = 10,
) -> dict[str, Any]:
    """Evaluate one generated answer, its citations, and reviewed claim labels."""

    if not isinstance(joint_cutoff, int) or isinstance(joint_cutoff, bool) or joint_cutoff < 1:
        raise ValueError("joint_cutoff must be a positive integer")
    qid = _require_nonempty_string(qa_record.get("qid"), "$.qa.qid")
    missing_prediction = prediction_record is None
    if prediction_record is None:
        prediction: Mapping[str, Any] = {
            "qid": qid,
            "answer": "",
            "citations": [],
            "claims": [],
        }
    else:
        validate_prediction_record(prediction_record)
        prediction = prediction_record
        if str(prediction.get("qid")) != qid:
            raise ValueError(
                f"prediction qid {prediction.get('qid')!r} does not match QA qid {qid!r}"
            )

    references = _gold_answers(qa_record)
    predicted_answer = str(prediction["answer"])
    exact_match = normalized_exact_match(predicted_answer, references)
    answer_f1 = token_f1(predicted_answer, references)

    claims = prediction.get("claims") or ()
    correctness_values = _reviewed_values(claims, "correctness")
    entailment_values = _reviewed_values(claims, "entailment")
    claim_correctness = _mean_boolean(correctness_values)
    faithfulness = _mean_boolean(entailment_values)

    citation_ids = [
        str(citation["evidence_id"])
        for citation in prediction.get("citations") or ()
    ]
    citation_result = citation_scores(qa_record, citation_ids)

    complete_metric = f"complete_evidence_all_hops@{joint_cutoff}"
    complete_evidence = _complete_evidence_value(
        retrieval_metrics.get(complete_metric)
        if isinstance(retrieval_metrics, Mapping)
        else None,
        complete_metric,
    )
    joint_key = f"joint_success@{joint_cutoff}"
    joint_success = (
        int(exact_match == 1.0 and complete_evidence == 1)
        if complete_evidence is not None
        else None
    )

    correctness_diagnostics = _judgment_diagnostics(claims, "correctness")
    entailment_diagnostics = _judgment_diagnostics(claims, "entailment")
    result = {
        "qid": qid,
        "normalized_em": exact_match,
        "token_f1": answer_f1,
        "claim_correctness": claim_correctness,
        "citation_precision": citation_result["citation_precision"],
        "citation_recall": citation_result["citation_recall"],
        "citation_f1": citation_result["citation_f1"],
        "faithfulness": faithfulness,
        "complete_evidence_retrieved": complete_evidence,
        joint_key: joint_success,
        "denominators": {
            "normalized_em": 1,
            "token_f1": 1,
            "claim_correctness": len(correctness_values),
            "citation_precision": citation_result["precision_denominator"],
            "citation_recall": citation_result["recall_denominator"],
            "citation_f1": int(citation_result["citation_f1"] is not None),
            "faithfulness": len(entailment_values),
            joint_key: int(complete_evidence is not None),
        },
        "diagnostics": {
            "missing_prediction": missing_prediction,
            "answer_correct": exact_match == 1.0,
            "gold_answer_count": len(references),
            "selected_gold_evidence_set_id": citation_result[
                "selected_gold_evidence_set_id"
            ],
            "matched_citation_ids": citation_result["matched_citation_ids"],
            "unknown_citation_ids": citation_result["unknown_citation_ids"],
            "correctness_labels": correctness_diagnostics,
            "entailment_labels": entailment_diagnostics,
        },
    }
    return result


def evaluate_answer_citation_dataset(
    qa_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    retrieval_report: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    joint_cutoff: int = 10,
) -> dict[str, Any]:
    """Evaluate a prediction JSONL collection and report explicit denominators."""

    if not isinstance(joint_cutoff, int) or isinstance(joint_cutoff, bool) or joint_cutoff < 1:
        raise ValueError("joint_cutoff must be a positive integer")
    qa_by_qid: dict[str, Mapping[str, Any]] = {}
    for index, qa in enumerate(qa_records):
        qid = _require_nonempty_string(qa.get("qid"), f"$.qa[{index}].qid")
        if qid in qa_by_qid:
            raise ValueError(f"duplicate QA record for qid={qid!r}")
        qa_by_qid[qid] = qa

    predictions_by_qid: dict[str, Mapping[str, Any]] = {}
    for index, prediction in enumerate(prediction_records):
        validate_prediction_record(prediction, path=f"$.predictions[{index}]")
        qid = str(prediction["qid"])
        if qid in predictions_by_qid:
            raise ValueError(f"duplicate prediction record for qid={qid!r}")
        predictions_by_qid[qid] = prediction

    retrieval_by_qid = _retrieval_metrics_by_qid(retrieval_report)
    query_results = [
        evaluate_answer_citation_query(
            qa,
            predictions_by_qid.get(qid),
            retrieval_metrics=retrieval_by_qid.get(qid),
            joint_cutoff=joint_cutoff,
        )
        for qid, qa in qa_by_qid.items()
    ]
    joint_key = f"joint_success@{joint_cutoff}"
    review_totals = {
        "correctness_reviewed": sum(
            query["denominators"]["claim_correctness"] for query in query_results
        ),
        "correctness_unreviewed_labels_ignored": sum(
            query["diagnostics"]["correctness_labels"]["unreviewed_labels_ignored"]
            for query in query_results
        ),
        "entailment_reviewed": sum(
            query["denominators"]["faithfulness"] for query in query_results
        ),
        "entailment_unreviewed_labels_ignored": sum(
            query["diagnostics"]["entailment_labels"]["unreviewed_labels_ignored"]
            for query in query_results
        ),
    }
    return {
        "schema_version": "answer-citation-evaluation-v1",
        "joint_cutoff": joint_cutoff,
        "aggregate": aggregate_answer_citation_metrics(query_results, joint_key),
        "queries": query_results,
        "query_count": len(query_results),
        "prediction_count": len(prediction_records),
        "missing_prediction_qids": [
            qid for qid in qa_by_qid if qid not in predictions_by_qid
        ],
        "unexpected_prediction_qids": sorted(
            set(predictions_by_qid) - set(qa_by_qid)
        ),
        "missing_retrieval_metric_qids": [
            query["qid"]
            for query in query_results
            if query["complete_evidence_retrieved"] is None
        ],
        "review_diagnostics": review_totals,
    }


def aggregate_answer_citation_metrics(
    query_metrics: Sequence[Mapping[str, Any]], joint_key: str = "joint_success@10"
) -> dict[str, Any]:
    """Aggregate query scores; reviewed claim metrics use label denominators."""

    metric_names = (
        "normalized_em",
        "token_f1",
        "claim_correctness",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "faithfulness",
        joint_key,
    )
    result: dict[str, Any] = {}
    for metric in metric_names:
        if metric in {"claim_correctness", "faithfulness"}:
            weighted_numerator = 0.0
            denominator = 0
            query_count = 0
            for query in query_metrics:
                value = query.get(metric)
                count = int((query.get("denominators") or {}).get(metric, 0))
                if value is None or count <= 0:
                    continue
                weighted_numerator += float(value) * count
                denominator += count
                query_count += 1
            result[metric] = {
                "mean": weighted_numerator / denominator if denominator else None,
                "denominator": denominator,
                "denominator_unit": "reviewed_claim_labels",
                "query_count": query_count,
            }
            continue
        values = [
            float(value)
            for query in query_metrics
            if (value := query.get(metric)) is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        result[metric] = {
            "mean": fmean(values) if values else None,
            "denominator": len(values),
            "denominator_unit": "queries",
        }
    return result


def _normalized_references(references: Sequence[str]) -> tuple[str, ...]:
    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise TypeError("references must be a sequence of strings")
    normalized = tuple(dict.fromkeys(normalize_answer(value) for value in references))
    if not normalized:
        raise ValueError("at least one reference answer is required")
    return normalized


def _token_f1_pair(predicted: Sequence[str], reference: Sequence[str]) -> float:
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return _harmonic_mean(precision, recall)


def _harmonic_mean(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _gold_answers(qa_record: Mapping[str, Any]) -> tuple[str, ...]:
    answer = qa_record.get("answer")
    if isinstance(answer, str):
        return (answer,)
    if not isinstance(answer, Mapping):
        raise ValueError("QA answer must be a string or an object")
    canonical = answer.get("canonical")
    if not isinstance(canonical, str):
        raise ValueError("QA answer.canonical must be a string")
    aliases = answer.get("aliases") or ()
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
        raise ValueError("QA answer.aliases must be an array of strings")
    if any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("QA answer.aliases must contain only strings")
    return tuple(dict.fromkeys((canonical, *(str(alias) for alias in aliases))))


def _gold_citation_routes(
    qa_record: Mapping[str, Any],
) -> tuple[list[tuple[str, frozenset[str]]], dict[str, str]]:
    evidence = qa_record.get("evidence") or ()
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ValueError("QA evidence must be an array")
    evidence_keys: dict[str, str] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping) or item.get("evidence_id") is None:
            raise ValueError(f"QA evidence[{index}] must contain evidence_id")
        evidence_id = str(item["evidence_id"])
        if evidence_id in evidence_keys:
            raise ValueError(f"duplicate QA evidence_id {evidence_id!r}")
        group = item.get("equivalence_group")
        evidence_keys[evidence_id] = (
            f"group:{group}" if group is not None else f"evidence:{evidence_id}"
        )

    raw_sets = qa_record.get("gold_evidence_sets") or ()
    routes: list[tuple[str, frozenset[str]]] = []
    if isinstance(raw_sets, (str, bytes)) or not isinstance(raw_sets, Sequence):
        raise ValueError("QA gold_evidence_sets must be an array")
    for index, raw_set in enumerate(raw_sets, start=1):
        if not isinstance(raw_set, Mapping):
            continue
        requirements: set[str] = set()
        for evidence_id in raw_set.get("required_evidence_ids") or ():
            key = evidence_keys.get(str(evidence_id), f"evidence:{evidence_id}")
            requirements.add(key)
        for group in raw_set.get("required_equivalence_groups") or ():
            requirements.add(f"group:{group}")
        if requirements:
            routes.append(
                (
                    str(raw_set.get("set_id", f"route-{index}")),
                    frozenset(requirements),
                )
            )
    if not routes and evidence_keys and _is_answerable(qa_record):
        routes.append(("implicit-all-evidence", frozenset(evidence_keys.values())))
    return routes, evidence_keys


def _is_answerable(record: Mapping[str, Any]) -> bool:
    value = record.get("answerability", True)
    if isinstance(value, str):
        return value.casefold() == "answerable"
    return bool(value)


def _reviewed_values(
    claims: Sequence[Mapping[str, Any]], judgment_name: str
) -> list[bool]:
    values: list[bool] = []
    for claim in claims:
        judgment = claim.get(judgment_name)
        if (
            isinstance(judgment, Mapping)
            and judgment.get("review_status") == REVIEWED_STATUS
        ):
            values.append(bool(judgment["label"]))
    return values


def _mean_boolean(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _judgment_diagnostics(
    claims: Sequence[Mapping[str, Any]], judgment_name: str
) -> dict[str, int]:
    reviewed = 0
    ignored = 0
    missing = 0
    for claim in claims:
        judgment = claim.get(judgment_name)
        if not isinstance(judgment, Mapping):
            missing += 1
        elif judgment.get("review_status") == REVIEWED_STATUS:
            reviewed += 1
        elif "label" in judgment:
            ignored += 1
        else:
            missing += 1
    return {
        "reviewed": reviewed,
        "unreviewed_labels_ignored": ignored,
        "missing": missing,
    }


def _complete_evidence_value(value: Any, metric_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            return int(numeric)
    raise ValueError(f"{metric_name} must be 0, 1, or null")


def _retrieval_metrics_by_qid(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if report is None:
        return {}
    if isinstance(report, Mapping):
        queries = report.get("queries")
        if queries is None:
            raise ValueError("retrieval report must contain a queries array")
    else:
        queries = report
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise ValueError("retrieval report queries must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise ValueError(f"retrieval report queries[{index}] must be an object")
        qid = _require_nonempty_string(
            query.get("qid"), f"$.retrieval.queries[{index}].qid"
        )
        if qid in result:
            raise ValueError(f"duplicate retrieval metrics for qid={qid!r}")
        result[qid] = query
    return result


def _validate_judgment(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: must be an object")
    _validate_fields(
        value,
        {"label", "review_status", "reviewers", "reviewed_at", "notes"},
        {"review_status"},
        path,
    )
    status = value.get("review_status")
    if status not in {"pending", REVIEWED_STATUS}:
        raise ValueError(f"{path}.review_status: must be 'pending' or 'reviewed'")
    if "label" in value and not isinstance(value["label"], bool):
        raise ValueError(f"{path}.label: must be boolean")
    reviewers = value.get("reviewers", [])
    reviewer_values = _require_array(reviewers, f"{path}.reviewers")
    if any(not isinstance(reviewer, str) or not reviewer.strip() for reviewer in reviewer_values):
        raise ValueError(f"{path}.reviewers: must contain non-empty strings")
    if len(set(reviewer_values)) != len(reviewer_values):
        raise ValueError(f"{path}.reviewers: values must be unique")
    if status == REVIEWED_STATUS:
        if "label" not in value:
            raise ValueError(f"{path}.label: required when review_status='reviewed'")
        if not reviewer_values:
            raise ValueError(f"{path}.reviewers: required when review_status='reviewed'")
    for optional_string in ("reviewed_at", "notes"):
        if optional_string in value and not isinstance(value[optional_string], str):
            raise ValueError(f"{path}.{optional_string}: must be a string")


def _validate_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"{path}: unsupported fields: {', '.join(extra)}")


def _require_array(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path}: must be an array")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: must be a non-empty string")
    return value


__all__ = [
    "PREDICTION_SCHEMA_VERSION",
    "aggregate_answer_citation_metrics",
    "citation_scores",
    "evaluate_answer_citation_dataset",
    "evaluate_answer_citation_query",
    "normalize_answer",
    "normalized_exact_match",
    "token_f1",
    "validate_prediction_record",
]
