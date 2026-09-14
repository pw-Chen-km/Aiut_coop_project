"""Dependency-free record and cross-record validation for QEGS datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any

from .atoms import atom_union_coverage
from .evidence import resolve_evidence_span
from .ids import (
    BLOCK_ID_RE,
    CHUNK_ID_RE,
    DOC_ID_RE,
    EQUIVALENCE_GROUP_RE,
    EVIDENCE_CANDIDATE_ID_RE,
    EVIDENCE_ID_RE,
    HEX64_RE,
    QID_RE,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.code}] {self.message}"


class SchemaValidationError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


@dataclass(frozen=True, slots=True)
class QuotaBand:
    min_count: int | None = None
    max_count: int | None = None
    min_ratio: float | None = None
    max_ratio: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuotaBand":
        return cls(
            min_count=value.get("min_count"),
            max_count=value.get("max_count"),
            min_ratio=value.get("min_ratio"),
            max_ratio=value.get("max_ratio"),
        )


@dataclass(frozen=True, slots=True)
class DatasetQuota:
    min_total: int | None = None
    max_total: int | None = None
    dimensions: Mapping[str, Mapping[str, QuotaBand]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetQuota":
        total = value.get("total", {})
        dimensions: dict[str, dict[str, QuotaBand]] = {}
        for dimension, bands in value.get("dimensions", {}).items():
            dimensions[str(dimension)] = {
                str(label): QuotaBand.from_mapping(band)
                for label, band in bands.items()
            }
        return cls(
            min_total=total.get("min"),
            max_total=total.get("max"),
            dimensions=dimensions,
        )


def phase1_pilot_quota() -> DatasetQuota:
    return DatasetQuota(
        min_total=200,
        max_total=200,
        dimensions={
            "question_type": {
                "simple": QuotaBand(min_count=90, max_count=90),
                "complex": QuotaBand(min_count=60, max_count=60),
                "multihop": QuotaBand(min_count=50, max_count=50),
            },
            "scope": {
                "kanban_core": QuotaBand(min_count=140, max_count=140),
                "cross_document_diagnostic": QuotaBand(min_count=60, max_count=60),
            },
            "visual_diagnostic": {
                "true": QuotaBand(min_count=20, max_count=20),
            },
            "split": {
                "dev": QuotaBand(min_count=40, max_count=40),
                "test": QuotaBand(min_count=160, max_count=160),
            },
            "reasoning.hop_count": {
                "2": QuotaBand(min_count=35, max_count=35),
                "3": QuotaBand(min_count=15, max_count=15),
            },
        },
    )


phase1_quota = phase1_pilot_quota


def _issue(issues: list[ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(ValidationIssue(code, path, message))


def _required(
    record: Mapping[str, Any], fields: Sequence[str], path: str, issues: list[ValidationIssue]
) -> None:
    for name in fields:
        if name not in record:
            _issue(issues, "required", f"{path}.{name}", "field is required")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_document(record: Mapping[str, Any], path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = [
        "dataset_version",
        "doc_id",
        "doc_family_id",
        "title",
        "product",
        "doc_type",
        "language",
        "document_version",
        "source_filename",
        "source_sha256",
        "page_count",
        "extraction_method",
        "access_scope",
        "product_surfaces",
        "rights",
        "canonical_lineage",
        "duplicate_lineage",
        "source_provider",
        "permissions",
    ]
    _required(record, required, path, issues)
    for name in required:
        if name in {
            "page_count",
            "product_surfaces",
            "rights",
            "duplicate_lineage",
            "permissions",
        }:
            continue
        if name in record and not _nonempty_string(record[name]):
            _issue(issues, "type", f"{path}.{name}", "must be a non-empty string")
    doc_id = record.get("doc_id")
    if isinstance(doc_id, str) and not DOC_ID_RE.fullmatch(doc_id):
        _issue(issues, "id", f"{path}.doc_id", "invalid document ID")
    source_hash = record.get("source_sha256")
    if isinstance(source_hash, str) and not HEX64_RE.fullmatch(source_hash):
        _issue(issues, "hash", f"{path}.source_sha256", "must be lowercase SHA-256")
    if (
        isinstance(doc_id, str)
        and DOC_ID_RE.fullmatch(doc_id)
        and isinstance(source_hash, str)
        and HEX64_RE.fullmatch(source_hash)
        and not doc_id.endswith(source_hash[:12])
    ):
        _issue(issues, "id_hash", f"{path}.doc_id", "digest suffix must match source_sha256")
    page_count = record.get("page_count")
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
        _issue(issues, "range", f"{path}.page_count", "must be a positive integer")
    surfaces = record.get("product_surfaces")
    if not isinstance(surfaces, list) or not surfaces or not all(_nonempty_string(item) for item in surfaces):
        _issue(issues, "type", f"{path}.product_surfaces", "must be a non-empty array of strings")
    lineage = record.get("duplicate_lineage")
    if not isinstance(lineage, list) or not all(isinstance(item, str) for item in lineage):
        _issue(issues, "type", f"{path}.duplicate_lineage", "must be an array of document IDs")
    rights = record.get("rights")
    if not isinstance(rights, Mapping):
        _issue(issues, "type", f"{path}.rights", "must be an object")
    else:
        _required(rights, ["classification", "owner", "public_upload_allowed"], f"{path}.rights", issues)
        if rights.get("classification") not in {"public", "internal", "confidential", "restricted"}:
            _issue(issues, "enum", f"{path}.rights.classification", "invalid classification")
        if not _nonempty_string(rights.get("owner")):
            _issue(issues, "type", f"{path}.rights.owner", "must be a non-empty string")
        if not isinstance(rights.get("public_upload_allowed"), bool):
            _issue(issues, "type", f"{path}.rights.public_upload_allowed", "must be boolean")
    permissions = record.get("permissions")
    if not isinstance(permissions, Mapping):
        _issue(issues, "type", f"{path}.permissions", "must be an object")
    else:
        _required(
            permissions,
            ["internal_processing_allowed", "public_upload_allowed", "allowed_roles"],
            f"{path}.permissions",
            issues,
        )
        if not isinstance(permissions.get("internal_processing_allowed"), bool):
            _issue(issues, "type", f"{path}.permissions.internal_processing_allowed", "must be boolean")
        if not isinstance(permissions.get("public_upload_allowed"), bool):
            _issue(issues, "type", f"{path}.permissions.public_upload_allowed", "must be boolean")
        if not isinstance(permissions.get("allowed_roles"), list) or not all(
            isinstance(item, str) for item in permissions.get("allowed_roles", [])
        ):
            _issue(issues, "type", f"{path}.permissions.allowed_roles", "must be an array of strings")
        if (
            isinstance(rights, Mapping)
            and isinstance(rights.get("public_upload_allowed"), bool)
            and isinstance(permissions.get("public_upload_allowed"), bool)
            and rights.get("public_upload_allowed") != permissions.get("public_upload_allowed")
        ):
            _issue(issues, "rights", f"{path}.permissions.public_upload_allowed", "must match rights policy")
    return issues


def _validate_chunk(record: Mapping[str, Any], path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = [
        "chunk_id",
        "chunk_schema_version",
        "doc_id",
        "text",
        "token_count",
        "section_path",
        "page_start",
        "page_end",
        "content_type",
        "source_spans",
    ]
    _required(record, required, path, issues)
    chunk_id = record.get("chunk_id")
    doc_id = record.get("doc_id")
    match = CHUNK_ID_RE.fullmatch(chunk_id) if isinstance(chunk_id, str) else None
    if match is None:
        _issue(issues, "id", f"{path}.chunk_id", "invalid chunk ID")
    elif match.group("doc") != doc_id:
        _issue(issues, "reference", f"{path}.chunk_id", "chunk ID doc prefix differs from doc_id")
    if isinstance(doc_id, str) and not DOC_ID_RE.fullmatch(doc_id):
        _issue(issues, "id", f"{path}.doc_id", "invalid document ID")
    content_type = record.get("content_type")
    if content_type not in {"prose", "table", "caption", "mixed", "visual"}:
        _issue(issues, "enum", f"{path}.content_type", "invalid content type")
    text = record.get("text")
    if not isinstance(text, str) or (content_type != "visual" and not text.strip()):
        _issue(issues, "type", f"{path}.text", "must be text; non-visual chunks cannot be empty")
    token_count = record.get("token_count")
    minimum_tokens = 0 if content_type == "visual" else 1
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < minimum_tokens:
        _issue(issues, "range", f"{path}.token_count", f"must be an integer >= {minimum_tokens}")
    if not isinstance(record.get("section_path"), list) or not all(
        isinstance(item, str) for item in record.get("section_path", [])
    ):
        _issue(issues, "type", f"{path}.section_path", "must be an array of strings")
    page_start, page_end = record.get("page_start"), record.get("page_end")
    if not isinstance(page_start, int) or page_start < 0:
        _issue(issues, "range", f"{path}.page_start", "must be a non-negative integer")
    if not isinstance(page_end, int) or page_end < 0:
        _issue(issues, "range", f"{path}.page_end", "must be a non-negative integer")
    if isinstance(page_start, int) and isinstance(page_end, int) and page_start > page_end:
        _issue(issues, "range", f"{path}.page_end", "must be >= page_start")
    spans = record.get("source_spans")
    if not isinstance(spans, list) or (content_type != "visual" and not spans):
        _issue(issues, "type", f"{path}.source_spans", "must be an array; text chunks require spans")
        return issues
    text_length = len(record.get("text", "")) if isinstance(record.get("text"), str) else 0
    for index, span in enumerate(spans):
        span_path = f"{path}.source_spans[{index}]"
        if not isinstance(span, Mapping):
            _issue(issues, "type", span_path, "must be an object")
            continue
        _required(
            span,
            [
                "block_id",
                "page_index",
                "page_label",
                "source_start_char",
                "source_end_char",
                "chunk_start_char",
                "chunk_end_char",
            ],
            span_path,
            issues,
        )
        block_match = BLOCK_ID_RE.fullmatch(span.get("block_id", ""))
        if block_match is None:
            _issue(issues, "id", f"{span_path}.block_id", "invalid block ID")
        elif block_match.group("doc") != doc_id:
            _issue(issues, "reference", f"{span_path}.block_id", "block belongs to another document")
        bounds = [
            span.get("source_start_char"),
            span.get("source_end_char"),
            span.get("chunk_start_char"),
            span.get("chunk_end_char"),
        ]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in bounds):
            _issue(issues, "type", span_path, "span bounds must be integers")
            continue
        ss, se, cs, ce = bounds
        if ss < 0 or se <= ss or cs < 0 or ce <= cs or ce > text_length:
            _issue(issues, "range", span_path, "invalid source or chunk span bounds")
        if se - ss != ce - cs:
            _issue(issues, "alignment", span_path, "source and chunk spans must have equal length")
    if "safety_sensitive" in record and not isinstance(record.get("safety_sensitive"), bool):
        _issue(issues, "type", f"{path}.safety_sensitive", "must be boolean")
    if "visual_only" in record and not isinstance(record.get("visual_only"), bool):
        _issue(issues, "type", f"{path}.visual_only", "must be boolean")
    if content_type == "visual":
        for visual_field in ("visual_ref", "figure_id", "bbox"):
            if visual_field not in record:
                _issue(issues, "required", f"{path}.{visual_field}", "visual chunk requires this field")
        visual_ref = record.get("visual_ref")
        if not _nonempty_string(visual_ref) or str(visual_ref).startswith(("/", "..")):
            _issue(issues, "path", f"{path}.visual_ref", "must be a local relative path")
        bbox = record.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in (bbox or [])
        ):
            _issue(issues, "type", f"{path}.bbox", "must contain four numbers")
        if not _nonempty_string(record.get("figure_id")):
            _issue(issues, "type", f"{path}.figure_id", "must be a non-empty string")
        if "ocr_text" in record and not isinstance(record.get("ocr_text"), str):
            _issue(issues, "type", f"{path}.ocr_text", "must be a string")
    group = record.get("equivalence_group_id")
    if group is not None and (not isinstance(group, str) or not EQUIVALENCE_GROUP_RE.fullmatch(group)):
        _issue(issues, "id", f"{path}.equivalence_group_id", "invalid equivalence group")
    duplicate_lineage = record.get("duplicate_lineage")
    if duplicate_lineage is not None and (
        not isinstance(duplicate_lineage, list)
        or not all(isinstance(item, str) for item in duplicate_lineage)
    ):
        _issue(issues, "type", f"{path}.duplicate_lineage", "must be an array of strings")
    canonical_chunk_id = record.get("canonical_chunk_id")
    if canonical_chunk_id is not None and (
        not isinstance(canonical_chunk_id, str) or not CHUNK_ID_RE.fullmatch(canonical_chunk_id)
    ):
        _issue(issues, "id", f"{path}.canonical_chunk_id", "invalid canonical chunk ID")
    return issues


def _validate_evidence_item(
    evidence: Mapping[str, Any], path: str, qid: str | None, *, candidate: bool = False
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    id_field = "evidence_candidate_id" if candidate else "evidence_id"
    required = [
        id_field,
        "doc_id",
        "chunk_id",
        "block_id",
        "page_index",
        "page_label",
        "section_path",
        "role",
        "equivalence_group",
        "citation",
    ]
    _required(evidence, required, path, issues)
    evidence_id = evidence.get(id_field)
    if candidate:
        if not isinstance(evidence_id, str) or not EVIDENCE_CANDIDATE_ID_RE.fullmatch(evidence_id):
            _issue(issues, "id", f"{path}.{id_field}", "invalid evidence candidate ID")
    else:
        match = EVIDENCE_ID_RE.fullmatch(evidence_id) if isinstance(evidence_id, str) else None
        if match is None:
            _issue(issues, "id", f"{path}.{id_field}", "invalid evidence ID")
        elif qid is not None and match.group("qid") != qid:
            _issue(issues, "reference", f"{path}.{id_field}", "evidence ID must be namespaced by qid")
        if qid is not None and qid.startswith("qursor:p2:"):
            atom_id = evidence.get("source_atom_id")
            if not isinstance(atom_id, str) or not re.fullmatch(r"sa:[0-9a-f]{24}", atom_id):
                _issue(issues, "id", f"{path}.source_atom_id", "phase-2 evidence requires a stable source atom ID")
    if not isinstance(evidence.get("doc_id"), str) or not DOC_ID_RE.fullmatch(evidence["doc_id"]):
        _issue(issues, "id", f"{path}.doc_id", "invalid document ID")
    chunk_match = CHUNK_ID_RE.fullmatch(evidence.get("chunk_id", ""))
    if chunk_match is None:
        _issue(issues, "id", f"{path}.chunk_id", "invalid chunk ID")
    elif chunk_match.group("doc") != evidence.get("doc_id"):
        _issue(issues, "reference", f"{path}.chunk_id", "chunk belongs to another document")
    block_match = BLOCK_ID_RE.fullmatch(evidence.get("block_id", ""))
    if block_match is None:
        _issue(issues, "id", f"{path}.block_id", "invalid block ID")
    elif block_match.group("doc") != evidence.get("doc_id"):
        _issue(issues, "reference", f"{path}.block_id", "block belongs to another document")
    if not isinstance(evidence.get("page_index"), int) or evidence.get("page_index", -1) < 0:
        _issue(issues, "range", f"{path}.page_index", "must be a non-negative integer")
    if not isinstance(evidence.get("section_path"), list):
        _issue(issues, "type", f"{path}.section_path", "must be an array")
    if not _nonempty_string(evidence.get("citation")):
        _issue(issues, "type", f"{path}.citation", "must be a non-empty string")
    allowed_roles = {"candidate", "visual"} if candidate else {
        "answer", "bridge", "constraint", "alternate", "visual"
    }
    if evidence.get("role") not in allowed_roles:
        _issue(issues, "enum", f"{path}.role", f"must be one of {sorted(allowed_roles)}")
    group = evidence.get("equivalence_group")
    if not isinstance(group, str) or not EQUIVALENCE_GROUP_RE.fullmatch(group):
        _issue(issues, "id", f"{path}.equivalence_group", "invalid equivalence group")
    visual_only = evidence.get("visual_only") is True
    if visual_only:
        for name in ("visual_ref", "figure_id", "bbox"):
            if name not in evidence:
                _issue(issues, "required", f"{path}.{name}", "visual evidence requires this field")
        bbox = evidence.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in (bbox or [])
        ):
            _issue(issues, "type", f"{path}.bbox", "must contain four numbers")
    else:
        if not _nonempty_string(evidence.get("quote")):
            _issue(issues, "type", f"{path}.quote", "text evidence needs a non-empty quote")
        span = evidence.get("span")
        if not isinstance(span, Mapping):
            _issue(issues, "type", f"{path}.span", "text evidence needs a span object")
        else:
            start, end = span.get("start_char"), span.get("end_char")
            if span.get("coordinate_space", "chunk_text") != "chunk_text":
                _issue(issues, "value", f"{path}.span.coordinate_space", "must be chunk_text")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                _issue(issues, "range", f"{path}.span", "invalid span bounds")
        source_span = evidence.get("source_span")
        if not isinstance(source_span, Mapping):
            _issue(issues, "type", f"{path}.source_span", "text evidence needs a block_text span")
        else:
            source_start = source_span.get("start_char")
            source_end = source_span.get("end_char")
            if source_span.get("coordinate_space", "block_text") != "block_text":
                _issue(issues, "value", f"{path}.source_span.coordinate_space", "must be block_text")
            if (
                not isinstance(source_start, int)
                or not isinstance(source_end, int)
                or source_start < 0
                or source_end <= source_start
            ):
                _issue(issues, "range", f"{path}.source_span", "invalid source span bounds")
    if "safety_sensitive" in evidence and not isinstance(evidence.get("safety_sensitive"), bool):
        _issue(issues, "type", f"{path}.safety_sensitive", "must be boolean")
    return issues


def _validate_qa(record: Mapping[str, Any], path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = [
        "qid",
        "split",
        "question_family_id",
        "evidence_cluster_id",
        "scope",
        "modalities",
        "visual_diagnostic",
        "safety_sensitive",
        "language",
        "question",
        "question_type",
        "subtype",
        "difficulty",
        "answerability",
        "answer",
        "claims",
        "evidence",
        "gold_evidence_sets",
        "reasoning",
        "hard_negatives",
        "provenance",
        "validation",
        "review",
    ]
    _required(record, required, path, issues)
    qid = record.get("qid")
    qid_match = QID_RE.fullmatch(qid) if isinstance(qid, str) else None
    if qid_match is None:
        _issue(issues, "id", f"{path}.qid", "invalid question ID")
    elif qid_match.group("split") != record.get("split"):
        _issue(issues, "reference", f"{path}.split", "split differs from qid")
    if qid_match is not None and qid_match.group("phase") == "p2":
        _required(record, ["source_qid", "scenario"], path, issues)
        source_qid = record.get("source_qid")
        source_match = QID_RE.fullmatch(source_qid) if isinstance(source_qid, str) else None
        if source_match is None or source_match.group("phase") != "p1":
            _issue(issues, "id", f"{path}.source_qid", "scenario candidate requires a p1 source qid")
        elif (
            source_match.group("ordinal") != qid_match.group("ordinal")
            or source_match.group("split") != qid_match.group("split")
        ):
            _issue(issues, "lineage", f"{path}.source_qid", "source qid must preserve split and ordinal")
        scenario = record.get("scenario")
        if not isinstance(scenario, Mapping):
            _issue(issues, "type", f"{path}.scenario", "must be a scenario object")
        else:
            scenario_path = f"{path}.scenario"
            _required(
                scenario,
                [
                    "scenario_id",
                    "scenario_family_id",
                    "persona",
                    "terminology_tier",
                    "user_intent",
                    "user_goal",
                    "current_state",
                    "observable_symptom",
                    "previous_action",
                    "visible_ui_terms",
                    "source_disposition",
                    "authenticity_rationale",
                    "authenticity_review_status",
                ],
                scenario_path,
                issues,
            )
            if not re.fullmatch(r"scn:[a-z0-9][a-z0-9._:-]{2,127}", str(scenario.get("scenario_id", ""))):
                _issue(issues, "id", f"{scenario_path}.scenario_id", "invalid scenario id")
            if not re.fullmatch(r"scn:[a-z0-9][a-z0-9._:-]{2,127}", str(scenario.get("scenario_family_id", ""))):
                _issue(issues, "id", f"{scenario_path}.scenario_family_id", "invalid scenario family id")
            if scenario.get("persona") not in {
                "frontline_operator", "fleet_operator", "maintenance_technician",
                "workflow_engineer", "system_integrator", "safety_support",
            }:
                _issue(issues, "enum", f"{scenario_path}.persona", "invalid persona")
            if scenario.get("terminology_tier") not in {"frontline", "technical", "expert"}:
                _issue(issues, "enum", f"{scenario_path}.terminology_tier", "invalid terminology tier")
            if scenario.get("user_intent") not in {
                "how_to", "troubleshooting", "decision", "status_check", "safety", "clarification_needed",
            }:
                _issue(issues, "enum", f"{scenario_path}.user_intent", "invalid user intent")
            if scenario.get("source_disposition") not in {"keep", "rewrite", "regenerate"}:
                _issue(issues, "enum", f"{scenario_path}.source_disposition", "invalid source disposition")
            if scenario.get("authenticity_review_status") not in {"pending", "approved", "rejected"}:
                _issue(issues, "enum", f"{scenario_path}.authenticity_review_status", "invalid authenticity status")
            if not isinstance(scenario.get("visible_ui_terms"), list) or not all(
                _nonempty_string(item) for item in scenario.get("visible_ui_terms", [])
            ):
                _issue(issues, "type", f"{scenario_path}.visible_ui_terms", "must be an array of non-empty strings")
            for field_name in ("user_goal", "authenticity_rationale"):
                if not _nonempty_string(scenario.get(field_name)):
                    _issue(issues, "type", f"{scenario_path}.{field_name}", "must be a non-empty string")
    if record.get("question_type") not in {"simple", "complex", "multihop"}:
        _issue(issues, "enum", f"{path}.question_type", "must be simple, complex, or multihop")
    if record.get("difficulty") not in {"easy", "medium", "hard"}:
        _issue(issues, "enum", f"{path}.difficulty", "must be easy, medium, or hard")
    if record.get("answerability") not in {"answerable", "unanswerable"}:
        _issue(issues, "enum", f"{path}.answerability", "invalid answerability")
    if record.get("scope") not in {"kanban_core", "cross_document_diagnostic"}:
        _issue(issues, "enum", f"{path}.scope", "invalid dataset scope")
    modalities = record.get("modalities")
    if (
        not isinstance(modalities, list)
        or not modalities
        or len(modalities) != len(set(modalities))
        or any(item not in {"text", "visual"} for item in modalities)
    ):
        _issue(issues, "enum", f"{path}.modalities", "must contain unique text/visual values")
    if not isinstance(record.get("visual_diagnostic"), bool):
        _issue(issues, "type", f"{path}.visual_diagnostic", "must be boolean")
    elif record.get("visual_diagnostic") and "visual" not in (modalities or []):
        _issue(issues, "modality", f"{path}.visual_diagnostic", "requires visual modality")
    if not isinstance(record.get("safety_sensitive"), bool):
        _issue(issues, "type", f"{path}.safety_sensitive", "must be boolean")
    for group_field in ("question_family_id", "evidence_cluster_id"):
        if not _nonempty_string(record.get(group_field)):
            _issue(issues, "type", f"{path}.{group_field}", "must be a non-empty string")
    if not _nonempty_string(record.get("question")):
        _issue(issues, "type", f"{path}.question", "must be a non-empty string")
    answer = record.get("answer")
    if not isinstance(answer, Mapping):
        _issue(issues, "type", f"{path}.answer", "must be an object")
    else:
        _required(answer, ["canonical", "aliases", "type", "unit", "tolerance"], f"{path}.answer", issues)
        if record.get("answerability") == "answerable" and not _nonempty_string(answer.get("canonical")):
            _issue(issues, "answer", f"{path}.answer.canonical", "answerable item needs an answer")
        if record.get("answerability") == "unanswerable" and answer.get("type") != "abstain":
            _issue(issues, "answer", f"{path}.answer.type", "unanswerable item must use abstain")
    evidence_records = record.get("evidence")
    evidence_ids: set[str] = set()
    evidence_groups: set[str] = set()
    if not isinstance(evidence_records, list):
        _issue(issues, "type", f"{path}.evidence", "must be an array")
        evidence_records = []
    for index, evidence in enumerate(evidence_records):
        evidence_path = f"{path}.evidence[{index}]"
        if not isinstance(evidence, Mapping):
            _issue(issues, "type", evidence_path, "must be an object")
            continue
        issues.extend(_validate_evidence_item(evidence, evidence_path, qid if isinstance(qid, str) else None))
        evidence_id = evidence.get("evidence_id")
        if isinstance(evidence_id, str):
            if evidence_id in evidence_ids:
                _issue(issues, "duplicate", f"{evidence_path}.evidence_id", "duplicate evidence ID")
            evidence_ids.add(evidence_id)
        if isinstance(evidence.get("equivalence_group"), str):
            evidence_groups.add(evidence["equivalence_group"])
    if record.get("answerability") == "answerable" and not evidence_records:
        _issue(issues, "evidence", f"{path}.evidence", "answerable item requires evidence")
    if record.get("visual_diagnostic") and not any(
        isinstance(item, Mapping)
        and (item.get("visual_only") is True or item.get("role") == "visual")
        for item in evidence_records
    ):
        _issue(issues, "modality", f"{path}.evidence", "visual diagnostic requires visual evidence")

    claims = record.get("claims")
    claim_ids: set[str] = set()
    if not isinstance(claims, list):
        _issue(issues, "type", f"{path}.claims", "must be an array")
        claims = []
    for index, claim in enumerate(claims):
        claim_path = f"{path}.claims[{index}]"
        if not isinstance(claim, Mapping):
            _issue(issues, "type", claim_path, "must be an object")
            continue
        _required(claim, ["claim_id", "text", "supported_by"], claim_path, issues)
        claim_id = claim.get("claim_id")
        if not _nonempty_string(claim_id):
            _issue(issues, "type", f"{claim_path}.claim_id", "must be non-empty")
        elif claim_id in claim_ids:
            _issue(issues, "duplicate", f"{claim_path}.claim_id", "duplicate claim ID")
        else:
            claim_ids.add(claim_id)
        supported_by = claim.get("supported_by")
        if not isinstance(supported_by, list) or not supported_by:
            _issue(issues, "evidence", f"{claim_path}.supported_by", "must reference evidence")
        else:
            for evidence_id in supported_by:
                if evidence_id not in evidence_ids:
                    _issue(issues, "reference", f"{claim_path}.supported_by", f"unknown evidence: {evidence_id}")

    gold_sets = record.get("gold_evidence_sets")
    set_ids: set[str] = set()
    if not isinstance(gold_sets, list):
        _issue(issues, "type", f"{path}.gold_evidence_sets", "must be an array")
        gold_sets = []
    for index, gold_set in enumerate(gold_sets):
        set_path = f"{path}.gold_evidence_sets[{index}]"
        if not isinstance(gold_set, Mapping):
            _issue(issues, "type", set_path, "must be an object")
            continue
        _required(gold_set, ["set_id", "required_evidence_ids"], set_path, issues)
        set_id = gold_set.get("set_id")
        if not _nonempty_string(set_id) or set_id in set_ids:
            _issue(issues, "id", f"{set_path}.set_id", "set ID must be non-empty and unique")
        elif isinstance(set_id, str):
            set_ids.add(set_id)
        required_ids = gold_set.get("required_evidence_ids")
        required_groups = gold_set.get("required_equivalence_groups", [])
        if not isinstance(required_ids, list) or not all(item in evidence_ids for item in required_ids):
            _issue(issues, "reference", f"{set_path}.required_evidence_ids", "contains unknown evidence")
        if not isinstance(required_groups, list) or not all(item in evidence_groups for item in required_groups):
            _issue(
                issues,
                "reference",
                f"{set_path}.required_equivalence_groups",
                "contains unknown equivalence group",
            )
        if not required_ids and not required_groups:
            _issue(issues, "evidence", set_path, "gold set must require evidence IDs or groups")
    if record.get("answerability") == "answerable" and not gold_sets:
        _issue(issues, "evidence", f"{path}.gold_evidence_sets", "answerable item requires a gold set")

    reasoning = record.get("reasoning")
    if not isinstance(reasoning, Mapping):
        _issue(issues, "type", f"{path}.reasoning", "must be an object")
        reasoning = {}
    _required(reasoning, ["hop_count", "operator", "decomposition", "hop_necessity"], f"{path}.reasoning", issues)
    hop_count = reasoning.get("hop_count")
    if not isinstance(hop_count, int) or isinstance(hop_count, bool) or not 0 <= hop_count <= 3:
        _issue(issues, "range", f"{path}.reasoning.hop_count", "must be an integer from 0 to 3")
    decomposition = reasoning.get("decomposition")
    if not isinstance(decomposition, list):
        _issue(issues, "type", f"{path}.reasoning.decomposition", "must be an array")
        decomposition = []
    for index, step in enumerate(decomposition):
        step_path = f"{path}.reasoning.decomposition[{index}]"
        if not isinstance(step, Mapping):
            _issue(issues, "type", step_path, "must be an object")
            continue
        _required(step, ["step", "subquestion", "evidence_ids"], step_path, issues)
        if step.get("step") != index + 1:
            _issue(issues, "order", f"{step_path}.step", "steps must be one-based and contiguous")
        step_evidence = step.get("evidence_ids")
        if not isinstance(step_evidence, list) or not step_evidence:
            _issue(issues, "evidence", f"{step_path}.evidence_ids", "must reference evidence")
        elif any(evidence_id not in evidence_ids for evidence_id in step_evidence):
            _issue(issues, "reference", f"{step_path}.evidence_ids", "contains unknown evidence")
    if record.get("question_type") == "multihop":
        if hop_count not in {2, 3}:
            _issue(issues, "hop", f"{path}.reasoning.hop_count", "multihop requires 2 or 3 hops")
        if isinstance(hop_count, int) and len(decomposition) != hop_count:
            _issue(issues, "hop", f"{path}.reasoning.decomposition", "length must equal hop_count")
        evidence_chunks = {
            item.get("chunk_id") for item in evidence_records if isinstance(item, Mapping)
        }
        evidence_blocks = {
            item.get("block_id") for item in evidence_records if isinstance(item, Mapping)
        }
        if len(evidence_chunks) < 2 or len(evidence_blocks) < 2:
            _issue(
                issues,
                "hop",
                f"{path}.evidence",
                "multihop evidence must span at least two blocks and two reference chunks",
            )
    elif isinstance(hop_count, int) and hop_count > 1:
        _issue(issues, "hop", f"{path}.reasoning.hop_count", "only multihop questions may exceed one hop")
    necessity = reasoning.get("hop_necessity")
    if not isinstance(necessity, Mapping):
        _issue(issues, "type", f"{path}.reasoning.hop_necessity", "must be an object")
    else:
        _required(necessity, ["required", "verified", "method", "tests"], f"{path}.reasoning.hop_necessity", issues)
        if record.get("question_type") == "multihop" and necessity.get("required") is not True:
            _issue(issues, "hop", f"{path}.reasoning.hop_necessity.required", "must be true for multihop")
        tests = necessity.get("tests")
        if not isinstance(tests, list):
            _issue(issues, "type", f"{path}.reasoning.hop_necessity.tests", "must be an array")
            tests = []
        for index, test in enumerate(tests):
            test_path = f"{path}.reasoning.hop_necessity.tests[{index}]"
            if not isinstance(test, Mapping):
                _issue(issues, "type", test_path, "must be an object")
                continue
            _required(test, ["hop", "removed_evidence_ids", "answerable", "notes"], test_path, issues)
            removed = test.get("removed_evidence_ids")
            if not isinstance(removed, list) or not removed or any(item not in evidence_ids for item in removed):
                _issue(issues, "reference", f"{test_path}.removed_evidence_ids", "must reference evidence")
            if necessity.get("verified") is True and test.get("answerable") is not False:
                _issue(issues, "hop", f"{test_path}.answerable", "verified ablation must make item unanswerable")
            test_hop = test.get("hop")
            if (
                isinstance(removed, list)
                and isinstance(test_hop, int)
                and 1 <= test_hop <= len(decomposition)
                and isinstance(decomposition[test_hop - 1], Mapping)
            ):
                step_evidence = set(decomposition[test_hop - 1].get("evidence_ids", []))
                if not step_evidence.intersection(removed):
                    _issue(
                        issues,
                        "hop",
                        f"{test_path}.removed_evidence_ids",
                        "ablation must remove evidence used by the declared hop",
                    )
        if (
            record.get("question_type") == "multihop"
            and necessity.get("verified") is True
            and isinstance(hop_count, int)
            and len(tests) < hop_count
        ):
            _issue(issues, "hop", f"{path}.reasoning.hop_necessity.tests", "requires at least one test per hop")
        if (
            record.get("question_type") == "multihop"
            and necessity.get("verified") is True
            and isinstance(hop_count, int)
            and {test.get("hop") for test in tests if isinstance(test, Mapping)}
            != set(range(1, hop_count + 1))
        ):
            _issue(issues, "hop", f"{path}.reasoning.hop_necessity.tests", "must cover every hop")

    hard_negatives = record.get("hard_negatives")
    if not isinstance(hard_negatives, list):
        _issue(issues, "type", f"{path}.hard_negatives", "must be an array")
    else:
        gold_chunks = {item.get("chunk_id") for item in evidence_records if isinstance(item, Mapping)}
        for index, negative in enumerate(hard_negatives):
            negative_path = f"{path}.hard_negatives[{index}]"
            if not isinstance(negative, Mapping):
                _issue(issues, "type", negative_path, "must be an object")
                continue
            _required(negative, ["chunk_id", "type"], negative_path, issues)
            if not isinstance(negative.get("chunk_id"), str) or not CHUNK_ID_RE.fullmatch(negative["chunk_id"]):
                _issue(issues, "id", f"{negative_path}.chunk_id", "invalid chunk ID")
            if negative.get("chunk_id") in gold_chunks:
                _issue(issues, "negative", f"{negative_path}.chunk_id", "gold chunk cannot be a hard negative")
    review = record.get("review")
    if not isinstance(review, Mapping):
        _issue(issues, "type", f"{path}.review", "must be an object")
    else:
        _required(review, ["status", "reviewers"], f"{path}.review", issues)
        if review.get("status") not in {"pending", "approved", "rejected"}:
            _issue(issues, "enum", f"{path}.review.status", "invalid review status")
        reviewers = review.get("reviewers")
        if not isinstance(reviewers, list) or not all(isinstance(item, str) for item in reviewers):
            _issue(issues, "type", f"{path}.review.reviewers", "must be an array of strings")
        elif review.get("status") == "approved" and not reviewers:
            _issue(issues, "review", f"{path}.review.reviewers", "approved item needs a reviewer")
    return issues


def _validate_qrel(record: Mapping[str, Any], path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _required(record, ["qid", "chunk_id", "relevance", "evidence_ids"], path, issues)
    if not isinstance(record.get("qid"), str) or not QID_RE.fullmatch(record["qid"]):
        _issue(issues, "id", f"{path}.qid", "invalid question ID")
    if not isinstance(record.get("chunk_id"), str) or not CHUNK_ID_RE.fullmatch(record["chunk_id"]):
        _issue(issues, "id", f"{path}.chunk_id", "invalid chunk ID")
    if record.get("relevance") not in {0, 1, 2}:
        _issue(issues, "enum", f"{path}.relevance", "must be 0, 1, or 2")
    if not isinstance(record.get("evidence_ids"), list):
        _issue(issues, "type", f"{path}.evidence_ids", "must be an array")
    group = record.get("equivalence_group")
    if group is not None and (not isinstance(group, str) or not EQUIVALENCE_GROUP_RE.fullmatch(group)):
        _issue(issues, "id", f"{path}.equivalence_group", "invalid equivalence group")
    return issues


def validate_record(
    kind: str,
    record: Mapping[str, Any],
    *,
    path: str = "$",
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    validators = {
        "document": _validate_document,
        "documents": _validate_document,
        "chunk": _validate_chunk,
        "chunks": _validate_chunk,
        "qa": _validate_qa,
        "qrel": _validate_qrel,
        "qrels": _validate_qrel,
        "evidence": lambda value, item_path: _validate_evidence_item(
            value, item_path, None, candidate=True
        ),
    }
    try:
        validator = validators[kind]
    except KeyError as exc:
        raise ValueError(f"unknown record kind: {kind}") from exc
    if not isinstance(record, Mapping):
        issues = [ValidationIssue("type", path, "record must be an object")]
    else:
        issues = validator(record, path)
    if issues and raise_on_error:
        raise SchemaValidationError(issues)
    return issues


def validate_quota(records: Sequence[Mapping[str, Any]], quota: DatasetQuota) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    total = len(records)
    if quota.min_total is not None and total < quota.min_total:
        _issue(issues, "quota", "$.quota.total", f"{total} is below minimum {quota.min_total}")
    if quota.max_total is not None and total > quota.max_total:
        _issue(issues, "quota", "$.quota.total", f"{total} exceeds maximum {quota.max_total}")
    for dimension, bands in quota.dimensions.items():
        def field_value(record: Mapping[str, Any]) -> Any:
            value: Any = record
            for part in dimension.split("."):
                if not isinstance(value, Mapping):
                    return None
                value = value.get(part)
            return value

        def label(value: Any) -> str:
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)

        counts = Counter(label(field_value(record)) for record in records)
        for label, band in bands.items():
            count = counts[label]
            ratio = count / total if total else 0.0
            band_path = f"$.quota.{dimension}.{label}"
            if band.min_count is not None and count < band.min_count:
                _issue(issues, "quota", band_path, f"count {count} is below {band.min_count}")
            if band.max_count is not None and count > band.max_count:
                _issue(issues, "quota", band_path, f"count {count} exceeds {band.max_count}")
            if band.min_ratio is not None and ratio < band.min_ratio:
                _issue(issues, "quota", band_path, f"ratio {ratio:.6f} is below {band.min_ratio}")
            if band.max_ratio is not None and ratio > band.max_ratio:
                _issue(issues, "quota", band_path, f"ratio {ratio:.6f} exceeds {band.max_ratio}")
    return issues


def _duplicates(records: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    values = [record.get(field) for record in records if isinstance(record.get(field), str)]
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def validate_dataset(
    *,
    documents: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    qa: Sequence[Mapping[str, Any]] = (),
    qrels: Sequence[Mapping[str, Any]] = (),
    source_atoms: Sequence[Mapping[str, Any]] = (),
    atom_chunk_map: Sequence[Mapping[str, Any]] = (),
    atom_qrels: Sequence[Mapping[str, Any]] = (),
    quota: DatasetQuota | None = None,
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for kind, records in (("document", documents), ("chunk", chunks), ("qa", qa), ("qrel", qrels)):
        for index, record in enumerate(records):
            issues.extend(validate_record(kind, record, path=f"$.{kind}[{index}]", raise_on_error=False))
    for field, records in (("doc_id", documents), ("chunk_id", chunks), ("qid", qa)):
        for duplicate in sorted(_duplicates(records, field)):
            _issue(issues, "duplicate", f"$.{field}", f"duplicate ID: {duplicate}")

    documents_by_id = {record.get("doc_id"): record for record in documents}
    chunks_by_id = {record.get("chunk_id"): record for record in chunks}
    qa_by_id = {record.get("qid"): record for record in qa}
    atoms_by_id: dict[str, Mapping[str, Any]] = {}
    mappings_by_atom: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, atom in enumerate(source_atoms):
        atom_path = f"$.source_atom[{index}]"
        _required(
            atom,
            [
                "source_atom_id", "atom_schema_version", "doc_id", "document_sha256",
                "page_index", "page_label", "block_id", "source_span", "quote",
                "quote_sha256", "atom_type", "equivalence_group", "canonical_atom_id",
            ],
            atom_path,
            issues,
        )
        atom_id = atom.get("source_atom_id")
        if not isinstance(atom_id, str) or not re.fullmatch(r"sa:[0-9a-f]{24}", atom_id):
            _issue(issues, "id", f"{atom_path}.source_atom_id", "invalid source atom ID")
        elif atom_id in atoms_by_id:
            _issue(issues, "duplicate", f"{atom_path}.source_atom_id", "duplicate source atom ID")
        else:
            atoms_by_id[atom_id] = atom
        document = documents_by_id.get(atom.get("doc_id"))
        if document is None:
            _issue(issues, "reference", f"{atom_path}.doc_id", "unknown document")
        elif atom.get("document_sha256") != document.get("source_sha256"):
            _issue(issues, "hash", f"{atom_path}.document_sha256", "must match source document hash")
        quote = atom.get("quote")
        if not _nonempty_string(quote):
            _issue(issues, "type", f"{atom_path}.quote", "must be a non-empty string")
        elif hashlib.sha256(str(quote).encode("utf-8")).hexdigest() != atom.get("quote_sha256"):
            _issue(issues, "hash", f"{atom_path}.quote_sha256", "does not match quote")
        source_span = atom.get("source_span")
        if not isinstance(source_span, Mapping):
            _issue(issues, "type", f"{atom_path}.source_span", "must be a block span")
        else:
            start, end = source_span.get("start_char"), source_span.get("end_char")
            if (
                source_span.get("coordinate_space") != "block_text"
                or not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or end <= start
            ):
                _issue(issues, "range", f"{atom_path}.source_span", "invalid block span")
        canonical = atom.get("canonical_atom_id")
        if not isinstance(canonical, str) or not re.fullmatch(r"sa:[0-9a-f]{24}", canonical):
            _issue(issues, "id", f"{atom_path}.canonical_atom_id", "invalid canonical atom ID")
    for index, mapping in enumerate(atom_chunk_map):
        mapping_path = f"$.atom_chunk_map[{index}]"
        _required(
            mapping,
            [
                "mapping_schema_version", "source_atom_id", "chunk_id", "doc_id",
                "block_id", "overlap_source_span", "overlap_chars", "coverage",
                "mapping_kind",
            ],
            mapping_path,
            issues,
        )
        atom = atoms_by_id.get(mapping.get("source_atom_id"))
        chunk = chunks_by_id.get(mapping.get("chunk_id"))
        if atom is None:
            _issue(issues, "reference", f"{mapping_path}.source_atom_id", "unknown source atom")
        if chunk is None:
            _issue(issues, "reference", f"{mapping_path}.chunk_id", "unknown chunk")
        if atom is not None and mapping.get("doc_id") != atom.get("doc_id"):
            _issue(issues, "reference", f"{mapping_path}.doc_id", "differs from source atom")
        if atom is not None and mapping.get("block_id") != atom.get("block_id"):
            _issue(issues, "reference", f"{mapping_path}.block_id", "differs from source atom")
        if chunk is not None and mapping.get("doc_id") != chunk.get("doc_id"):
            _issue(issues, "reference", f"{mapping_path}.doc_id", "differs from chunk")
        overlap_span = mapping.get("overlap_source_span")
        overlap_start: int | None = None
        overlap_end: int | None = None
        if not isinstance(overlap_span, Mapping):
            _issue(issues, "type", f"{mapping_path}.overlap_source_span", "must be a block span")
        else:
            raw_start, raw_end = overlap_span.get("start_char"), overlap_span.get("end_char")
            if (
                overlap_span.get("coordinate_space") != "block_text"
                or not isinstance(raw_start, int) or isinstance(raw_start, bool)
                or not isinstance(raw_end, int) or isinstance(raw_end, bool)
                or raw_start < 0 or raw_end <= raw_start
            ):
                _issue(issues, "range", f"{mapping_path}.overlap_source_span", "invalid block span")
            else:
                overlap_start, overlap_end = raw_start, raw_end
        overlap_chars = mapping.get("overlap_chars")
        if not isinstance(overlap_chars, int) or isinstance(overlap_chars, bool) or overlap_chars < 1:
            _issue(issues, "range", f"{mapping_path}.overlap_chars", "must be a positive integer")
        elif overlap_start is not None and overlap_end is not None and overlap_chars != overlap_end - overlap_start:
            _issue(issues, "alignment", f"{mapping_path}.overlap_chars", "must equal overlap span length")
        coverage = mapping.get("coverage")
        if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0 < coverage <= 1:
            _issue(issues, "range", f"{mapping_path}.coverage", "must be in (0, 1]")
        if atom is not None and overlap_start is not None and overlap_end is not None:
            atom_start = int(atom["source_span"]["start_char"])
            atom_end = int(atom["source_span"]["end_char"])
            if overlap_start < atom_start or overlap_end > atom_end:
                _issue(issues, "alignment", f"{mapping_path}.overlap_source_span", "must lie within source atom")
            expected_coverage = (overlap_end - overlap_start) / (atom_end - atom_start)
            if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) and abs(float(coverage) - expected_coverage) > 1e-12:
                _issue(issues, "alignment", f"{mapping_path}.coverage", "does not match overlap/source-atom lengths")
            if mapping.get("mapping_kind") != ("contains" if expected_coverage == 1 else "partial"):
                _issue(issues, "value", f"{mapping_path}.mapping_kind", "does not match coverage")
        if chunk is not None and overlap_start is not None and overlap_end is not None:
            valid_overlap = any(
                span.get("block_id") == mapping.get("block_id")
                and max(int(atom["source_span"]["start_char"]), int(span.get("source_start_char", -1))) == overlap_start
                and min(int(atom["source_span"]["end_char"]), int(span.get("source_end_char", -1))) == overlap_end
                for span in chunk.get("source_spans", [])
            ) if atom is not None else False
            if atom is not None and not valid_overlap:
                _issue(issues, "alignment", f"{mapping_path}.overlap_source_span", "does not equal atom/chunk intersection")
        if isinstance(mapping.get("source_atom_id"), str):
            mappings_by_atom[str(mapping["source_atom_id"])].append(mapping)
    if source_atoms:
        for atom_id, atom in atoms_by_id.items():
            canonical = atoms_by_id.get(str(atom.get("canonical_atom_id")))
            if canonical is None:
                _issue(issues, "reference", f"$.source_atom.{atom_id}.canonical_atom_id", "unknown canonical atom")
            elif canonical.get("equivalence_group") != atom.get("equivalence_group"):
                _issue(issues, "equivalence", f"$.source_atom.{atom_id}", "canonical atom is in another group")
            if atom_chunk_map and atom_union_coverage(
                atom,
                {str(item.get("chunk_id")) for item in mappings_by_atom.get(atom_id, ())},
                mappings_by_atom.get(atom_id, ()),
            ) < 0.999999:
                _issue(issues, "mapping", f"$.source_atom.{atom_id}", "chunk mappings do not cover the atom")
    for index, chunk in enumerate(chunks):
        if chunk.get("doc_id") not in documents_by_id:
            _issue(issues, "reference", f"$.chunk[{index}].doc_id", "unknown document")
    family_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    scenario_family_splits: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(qa):
        qid = item.get("qid")
        family_id = item.get("question_family_id")
        split = item.get("split")
        if isinstance(family_id, str) and isinstance(split, str):
            family_splits[family_id].add(split)
        cluster_id = item.get("evidence_cluster_id")
        if isinstance(cluster_id, str) and isinstance(split, str):
            cluster_splits[cluster_id].add(split)
        scenario = item.get("scenario")
        if isinstance(scenario, Mapping) and isinstance(split, str):
            scenario_family_id = scenario.get("scenario_family_id")
            if isinstance(scenario_family_id, str):
                scenario_family_splits[scenario_family_id].add(split)
        gold_chunks: set[str] = set()
        for evidence_index, evidence in enumerate(item.get("evidence", [])):
            if not isinstance(evidence, Mapping):
                continue
            if source_atoms and isinstance(qid, str) and qid.startswith("qursor:p2:"):
                atom = atoms_by_id.get(str(evidence.get("source_atom_id")))
                evidence_path = f"$.qa[{index}].evidence[{evidence_index}]"
                if atom is None:
                    _issue(issues, "reference", f"{evidence_path}.source_atom_id", "unknown source atom")
                else:
                    for field_name in ("doc_id", "block_id", "page_index", "quote", "source_span"):
                        if evidence.get(field_name) != atom.get(field_name):
                            _issue(
                                issues,
                                "anchor",
                                f"{evidence_path}.{field_name}",
                                "must match the stable source atom",
                            )
            chunk_id = evidence.get("chunk_id")
            if chunk_id not in chunks_by_id:
                _issue(
                    issues,
                    "reference",
                    f"$.qa[{index}].evidence[{evidence_index}].chunk_id",
                    "unknown chunk",
                )
                continue
            resolution = None if evidence.get("visual_only") is True else resolve_evidence_span(evidence, chunks_by_id)
            if resolution is not None and not resolution.resolved:
                _issue(
                    issues,
                    "span_resolution",
                    f"$.qa[{index}].evidence[{evidence_index}].span",
                    resolution.reason or "unresolved evidence",
                )
            gold_chunks.add(str(chunk_id))
        for negative_index, negative in enumerate(item.get("hard_negatives", [])):
            if isinstance(negative, Mapping) and negative.get("chunk_id") not in chunks_by_id:
                _issue(
                    issues,
                    "reference",
                    f"$.qa[{index}].hard_negatives[{negative_index}].chunk_id",
                    "unknown chunk",
                )
        if qid is None:
            continue
        positive_qrels = {
            record.get("chunk_id")
            for record in qrels
            if record.get("qid") == qid and record.get("relevance") == 2
        }
        if qrels and not gold_chunks.issubset(positive_qrels):
            missing = sorted(gold_chunks - positive_qrels)
            _issue(issues, "qrels", f"$.qa[{index}]", f"gold chunks missing relevance=2 qrels: {missing}")
    for family_id, splits in sorted(family_splits.items()):
        if len(splits) > 1:
            _issue(
                issues,
                "leakage",
                "$.qa.question_family_id",
                f"{family_id} crosses splits: {sorted(splits)}",
            )
    for cluster_id, splits in sorted(cluster_splits.items()):
        if len(splits) > 1:
            _issue(
                issues,
                "leakage",
                "$.qa.evidence_cluster_id",
                f"{cluster_id} crosses splits: {sorted(splits)}",
            )
    for scenario_family_id, splits in sorted(scenario_family_splits.items()):
        if len(splits) > 1:
            _issue(
                issues,
                "leakage",
                "$.qa.scenario.scenario_family_id",
                f"{scenario_family_id} crosses splits: {sorted(splits)}",
            )
    qrel_keys: set[tuple[Any, Any]] = set()
    evidence_by_qid: dict[str, set[str]] = {
        str(item.get("qid")): {
            evidence.get("evidence_id")
            for evidence in item.get("evidence", [])
            if isinstance(evidence, Mapping) and isinstance(evidence.get("evidence_id"), str)
        }
        for item in qa
    }
    evidence_records_by_qid: dict[str, dict[str, Mapping[str, Any]]] = {
        str(item.get("qid")): {
            str(evidence["evidence_id"]): evidence
            for evidence in item.get("evidence", [])
            if isinstance(evidence, Mapping) and isinstance(evidence.get("evidence_id"), str)
        }
        for item in qa
    }
    atom_qrel_keys: set[tuple[str, str]] = set()
    atom_qrel_atoms_by_qid: dict[str, set[str]] = defaultdict(set)
    for index, atom_qrel in enumerate(atom_qrels):
        atom_qrel_path = f"$.atom_qrel[{index}]"
        _required(
            atom_qrel,
            [
                "atom_qrel_schema_version", "qid", "source_atom_id",
                "canonical_atom_id", "equivalence_group", "relevance",
                "evidence_ids", "roles", "hop_ids",
            ],
            atom_qrel_path,
            issues,
        )
        qid = str(atom_qrel.get("qid", ""))
        atom_id = str(atom_qrel.get("source_atom_id", ""))
        key = (qid, atom_id)
        if key in atom_qrel_keys:
            _issue(issues, "duplicate", atom_qrel_path, "duplicate qid/source_atom_id pair")
        atom_qrel_keys.add(key)
        atom_qrel_atoms_by_qid[qid].add(atom_id)
        qa_item = qa_by_id.get(qid)
        atom = atoms_by_id.get(atom_id)
        if qa_item is None:
            _issue(issues, "reference", f"{atom_qrel_path}.qid", "unknown question")
        if atom is None:
            _issue(issues, "reference", f"{atom_qrel_path}.source_atom_id", "unknown source atom")
        else:
            if atom_qrel.get("canonical_atom_id") != atom.get("canonical_atom_id"):
                _issue(issues, "reference", f"{atom_qrel_path}.canonical_atom_id", "differs from source atom")
            if atom_qrel.get("equivalence_group") != atom.get("equivalence_group"):
                _issue(issues, "reference", f"{atom_qrel_path}.equivalence_group", "differs from source atom")
        if atom_qrel.get("atom_qrel_schema_version") != "aqrv1":
            _issue(issues, "value", f"{atom_qrel_path}.atom_qrel_schema_version", "must be aqrv1")
        if atom_qrel.get("relevance") != 2:
            _issue(issues, "value", f"{atom_qrel_path}.relevance", "gold source atoms require relevance=2")
        evidence_ids = atom_qrel.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            _issue(issues, "type", f"{atom_qrel_path}.evidence_ids", "must be a non-empty unique array")
        else:
            for evidence_id in evidence_ids:
                evidence = evidence_records_by_qid.get(qid, {}).get(str(evidence_id))
                if evidence is None:
                    _issue(issues, "reference", f"{atom_qrel_path}.evidence_ids", f"unknown evidence: {evidence_id}")
                elif evidence.get("source_atom_id") != atom_id:
                    _issue(issues, "reference", f"{atom_qrel_path}.evidence_ids", f"evidence {evidence_id} belongs to another atom")
        roles = atom_qrel.get("roles")
        if (
            not isinstance(roles, list) or not roles or len(roles) != len(set(roles))
            or any(role not in {"answer", "bridge", "constraint", "alternate", "visual"} for role in roles)
        ):
            _issue(issues, "enum", f"{atom_qrel_path}.roles", "invalid roles")
        hop_ids = atom_qrel.get("hop_ids")
        if (
            not isinstance(hop_ids, list) or len(hop_ids) != len(set(hop_ids))
            or any(not isinstance(hop, int) or isinstance(hop, bool) or not 1 <= hop <= 3 for hop in (hop_ids or []))
        ):
            _issue(issues, "range", f"{atom_qrel_path}.hop_ids", "must contain unique hop integers 1..3")
    if atom_qrels:
        for index, item in enumerate(qa):
            qid = str(item.get("qid", ""))
            expected_atoms = {
                str(evidence.get("source_atom_id"))
                for evidence in item.get("evidence", [])
                if isinstance(evidence, Mapping) and evidence.get("source_atom_id") is not None
            }
            missing_atoms = expected_atoms - atom_qrel_atoms_by_qid.get(qid, set())
            extra_atoms = atom_qrel_atoms_by_qid.get(qid, set()) - expected_atoms
            if missing_atoms or extra_atoms:
                _issue(
                    issues,
                    "atom_qrels",
                    f"$.qa[{index}]",
                    f"source atom qrels differ; missing={sorted(missing_atoms)}, extra={sorted(extra_atoms)}",
                )
    for index, qrel in enumerate(qrels):
        key = (qrel.get("qid"), qrel.get("chunk_id"))
        if key in qrel_keys:
            _issue(issues, "duplicate", f"$.qrel[{index}]", "duplicate qid/chunk_id pair")
        qrel_keys.add(key)
        if qrel.get("qid") not in qa_by_id:
            _issue(issues, "reference", f"$.qrel[{index}].qid", "unknown question")
        if qrel.get("chunk_id") not in chunks_by_id:
            _issue(issues, "reference", f"$.qrel[{index}].chunk_id", "unknown chunk")
        unknown_evidence = set(qrel.get("evidence_ids", [])) - evidence_by_qid.get(str(qrel.get("qid")), set())
        if unknown_evidence:
            _issue(
                issues,
                "reference",
                f"$.qrel[{index}].evidence_ids",
                f"unknown evidence IDs: {sorted(unknown_evidence)}",
            )
    if quota is not None:
        issues.extend(validate_quota(qa, quota))
    if issues and raise_on_error:
        raise SchemaValidationError(issues)
    return issues
