"""QEGS: deterministic evidence-first dataset construction and validation."""

from .ids import (
    make_block_id,
    make_chunk_id,
    make_document_id,
    make_evidence_id,
    make_question_id,
)
from .manifest import build_manifest, verify_manifest
from .validation import (
    DatasetQuota,
    SchemaValidationError,
    ValidationIssue,
    phase1_pilot_quota,
    phase1_quota,
    validate_dataset,
    validate_quota,
    validate_record,
)

__all__ = [
    "DatasetQuota",
    "SchemaValidationError",
    "ValidationIssue",
    "build_manifest",
    "make_block_id",
    "make_chunk_id",
    "make_document_id",
    "make_evidence_id",
    "make_question_id",
    "phase1_pilot_quota",
    "phase1_quota",
    "validate_dataset",
    "validate_quota",
    "validate_record",
    "verify_manifest",
]

__version__ = "0.1.0"
