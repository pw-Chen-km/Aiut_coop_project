"""Stable identifiers used by the QEGS corpus and QA records."""

from __future__ import annotations

import hashlib
import re
import unicodedata


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DOC_ID_RE = re.compile(
    r"^qursor:[a-z0-9][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*:[0-9a-f]{12}$"
)
CHUNK_ID_RE = re.compile(
    r"^(?P<doc>qursor:[a-z0-9][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*:[0-9a-f]{12})"
    r":(?P<schema>ckv[1-9][0-9]*):c(?P<ordinal>[0-9]{5})$"
)
BLOCK_ID_RE = re.compile(
    r"^(?P<doc>qursor:[a-z0-9][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*:[0-9a-f]{12})"
    r":p(?P<page>[0-9]{4}):b(?P<ordinal>[0-9]{4})$"
)
QID_RE = re.compile(
    r"^qursor:(?P<phase>p[1-9][0-9]*):(?P<split>train|dev|test|test_version):q(?P<ordinal>[0-9]{6})$"
)
EVIDENCE_ID_RE = re.compile(
    r"^(?P<qid>qursor:p[1-9][0-9]*:(?:train|dev|test|test_version):q[0-9]{6}):e(?P<ordinal>[0-9]{2,4})$"
)
EVIDENCE_CANDIDATE_ID_RE = re.compile(
    r"^(?P<block>qursor:[a-z0-9][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*:[0-9a-f]{12}:p[0-9]{4}:b[0-9]{4})"
    r":s(?P<ordinal>[0-9]{3})$"
)
EQUIVALENCE_GROUP_RE = re.compile(r"^eg:[a-z0-9][a-z0-9._:-]{2,127}$")


def slugify(value: str, *, fallback: str = "document") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug or fallback


def normalize_version(version: str) -> str:
    value = unicodedata.normalize("NFKC", version).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9._-]", "-", value).strip("-.")
    if not value:
        value = "unversioned"
    if value[0].isdigit():
        value = f"v{value}"
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_document_id(family: str, version: str, source_sha256: str) -> str:
    digest = source_sha256.lower()
    if not HEX64_RE.fullmatch(digest):
        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 digest")
    return f"qursor:{slugify(family)}:{normalize_version(version)}:{digest[:12]}"


def make_chunk_id(doc_id: str, ordinal: int, *, schema_version: str = "ckv1") -> str:
    if not DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f"invalid doc_id: {doc_id}")
    if not re.fullmatch(r"ckv[1-9][0-9]*", schema_version):
        raise ValueError(f"invalid chunk schema version: {schema_version}")
    if not 0 <= ordinal <= 99_999:
        raise ValueError("chunk ordinal must be between 0 and 99999")
    return f"{doc_id}:{schema_version}:c{ordinal:05d}"


def make_block_id(doc_id: str, page_index: int, ordinal: int) -> str:
    if not DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f"invalid doc_id: {doc_id}")
    if not 0 <= page_index <= 9_999 or not 0 <= ordinal <= 9_999:
        raise ValueError("page and block ordinals must be between 0 and 9999")
    return f"{doc_id}:p{page_index:04d}:b{ordinal:04d}"


def make_question_id(split: str, ordinal: int, *, phase: str = "p1") -> str:
    if split not in {"train", "dev", "test", "test_version"}:
        raise ValueError(f"invalid split: {split}")
    if not re.fullmatch(r"p[1-9][0-9]*", phase):
        raise ValueError(f"invalid phase: {phase}")
    if not 0 <= ordinal <= 999_999:
        raise ValueError("question ordinal must be between 0 and 999999")
    return f"qursor:{phase}:{split}:q{ordinal:06d}"


def make_evidence_id(qid: str, ordinal: int) -> str:
    if not QID_RE.fullmatch(qid):
        raise ValueError(f"invalid qid: {qid}")
    if not 0 <= ordinal <= 9_999:
        raise ValueError("evidence ordinal must be between 0 and 9999")
    width = max(2, len(str(ordinal)))
    return f"{qid}:e{ordinal:0{width}d}"


def make_equivalence_group(*parts: str) -> str:
    material = "\x1f".join(unicodedata.normalize("NFKC", part).strip() for part in parts)
    return f"eg:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"
