"""Evidence candidate generation and exact span resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
import unicodedata

from .ids import make_equivalence_group
from .model import CanonicalBlock


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？;；])(?:[ \t]+|\n+)|\n+")


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    evidence_id: str
    resolved: bool
    resolved_text: str | None = None
    reason: str | None = None


def normalize_quote(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def _sentence_spans(text: str, min_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        if len(text[start:end].strip()) >= min_chars:
            left = start
            right = end
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            spans.append((left, right))
        start = match.end()
    if len(text[start:].strip()) >= min_chars:
        left = start
        right = len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        spans.append((left, right))
    if not spans and len(text.strip()) >= min_chars:
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        spans.append((left, right))
    return spans


def build_evidence_candidates(
    document: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    blocks: Sequence[CanonicalBlock],
    *,
    min_chars: int = 24,
) -> list[dict]:
    """Create one canonical candidate per sentence anchored to its first full chunk."""

    candidates: list[dict] = []
    for block in blocks:
        if block.content_type == "visual":
            visual_chunk = next(
                (
                    chunk
                    for chunk in chunks
                    if chunk.get("content_type") == "visual"
                    and chunk.get("figure_id") == block.figure_id
                    and chunk.get("page_start") == block.page_index
                ),
                None,
            )
            if visual_chunk is None:
                continue
            candidate_id = f"{block.block_id}:s000"
            candidate = {
                "evidence_candidate_id": candidate_id,
                "doc_id": document["doc_id"],
                "chunk_id": visual_chunk["chunk_id"],
                "block_id": block.block_id,
                "page_index": block.page_index,
                "page_label": block.page_label,
                "section_path": list(block.section_path),
                "role": "visual",
                "equivalence_group": block.equivalence_group_id
                or make_equivalence_group(candidate_id),
                "citation": (
                    f"[{document['title']} {document['document_version']}, "
                    f"p.{block.page_label}, {block.figure_id or candidate_id}]"
                ),
                "visual_ref": block.visual_ref,
                "figure_id": block.figure_id,
                "bbox": list(block.bbox) if block.bbox is not None else None,
                "ocr_text": block.ocr_text,
                "visual_only": block.visual_only,
                "safety_sensitive": block.safety_sensitive,
            }
            visual_text = visual_chunk.get("text", "")
            if not block.visual_only and visual_text:
                candidate.update(
                    {
                        "span": {
                            "coordinate_space": "chunk_text",
                            "start_char": 0,
                            "end_char": len(visual_text),
                        },
                        "source_span": {
                            "coordinate_space": "block_text",
                            "start_char": 0,
                            "end_char": len(visual_text),
                        },
                        "quote": visual_text,
                    }
                )
            candidates.append(candidate)
            continue
        for sentence_index, (source_start, source_end) in enumerate(
            _sentence_spans(block.text, min_chars)
        ):
            canonical_chunk: Mapping[str, Any] | None = None
            canonical_span: Mapping[str, Any] | None = None
            for chunk in chunks:
                for span in chunk.get("source_spans", []):
                    if (
                        span.get("block_id") == block.block_id
                        and span.get("source_start_char", 0) <= source_start
                        and span.get("source_end_char", 0) >= source_end
                    ):
                        canonical_chunk = chunk
                        canonical_span = span
                        break
                if canonical_chunk is not None:
                    break
            if canonical_chunk is None or canonical_span is None:
                continue
            chunk_start = canonical_span["chunk_start_char"] + (
                source_start - canonical_span["source_start_char"]
            )
            chunk_end = chunk_start + (source_end - source_start)
            quote = canonical_chunk["text"][chunk_start:chunk_end]
            candidate_id = f"{block.block_id}:s{sentence_index:03d}"
            candidates.append(
                {
                    "evidence_candidate_id": candidate_id,
                    "doc_id": document["doc_id"],
                    "chunk_id": canonical_chunk["chunk_id"],
                    "block_id": block.block_id,
                    "page_index": block.page_index,
                    "page_label": block.page_label,
                    "section_path": list(block.section_path),
                    "span": {
                        "coordinate_space": "chunk_text",
                        "start_char": chunk_start,
                        "end_char": chunk_end,
                    },
                    "source_span": {
                        "coordinate_space": "block_text",
                        "start_char": source_start,
                        "end_char": source_end,
                    },
                    "quote": quote,
                    "role": "candidate",
                    "equivalence_group": make_equivalence_group(candidate_id),
                    "citation": (
                        f"[{document['title']} {document['document_version']}, "
                        f"p.{block.page_label}, {candidate_id}]"
                    ),
                }
            )
    return candidates


def resolve_evidence_span(
    evidence: Mapping[str, Any], chunks_by_id: Mapping[str, Mapping[str, Any]]
) -> EvidenceResolution:
    evidence_id = str(evidence.get("evidence_id") or evidence.get("evidence_candidate_id") or "")
    chunk_id = evidence.get("chunk_id")
    chunk = chunks_by_id.get(str(chunk_id))
    if chunk is None:
        return EvidenceResolution(evidence_id, False, reason=f"unknown chunk_id: {chunk_id}")
    if evidence.get("doc_id") != chunk.get("doc_id"):
        return EvidenceResolution(evidence_id, False, reason="evidence doc_id differs from chunk doc_id")
    block_id = evidence.get("block_id")
    matching_block_spans = [
        item for item in chunk.get("source_spans", []) if item.get("block_id") == block_id
    ]
    if not matching_block_spans:
        return EvidenceResolution(evidence_id, False, reason="block_id is not represented by chunk")
    span = evidence.get("span")
    if not isinstance(span, Mapping):
        return EvidenceResolution(evidence_id, False, reason="missing span object")
    if span.get("coordinate_space", "chunk_text") != "chunk_text":
        return EvidenceResolution(evidence_id, False, reason="only chunk_text spans are directly resolvable")
    start = span.get("start_char")
    end = span.get("end_char")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return EvidenceResolution(evidence_id, False, reason="invalid span bounds")
    text = chunk.get("text")
    if not isinstance(text, str) or end > len(text):
        return EvidenceResolution(evidence_id, False, reason="span exceeds chunk text")
    resolved_text = text[start:end]
    quote = evidence.get("quote")
    if not isinstance(quote, str) or normalize_quote(quote) != normalize_quote(resolved_text):
        return EvidenceResolution(
            evidence_id,
            False,
            resolved_text=resolved_text,
            reason="quote does not match resolved span",
        )
    containing_source_span = next(
        (
            item
            for item in matching_block_spans
            if item.get("chunk_start_char", -1) <= start
            and item.get("chunk_end_char", -1) >= end
        ),
        None,
    )
    if containing_source_span is None:
        return EvidenceResolution(
            evidence_id,
            False,
            resolved_text=resolved_text,
            reason="chunk span is not contained by the declared source block",
        )
    source_span = evidence.get("source_span")
    if isinstance(source_span, Mapping):
        expected_start = containing_source_span["source_start_char"] + (
            start - containing_source_span["chunk_start_char"]
        )
        expected_end = expected_start + (end - start)
        if (
            source_span.get("start_char") != expected_start
            or source_span.get("end_char") != expected_end
        ):
            return EvidenceResolution(
                evidence_id,
                False,
                resolved_text=resolved_text,
                reason="source_span is inconsistent with chunk/block alignment",
            )
    return EvidenceResolution(evidence_id, True, resolved_text=resolved_text)
