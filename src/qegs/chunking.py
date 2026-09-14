"""Deterministic structure-preserving chunk construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

from .ids import make_block_id, make_chunk_id
from .model import CanonicalBlock, Page, PageBlock


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def pages_to_blocks(pages: Sequence[Page], doc_id: str) -> list[PageBlock]:
    blocks: list[PageBlock] = []
    for page in sorted(pages, key=lambda value: value.page_index):
        if page.blocks:
            page_blocks = page.blocks
        else:
            paragraphs = [part for part in PARAGRAPH_SPLIT_RE.split(page.text) if part.strip()]
            if not paragraphs and page.text.strip():
                paragraphs = [page.text]
            page_blocks = tuple(
                PageBlock(
                    page_index=page.page_index,
                    page_label=page.page_label,
                    block_index=index,
                    text=paragraph,
                )
                for index, paragraph in enumerate(paragraphs)
            )
        for ordinal, block in enumerate(page_blocks):
            text = _clean_text(block.text)
            if not text and block.content_type != "visual":
                continue
            blocks.append(
                PageBlock(
                    page_index=page.page_index,
                    page_label=page.page_label,
                    block_index=ordinal,
                    text=text,
                    section_path=block.section_path,
                    content_type=block.content_type,
                    visual_ref=block.visual_ref,
                    figure_id=block.figure_id,
                    bbox=block.bbox,
                    ocr_text=block.ocr_text,
                    visual_only=block.visual_only,
                    safety_sensitive=block.safety_sensitive,
                    equivalence_group_id=block.equivalence_group_id,
                    canonical_chunk_id=block.canonical_chunk_id,
                    duplicate_lineage=block.duplicate_lineage,
                )
            )
    return blocks


def build_canonical_text(
    pages: Sequence[Page], doc_id: str
) -> tuple[str, list[CanonicalBlock]]:
    raw_blocks = pages_to_blocks(pages, doc_id)
    text_parts: list[str] = []
    canonical_blocks: list[CanonicalBlock] = []
    cursor = 0
    text_block_count = 0
    for block in raw_blocks:
        if block.content_type == "visual":
            canonical_blocks.append(
                CanonicalBlock(
                    block_id=make_block_id(doc_id, block.page_index, block.block_index),
                    page_index=block.page_index,
                    page_label=block.page_label,
                    block_index=block.block_index,
                    text=block.text,
                    section_path=block.section_path,
                    content_type=block.content_type,
                    global_start=cursor,
                    global_end=cursor,
                    visual_ref=block.visual_ref,
                    figure_id=block.figure_id,
                    bbox=block.bbox,
                    ocr_text=block.ocr_text,
                    visual_only=block.visual_only,
                    safety_sensitive=block.safety_sensitive,
                    equivalence_group_id=block.equivalence_group_id,
                    canonical_chunk_id=block.canonical_chunk_id,
                    duplicate_lineage=block.duplicate_lineage,
                )
            )
            continue
        if text_block_count:
            text_parts.append("\n\n")
            cursor += 2
        text_block_count += 1
        start = cursor
        text_parts.append(block.text)
        cursor += len(block.text)
        canonical_blocks.append(
            CanonicalBlock(
                block_id=make_block_id(doc_id, block.page_index, block.block_index),
                page_index=block.page_index,
                page_label=block.page_label,
                block_index=block.block_index,
                text=block.text,
                section_path=block.section_path,
                content_type=block.content_type,
                global_start=start,
                global_end=cursor,
                safety_sensitive=block.safety_sensitive,
                equivalence_group_id=block.equivalence_group_id,
                canonical_chunk_id=block.canonical_chunk_id,
                duplicate_lineage=block.duplicate_lineage,
            )
        )
    return "".join(text_parts), canonical_blocks


def _common_prefix(paths: Iterable[tuple[str, ...]]) -> list[str]:
    values = list(paths)
    if not values:
        return []
    prefix = list(values[0])
    for path in values[1:]:
        while prefix and tuple(prefix) != path[: len(prefix)]:
            prefix.pop()
    return prefix


def _window_end(text: str, start: int, max_chars: int) -> int:
    proposed = min(len(text), start + max_chars)
    if proposed == len(text):
        return proposed
    floor = start + int(max_chars * 0.7)
    paragraph_break = text.rfind("\n\n", floor, proposed)
    if paragraph_break > start:
        return paragraph_break
    break_at = text.rfind(" ", floor, proposed)
    return break_at if break_at > start else proposed


def _next_start(text: str, previous_start: int, end: int, overlap_chars: int) -> int:
    candidate = max(previous_start + 1, end - overlap_chars)
    if candidate <= 0:
        return 0
    while candidate < end and text[candidate].isspace():
        candidate += 1
    return candidate


def chunk_pages(
    pages: Sequence[Page],
    doc_id: str,
    *,
    max_chars: int = 2_400,
    overlap_chars: int = 240,
    schema_version: str = "ckv1",
) -> tuple[list[dict], list[CanonicalBlock]]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars // 2:
        raise ValueError("overlap_chars must be non-negative and less than half max_chars")
    canonical_text, blocks = build_canonical_text(pages, doc_id)

    chunks: list[dict] = []
    start = 0
    ordinal = 0
    while start < len(canonical_text):
        while start < len(canonical_text) and canonical_text[start].isspace():
            start += 1
        if start >= len(canonical_text):
            break
        end = _window_end(canonical_text, start, max_chars)
        while end > start and canonical_text[end - 1].isspace():
            end -= 1
        if end <= start:
            break
        intersecting = [
            block
            for block in blocks
            if block.content_type != "visual"
            and block.global_end > start
            and block.global_start < end
        ]
        source_spans: list[dict] = []
        for block in intersecting:
            intersection_start = max(start, block.global_start)
            intersection_end = min(end, block.global_end)
            source_spans.append(
                {
                    "block_id": block.block_id,
                    "page_index": block.page_index,
                    "page_label": block.page_label,
                    "source_start_char": intersection_start - block.global_start,
                    "source_end_char": intersection_end - block.global_start,
                    "chunk_start_char": intersection_start - start,
                    "chunk_end_char": intersection_end - start,
                }
            )
        content_types = sorted({block.content_type for block in intersecting})
        chunk = {
            "chunk_id": make_chunk_id(doc_id, ordinal, schema_version=schema_version),
            "chunk_schema_version": schema_version,
            "doc_id": doc_id,
            "text": canonical_text[start:end],
            "token_count": len(re.findall(r"\S+", canonical_text[start:end])),
            "section_path": _common_prefix(block.section_path for block in intersecting),
            "page_start": min(block.page_index for block in intersecting),
            "page_end": max(block.page_index for block in intersecting),
            "content_type": content_types[0] if len(content_types) == 1 else "mixed",
            "source_spans": source_spans,
            "safety_sensitive": any(block.safety_sensitive for block in intersecting),
            "duplicate_lineage": sorted(
                {item for block in intersecting for item in block.duplicate_lineage}
            ),
        }
        groups = {block.equivalence_group_id for block in intersecting if block.equivalence_group_id}
        if len(groups) == 1:
            chunk["equivalence_group_id"] = next(iter(groups))
        chunks.append(chunk)
        ordinal += 1
        if end == len(canonical_text):
            break
        start = _next_start(canonical_text, start, end, overlap_chars)
    for block in (item for item in blocks if item.content_type == "visual"):
        visual_text = block.ocr_text or block.text
        source_spans = []
        if visual_text:
            source_spans.append(
                {
                    "block_id": block.block_id,
                    "page_index": block.page_index,
                    "page_label": block.page_label,
                    "source_start_char": 0,
                    "source_end_char": len(visual_text),
                    "chunk_start_char": 0,
                    "chunk_end_char": len(visual_text),
                }
            )
        visual_chunk = {
            "chunk_id": make_chunk_id(doc_id, ordinal, schema_version=schema_version),
            "chunk_schema_version": schema_version,
            "doc_id": doc_id,
            "text": visual_text,
            "token_count": len(re.findall(r"\S+", visual_text)),
            "section_path": list(block.section_path),
            "page_start": block.page_index,
            "page_end": block.page_index,
            "content_type": "visual",
            "source_spans": source_spans,
            "visual_ref": block.visual_ref,
            "figure_id": block.figure_id,
            "bbox": list(block.bbox) if block.bbox is not None else None,
            "ocr_text": block.ocr_text,
            "visual_only": block.visual_only,
            "safety_sensitive": block.safety_sensitive,
            "duplicate_lineage": list(block.duplicate_lineage),
        }
        if block.equivalence_group_id:
            visual_chunk["equivalence_group_id"] = block.equivalence_group_id
        if block.canonical_chunk_id:
            visual_chunk["canonical_chunk_id"] = block.canonical_chunk_id
        chunks.append(visual_chunk)
        ordinal += 1
    return chunks, blocks
