"""Small dependency-free data objects used during deterministic corpus building."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PageBlock:
    page_index: int
    page_label: str
    block_index: int
    text: str
    section_path: tuple[str, ...] = field(default_factory=tuple)
    content_type: str = "prose"
    visual_ref: str | None = None
    figure_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    ocr_text: str = ""
    visual_only: bool = False
    safety_sensitive: bool = False
    equivalence_group_id: str | None = None
    canonical_chunk_id: str | None = None
    duplicate_lineage: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Page:
    page_index: int
    page_label: str
    text: str = ""
    blocks: tuple[PageBlock, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    block_id: str
    page_index: int
    page_label: str
    block_index: int
    text: str
    section_path: tuple[str, ...]
    content_type: str
    global_start: int
    global_end: int
    visual_ref: str | None = None
    figure_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    ocr_text: str = ""
    visual_only: bool = False
    safety_sensitive: bool = False
    equivalence_group_id: str | None = None
    canonical_chunk_id: str | None = None
    duplicate_lineage: tuple[str, ...] = field(default_factory=tuple)
