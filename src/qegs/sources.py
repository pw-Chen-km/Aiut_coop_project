"""Read extracted pages or PDFs without requiring a network service."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable

from .model import Page, PageBlock


class SourceExtractionError(RuntimeError):
    """Raised when a local input cannot be converted to pages."""


def _page_from_mapping(raw: dict[str, Any], default_index: int) -> Page:
    page_index = int(raw.get("page_index", default_index))
    page_label = str(raw.get("page_label", page_index + 1))
    raw_blocks = raw.get("blocks") or []
    blocks: list[PageBlock] = []
    for block_index, block in enumerate(raw_blocks):
        if isinstance(block, str):
            block = {"text": block}
        if not isinstance(block, dict):
            raise SourceExtractionError(
                f"page {page_index} block {block_index} must be a string or object with text"
            )
        content_type = str(block.get("content_type", "prose"))
        block_text = block.get("text", block.get("ocr_text", ""))
        if not isinstance(block_text, str):
            raise SourceExtractionError(f"page {page_index} block {block_index} text must be a string")
        bbox = block.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox
            ):
                raise SourceExtractionError(
                    f"page {page_index} block {block_index} bbox must contain four numbers"
                )
            bbox = tuple(float(value) for value in bbox)
        section_path = block.get("section_path", raw.get("section_path", []))
        if isinstance(section_path, str):
            section_path = [section_path]
        blocks.append(
            PageBlock(
                page_index=page_index,
                page_label=page_label,
                block_index=block_index,
                text=block_text,
                section_path=tuple(str(item) for item in section_path),
                content_type=content_type,
                visual_ref=block.get("visual_ref"),
                figure_id=block.get("figure_id"),
                bbox=bbox,
                ocr_text=str(block.get("ocr_text", "")),
                visual_only=bool(block.get("visual_only", content_type == "visual")),
                safety_sensitive=bool(block.get("safety_sensitive", False)),
                equivalence_group_id=block.get("equivalence_group_id"),
                canonical_chunk_id=block.get("canonical_chunk_id"),
                duplicate_lineage=tuple(str(item) for item in block.get("duplicate_lineage", [])),
            )
        )
    text = raw.get("text", "")
    if not isinstance(text, str):
        raise SourceExtractionError(f"page {page_index} text must be a string")
    return Page(page_index=page_index, page_label=page_label, text=text, blocks=tuple(blocks))


def _read_json_pages(path: Path) -> list[Page]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceExtractionError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SourceExtractionError(f"JSONL line {line_number} must contain an object")
            rows.append(value)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("pages")
        if not isinstance(value, list):
            raise SourceExtractionError("JSON page input must be an array or an object with pages")
        rows = value
    pages = [_page_from_mapping(row, index) for index, row in enumerate(rows)]
    return _normalize_page_order(pages)


def _normalize_page_order(pages: Iterable[Page]) -> list[Page]:
    result = sorted(pages, key=lambda page: (page.page_index, page.page_label))
    indices = [page.page_index for page in result]
    if len(indices) != len(set(indices)):
        raise SourceExtractionError("page_index values must be unique")
    return result


def _read_text_pages(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8")
    raw_pages = text.split("\f")
    return [
        Page(page_index=index, page_label=str(index + 1), text=page)
        for index, page in enumerate(raw_pages)
        if page.strip()
    ]


def _read_directory_pages(path: Path) -> list[Page]:
    page_files = sorted(
        (candidate for candidate in path.iterdir() if candidate.suffix.lower() in {".txt", ".text"}),
        key=lambda candidate: candidate.name,
    )
    if not page_files:
        json_candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if len(json_candidates) == 1:
            return _read_json_pages(json_candidates[0])
        raise SourceExtractionError("page directory must contain .txt files or one JSON/JSONL page file")
    pages: list[Page] = []
    for index, page_file in enumerate(page_files):
        match = re.search(r"([0-9]+)", page_file.stem)
        page_label = match.group(1) if match else str(index + 1)
        pages.append(
            Page(
                page_index=index,
                page_label=page_label,
                text=page_file.read_text(encoding="utf-8"),
            )
        )
    return pages


def _extract_pdf_with_pypdf(path: Path) -> list[Page]:
    from pypdf import PdfReader  # type: ignore[import-not-found]

    reader = PdfReader(str(path))
    return [
        Page(page_index=index, page_label=str(index + 1), text=page.extract_text() or "")
        for index, page in enumerate(reader.pages)
    ]


def _extract_pdf_with_pdftotext(path: Path, executable: str) -> list[Page]:
    completed = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        raise SourceExtractionError(f"pdftotext failed: {message}")
    text = completed.stdout.decode("utf-8", "replace")
    return [
        Page(page_index=index, page_label=str(index + 1), text=page)
        for index, page in enumerate(text.split("\f"))
        if page.strip()
    ]


def read_pages(path: str | Path) -> tuple[list[Page], str]:
    """Return pages and the deterministic extraction method label.

    PDF extraction uses an installed ``pypdf`` package first and the local
    ``pdftotext`` executable second. No remote service is contacted.
    """

    source = Path(path)
    if not source.exists():
        raise SourceExtractionError(f"input does not exist: {source}")
    if source.is_dir():
        return _read_directory_pages(source), "extracted-directory"
    suffix = source.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _read_json_pages(source), "extracted-json"
    if suffix in {".txt", ".text"}:
        return _read_text_pages(source), "extracted-text"
    if suffix == ".pdf":
        if importlib.util.find_spec("pypdf") is not None:
            return _extract_pdf_with_pypdf(source), "pypdf"
        executable = shutil.which("pdftotext")
        if executable:
            return _extract_pdf_with_pdftotext(source, executable), "pdftotext-layout"
        raise SourceExtractionError(
            "PDF extraction requires the optional 'pypdf' package or local pdftotext; "
            "alternatively provide extracted .jsonl/.json/.txt pages"
        )
    raise SourceExtractionError(f"unsupported input type: {source.suffix or '<none>'}")
