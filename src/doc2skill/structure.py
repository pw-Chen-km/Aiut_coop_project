"""Lossless canonical text/structure projection of a Docling JSON document.

This module is dependency-free. Generated navigation descriptions never enter the
canonical source layer. Offsets always address the exact ``block['text']`` string.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def ordered_items(raw: dict) -> tuple[list[dict], list[str]]:
    """Walk native references; retain unlinked items with an explicit audit warning."""
    nodes: dict[str, dict] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and "self_ref" in item:
                    nodes[item["self_ref"]] = {**item, "_collection": key}
        elif isinstance(value, dict) and "self_ref" in value:
            nodes[value["self_ref"]] = {**value, "_collection": key}
    visited: set[str] = set()
    output: list[dict] = []
    warnings: list[str] = []

    def visit(ref: str, layer: str = "body", orphan: bool = False) -> None:
        if ref in visited:
            return
        visited.add(ref)
        item = nodes.get(ref)
        if item is None:
            warnings.append(f"unresolved_reference:{ref}")
            return
        item = {**item, "_layer": item.get("content_layer", layer), "_orphan": orphan}
        collection = item.get("_collection")
        if collection not in {"body", "furniture", "groups"}:
            output.append(item)
        for child in item.get("children", []):
            visit(child.get("$ref", child.get("cref", "")), item["_layer"], orphan)
        # Captions can be references outside the body's direct child list.
        for child in item.get("captions", []):
            visit(child.get("$ref", child.get("cref", "")), item["_layer"], orphan)

    if raw.get("body", {}).get("self_ref"):
        visit(raw["body"]["self_ref"])
    if raw.get("furniture", {}).get("self_ref"):
        visit(raw["furniture"]["self_ref"], "furniture")
    for ref, item in nodes.items():
        if ref not in visited and item.get("_collection") not in {"body", "groups", "furniture"}:
            warnings.append(f"unlinked_item_appended_unknown_reading_order:{ref}")
            visit(ref, str(item.get("content_layer", "body")), True)
    return output, warnings


def _provenance(item: dict) -> list[dict]:
    result = []
    for prov in item.get("prov", []):
        page_no = prov.get("page_no")
        result.append({**prov, "page_index": page_no - 1 if isinstance(page_no, int) else None})
    return result


def _page(item: dict) -> int | None:
    return next((p["page_index"] for p in _provenance(item) if p["page_index"] is not None), None)


_CONTINUATION = re.compile(r"^(.*?)\s*\(?\s*(\d+)\s*/\s*(\d+)\s*\)?\s*$")


def continuation_heading(title: str) -> tuple[str, int, int] | None:
    match = _CONTINUATION.fullmatch(title.strip())
    if match and match[1].strip() and 1 <= int(match[2]) <= int(match[3]):
        return match[1].strip(), int(match[2]), int(match[3])
    return None


def _table_rows(item: dict) -> list[dict]:
    """Canonical rows preserve original cell text and cell-level layout metadata.

    A spanning cell appears at its start row only. We do not hallucinate repeated
    headers, expand merged cells, or claim row offsets are raw PDF byte offsets.
    """
    cells = item.get("data", {}).get("table_cells", [])
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for index, cell in enumerate(cells):
        grouped[int(cell.get("start_row_offset_idx", 0))].append((index, cell))
    rows = []
    for row_index, row in sorted(grouped.items()):
        row.sort(key=lambda pair: (pair[1].get("start_col_offset_idx", 0), pair[0]))
        pieces, offsets = [], []
        cursor = 0
        for index, cell in row:
            text = str(cell.get("text", ""))
            if pieces:
                cursor += 3  # explicit canonical separator, not PDF text
            offsets.append({"cell_index": index, "start_char": cursor, "end_char": cursor + len(text),
                            "docling_cell": cell})
            pieces.append(text)
            cursor += len(text)
        rows.append({"text": " | ".join(pieces), "row_index": row_index, "cell_spans": offsets})
    return rows


def build_sections(items: list[dict], document: dict, parse_version: str) -> tuple[list, list, list, list]:
    doc_id = document["doc_id"]
    namespace = (doc_id, document.get("source_sha256"), parse_version)
    root_id = stable_id("sec", *namespace, "document-root")
    root = {"section_id": root_id, "doc_id": doc_id, "parent_id": None,
            "title": document.get("title", doc_id), "path": [document.get("title", doc_id)],
            "level": 0, "block_ids": [], "page_start": None, "page_end": None,
            "kind": "document_root", "native_heading_refs": [], "continuation_parts": []}
    sections, blocks, figures, furniture = [root], [], [], []
    stack = [root]
    previous_heading: dict | None = None

    def add_block(item: dict, text: str, kind: str, suffix: Any = None, **extra: Any) -> dict:
        section = stack[-1]
        prov = _provenance(item)
        block = {"block_id": stable_id("blk", *namespace, item["self_ref"], suffix, text),
                 "doc_id": doc_id, "section_id": section["section_id"], "text": text,
                 "page_index": _page(item), "kind": kind, "provenance": prov,
                 "bbox": prov[0].get("bbox") if prov else None,
                 "native_ref": item["self_ref"], "native_parent_ref": item.get("parent", {}).get("$ref"),
                 "content_layer": item.get("_layer", "body"), "parse_version": parse_version,
                 "reading_order_unresolved": bool(item.get("_orphan")), **extra}
        if item.get('_source_role'):
            block['source_role'] = item['_source_role']
        if item.get('_source_repair'):
            block['source_repair'] = item['_source_repair']
        blocks.append(block)
        section["block_ids"].append(block["block_id"])
        return block

    for item in items:
        label = str(item.get("label", item.get("_collection", "unknown")))
        layer = str(item.get("_layer", "body"))
        if layer == "furniture" or label in {"page_header", "page_footer"}:
            prov = _provenance(item)
            furniture.append({"furniture_id": stable_id("fur", *namespace, item["self_ref"]),
                              "doc_id": doc_id, "native_ref": item["self_ref"],
                              "text": item.get("text", ""), "kind": label,
                              "page_index": _page(item), "provenance": prov, "native_item": item})
            continue
        text = str(item.get("text", ""))
        if label in {"title", "section_header"}:
            level = max(1, int(item.get("level", 1)))
            part = None if item.get('_preserve_heading') else continuation_heading(text)
            merge = False
            if part and previous_heading:
                prev_part = previous_heading.get("_last_part")
                merge = bool(prev_part and part[0].casefold() == prev_part[0].casefold()
                             and part[2] == prev_part[2] and part[1] == prev_part[1] + 1
                             and previous_heading["level"] == level and stack[-1] is previous_heading)
            if merge:
                section = previous_heading
            else:
                while len(stack) > 1 and stack[-1]["level"] >= level:
                    stack.pop()
                title = item.get('_combined_title') or (part[0] if part else text)
                section = {"section_id": stable_id("sec", *namespace, item["self_ref"]),
                           "doc_id": doc_id, "parent_id": stack[-1]["section_id"],
                           "title": title, "path": stack[-1]["path"] + [title],
                           "level": level, "block_ids": [], "page_start": None, "page_end": None,
                           "kind": "native_heading", "native_heading_refs": [], "continuation_parts": []}
                sections.append(section)
                stack.append(section)
            section["native_heading_refs"].append(item["self_ref"])
            section['native_heading_refs'].extend(item.get('_fragment_refs', []))
            if part:
                section["continuation_parts"].append({"part": part[1], "total": part[2],
                                                      "native_ref": item["self_ref"], "page_index": _page(item)})
            section["_last_part"] = part
            previous_heading = section
            add_block(item, text, "heading")
        elif label in {"table", "document_index"}:
            # Docling stores a table of contents in table_cells too, without
            # top-level text. Keep its rows and identify them as navigation data.
            index_metadata = ({"native_label": label, "source_role": "table_of_contents"}
                              if label == "document_index" else {})
            rows = _table_rows(item)
            for row in rows:
                add_block(item, row.pop("text"), "table_row", suffix=row["row_index"], **row,
                          table_ref=item["self_ref"], caption_refs=item.get("captions", []), **index_metadata)
            if not rows:
                add_block(item, text, label, table_ref=item["self_ref"],
                          audit_status="no_table_cells_available", native_data=item.get("data", {}), **index_metadata)
        elif label == "picture" or item.get("_collection") == "pictures":
            block = add_block(item, text, "picture", caption_refs=item.get("captions", []))
            figures.append({"figure_id": stable_id("fig", *namespace, item["self_ref"]),
                            "doc_id": doc_id, "section_id": block["section_id"], "block_id": block["block_id"],
                            "native_ref": item["self_ref"], "page_index": block["page_index"],
                            "provenance": block["provenance"], "bbox": block["bbox"],
                            "caption_refs": item.get("captions", []), "annotations": item.get("annotations", []),
                            "image": item.get("image"), "visual_status": "unreviewed"})
        else:
            add_block(item, text, label, native_item=item if not text else None)

    by_id = {s["section_id"]: s for s in sections}
    for block in blocks:
        pages = [p["page_index"] for p in block["provenance"] if p["page_index"] is not None]
        sid = block["section_id"]
        while sid:
            section = by_id[sid]
            if pages:
                section["page_start"] = min(pages + ([section["page_start"]] if section["page_start"] is not None else []))
                section["page_end"] = max(pages + ([section["page_end"]] if section["page_end"] is not None else []))
            sid = section["parent_id"]
    for section in sections:
        section.pop("_last_part", None)
    return sections, blocks, figures, furniture


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Deterministic conservative sentence boundaries, preserving exact offsets."""
    boundaries = [0]
    for match in re.finditer(r"[.!?][\"')\]]*\s+(?=[A-Z0-9\u4e00-\u9fff])|[。！？]\s*", text):
        prefix = text[max(0, match.start() - 8):match.start() + 1]
        if re.search(r"(?:\b(?:Mr|Mrs|Dr|vs|Fig|No|e\.g|i\.e)|\b[A-Z])\.$", prefix):
            continue
        boundaries.append(match.end())
    boundaries.append(len(text))
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            result.append((start, end))
    return result


def make_source_units(blocks: list[dict]) -> list[dict]:
    units = []
    for block in blocks:
        if not block["text"].strip():
            continue
        kind = "table_row" if block["kind"] == "table_row" else "sentence"
        spans = [(0, len(block["text"]))] if kind == "table_row" else sentence_spans(block["text"])
        for start, end in spans:
            units.append({"source_unit_id": stable_id("su", block["parse_version"], block["block_id"], start, end),
                          "doc_id": block["doc_id"], "section_id": block["section_id"],
                          "block_id": block["block_id"], "page_index": block["page_index"],
                          "source_start_char": start, "source_end_char": end,
                          "text": block["text"][start:end], "kind": kind,
                          "provenance": block["provenance"], "parse_version": block["parse_version"],
                          "provenance_granularity": "native_block_geometry_with_exact_canonical_text_offsets",
                          "segmentation": "conservative-sentence-v1" if kind == "sentence" else "docling-table-row-v1"})
    return units


def _coverage_text(text: str, *, canonical: bool = False) -> str:
    # Preserve case/punctuation: equality is evidence, not semantic similarity.
    normalized = unicodedata.normalize("NFKC", text)
    if canonical:
        normalized = normalized.replace(" | ", " ")  # canonical table separator only
    return "".join(normalized.split())


def _top_left_box(box: dict | None, page_height: float) -> tuple[float, float, float, float] | None:
    if not box:
        return None
    if "r_x0" in box:
        xs = [float(box[f"r_x{i}"]) for i in range(4)]
        ys = [float(box[f"r_y{i}"]) for i in range(4)]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    elif all(key in box for key in ("l", "r", "t", "b")):
        left, right = sorted((float(box["l"]), float(box["r"])))
        top, bottom = sorted((float(box["t"]), float(box["b"])))
    else:
        return None
    if str(box.get("coord_origin", "BOTTOMLEFT")) == "BOTTOMLEFT":
        top, bottom = page_height - bottom, page_height - top
    return left, top, right, bottom


def _boxes_overlap(first: tuple | None, second: tuple | None, tolerance: float = 2.0) -> bool:
    if first is None or second is None:
        return False
    return (min(first[2], second[2]) + tolerance >= max(first[0], second[0])
            and min(first[3], second[3]) + tolerance >= max(first[1], second[1]))


def recover_native_pdf_cells(fragment: dict, native_pages: list[dict]) -> dict:
    """Audit native text against assembled layout, retaining missing text verbatim.

    Recovery deliberately does NOT infer table rows/columns or normal reading
    order. Native lines are stored as flagged evidence in their closest existing
    section. Complete native page cell registries remain the audit authority.
    This is text-preservation accounting, not proof of layout/semantic accuracy.
    """
    doc = fragment["documents"][0]
    parse_version = doc["parse_version"]
    sections = {s["section_id"]: s for s in fragment["sections"]}
    source_candidates: dict[int, list[tuple[dict, dict, str]]] = defaultdict(list)
    for item in fragment["blocks"] + fragment["furniture"]:
        for prov in item.get("provenance", []):
            page_index = prov.get("page_index")
            if page_index is None and isinstance(prov.get("page_no"), int):
                page_index = prov["page_no"] - 1
            if page_index is not None:
                cells = item.get("cell_spans", []) if item.get("kind") == "table_row" else []
                located_cells = [entry["docling_cell"] for entry in cells if entry.get("docling_cell", {}).get("bbox")]
                if located_cells:
                    # A repeated word in another table row is not evidence that
                    # this native PDF occurrence survived. Prefer cell geometry.
                    for cell in located_cells:
                        source_candidates[page_index].append((item, {**prov, "bbox": cell["bbox"]},
                                                              _coverage_text(str(cell.get("text", "")))))
                else:
                    source_candidates[page_index].append((item, prov, _coverage_text(item.get("text", ""), canonical=True)))
    seen_page_nos = {page["page_no"] for page in native_pages}
    expected = set(range(1, int(doc.get("page_count", 0)) + 1))
    diagnostics, recovered = [], []
    missing_native_pages = sorted(expected - seen_page_nos)
    for page in native_pages:
        page_no = page["page_no"]
        page_index = page_no - 1
        parsed = page.get("parsed_page")
        if parsed is None:
            missing_native_pages.append(page_no)
            continue
        geometry = parsed.get("dimension", {})
        dimension_box = _top_left_box(geometry.get("rect"), 0)
        height = float(page.get("height") or (abs(dimension_box[3] - dimension_box[1]) if dimension_box else 0))
        unit_level = next((key for key in ("textline_cells", "word_cells", "char_cells") if parsed.get(key)), None)
        page_stats = {"page_no": page_no, "page_index": page_index, "audit_cell_level": unit_level,
                      "native_char_cells": len(parsed.get("char_cells", [])),
                      "native_word_cells": len(parsed.get("word_cells", [])),
                      "native_textline_cells": len(parsed.get("textline_cells", [])),
                      "meaningful_cells": 0, "matched_cells": 0, "recovered_cells": 0,
                      "invisible_or_clipping_cells": 0, "recovery_block_ids": [], "cells": []}
        for ordinal, cell in enumerate(parsed.get(unit_level, []) if unit_level else []):
            text = str(cell.get("text", ""))
            native_ref = f"#/native_pages/{page_no}/{unit_level}/{ordinal}"
            status = "nonsemantic_text_retained_in_registry"
            if cell.get("rendering_mode") in (3, 7):
                page_stats["invisible_or_clipping_cells"] += 1
                status = "invisible_or_clipping_retained_in_registry"
            elif text.strip() and (any(c.isalnum() for c in text) or any(c in "=<>+&|" for c in text)):
                page_stats["meaningful_cells"] += 1
                needle = _coverage_text(text)
                native_box = _top_left_box(cell.get("rect"), height)
                nearby = [(item, prov, candidate) for item, prov, candidate in source_candidates[page_index]
                          if _boxes_overlap(native_box, _top_left_box(prov.get("bbox"), height))]
                exact = [(item, prov) for item, prov, candidate in nearby if needle and needle in candidate]
                if exact:
                    page_stats["matched_cells"] += 1
                    status = "text_and_geometry_matched"
                else:
                    # Text may span multiple already-assembled items. Joining in
                    # original block order is conservative; no fuzzy matching.
                    joined = "".join(candidate for _, _, candidate in nearby)
                    if needle and needle in joined:
                        page_stats["matched_cells"] += 1
                        status = "text_and_geometry_matched_across_items"
                    else:
                        # Prefer an overlapping table/body block; otherwise an
                        # existing page section. Scope is a hint, not a row claim.
                        scoped = [item for item, _, _ in nearby if item.get("section_id") in sections]
                        if not scoped:
                            scoped = [item for item, _, _ in source_candidates[page_index]
                                      if item.get("section_id") in sections]
                        section = sections[scoped[0]["section_id"]] if scoped else fragment["sections"][0]
                        box = None if native_box is None else dict(zip(("l", "t", "r", "b"), native_box))
                        if box is not None:
                            box["coord_origin"] = "TOPLEFT"
                        provenance = [{"page_no": page_no, "page_index": page_index, "bbox": box,
                                       "charspan": [0, len(text)], "native_cell_ref": native_ref}]
                        block = {"block_id": stable_id("blk", doc["doc_id"], doc.get("source_sha256"),
                                                       parse_version, native_ref, text),
                                 "doc_id": doc["doc_id"], "section_id": section["section_id"], "text": text,
                                 "page_index": page_index, "kind": "native_pdf_text_recovery",
                                 "provenance": provenance, "bbox": box, "native_ref": native_ref,
                                 "native_parent_ref": None, "content_layer": "body", "parse_version": parse_version,
                                 "reading_order_unresolved": True,
                                 "recovery_reason": "native_text_not_matched_in_assembled_layout",
                                 "native_cell_level": unit_level, "native_cell_index": cell.get("index", ordinal),
                                 "structure_status": "unreviewed_no_table_relationship_inferred"}
                        recovered.append(block)
                        section["block_ids"].append(block["block_id"])
                        sid = section["section_id"]
                        while sid:
                            parent = sections[sid]
                            parent["page_start"] = min(page_index, parent["page_start"]) if parent["page_start"] is not None else page_index
                            parent["page_end"] = max(page_index, parent["page_end"]) if parent["page_end"] is not None else page_index
                            sid = parent["parent_id"]
                        page_stats["recovered_cells"] += 1
                        page_stats["recovery_block_ids"].append(block["block_id"])
                        status = "recovered_verbatim_requires_structure_review"
            page_stats["cells"].append({"native_cell_ref": native_ref, "status": status})
        if not unit_level:
            page_stats["status"] = "no_native_text_requires_visual_or_ocr_review"
        diagnostics.append(page_stats)
    fragment["blocks"].extend(recovered)
    fragment["source_units"].extend(make_source_units(recovered))
    coverage = {"method": "native-line-text-and-overlapping-geometry-v1",
                "scope": "native PDF text only; not OCR/visual completeness or semantic table validation",
                "pages": diagnostics, "missing_parsed_pages": missing_native_pages,
                "meaningful_cells": sum(p["meaningful_cells"] for p in diagnostics),
                "matched_cells": sum(p["matched_cells"] for p in diagnostics),
                "recovered_cells": len(recovered), "requires_structure_review": bool(recovered),
                "status": "incomplete_native_page_registry" if missing_native_pages else
                          "text_preserved_with_flagged_recovery" if recovered else "native_text_matched"}
    fragment["audit"]["native_text_coverage"] = coverage
    fragment["audit"]["blocks"] = len(fragment["blocks"])
    fragment["audit"]["source_units"] = len(fragment["source_units"])
    if recovered:
        fragment["audit"]["warnings"].append(f"native_text_recovered_structure_review_required:{len(recovered)}")
    if missing_native_pages:
        fragment["audit"]["warnings"].append(f"native_parsed_pages_missing:{missing_native_pages}")
    return coverage
