"""Auditable, fail-closed alignment to the frozen legacy extraction.

Only the full legacy retrieval corpus is read: gold QA/atoms never influence
which new blocks align. Whitespace changes are allowed; fuzzy textual credit,
case folding, inferred missing text and arbitrary duplicate selection are not.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S|\s+", text):
        token = match.group()
        if token.isspace():
            if not characters or match.end() == len(text):
                continue
            token = " "
        characters.append(token)
        offsets.append((match.start(), match.end()))
    return "".join(characters), offsets


def _reconstruct(path: Path) -> list[dict[str, Any]]:
    """Reconstruct observed source characters, retaining gaps as separate runs."""
    blocks: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = str(chunk.get("text", ""))
            for span in chunk.get("source_spans", ()):
                required = {"block_id", "source_start_char", "source_end_char", "chunk_start_char", "chunk_end_char"}
                if not required <= span.keys():
                    raise ValueError(f"legacy chunk line {line_number} lacks reconstruction offsets")
                a, b = int(span["source_start_char"]), int(span["source_end_char"])
                c, d = int(span["chunk_start_char"]), int(span["chunk_end_char"])
                if a < 0 or b <= a or c < 0 or d > len(text) or d <= c or b - a != d - c:
                    raise ValueError(f"invalid legacy source span on line {line_number}")
                key = (str(chunk["doc_id"]), str(span["block_id"]))
                block = blocks.setdefault(key, {"doc_id": key[0], "block_id": key[1],
                                                "page_index": span.get("page_index"), "characters": {}})
                if span.get("page_index") != block["page_index"]:
                    raise ValueError("legacy block has inconsistent page provenance")
                for offset, character in enumerate(text[c:d], a):
                    previous = block["characters"].get(offset)
                    if previous is not None and previous != character:
                        raise ValueError(f"conflicting legacy character at {key[1]}:{offset}")
                    block["characters"][offset] = character
    runs: list[dict[str, Any]] = []
    for block in blocks.values():
        coordinates = sorted(block.pop("characters").items())
        if not coordinates:
            continue
        start = previous = coordinates[0][0]
        text = [coordinates[0][1]]
        for offset, character in coordinates[1:]:
            if offset != previous + 1:
                runs.append(dict(block, start_char=start, text="".join(text)))
                start, text = offset, []
            text.append(character)
            previous = offset
        runs.append(dict(block, start_char=start, text="".join(text)))
    for run in runs:
        run["normalized_text"], run["normalized_offsets"] = _normalize(run["text"])
    return runs


def _document_map(corpus: Mapping[str, Any], legacy_doc_ids: set[str]) -> dict[str, str | None]:
    result = {}
    for document in corpus.get("documents", ()):
        doc_id = str(document["doc_id"])
        digest = str(document.get("source_sha256") or "")
        candidates = [value for value in legacy_doc_ids
                      if value == doc_id or (len(digest) == 64 and value.endswith(":" + digest[:12]))]
        result[doc_id] = candidates[0] if len(candidates) == 1 else None
    return result


def align_corpus(corpus: Mapping[str, Any], legacy_chunks_path: str | Path) -> dict[str, Any]:
    """Return per-block alignment records and coverage diagnostics; never write."""
    path = Path(legacy_chunks_path)
    legacy_runs = _reconstruct(path)
    documents = _document_map(corpus, {run["doc_id"] for run in legacy_runs})
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in legacy_runs:
        by_doc[run["doc_id"]].append(run)
    mappings: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()
    for block in corpus.get("blocks", ()):
        block_id, doc_id = str(block["block_id"]), str(block["doc_id"])
        if block_id in seen_blocks:
            raise ValueError(f"duplicate new block ID: {block_id}")
        seen_blocks.add(block_id)
        text = str(block.get("text", ""))
        normalized, new_offsets = _normalize(text)
        mapping: dict[str, Any] = {
            "new_block_id": block_id, "doc_id": doc_id,
            "page_index": block.get("page_index"), "new_text_sha256": _sha(text),
            "new_char_count": len(text), "status": "unmapped", "segments": [],
        }
        if not normalized:
            mapping["reason"] = "empty_text"
            mappings.append(mapping)
            continue
        legacy_doc = documents.get(doc_id)
        if legacy_doc is None:
            mapping["reason"] = "unresolved_document_identity"
            mappings.append(mapping)
            continue
        candidates = []
        for run in by_doc[legacy_doc]:
            if block.get("page_index") is not None and run["page_index"] != block["page_index"]:
                continue
            start = run["normalized_text"].find(normalized)
            while start >= 0:
                candidates.append((run, start))
                start = run["normalized_text"].find(normalized, start + 1)
        if len(candidates) != 1:
            mapping.update(status="ambiguous" if candidates else "unmapped",
                           reason="multiple_exact_occurrences" if candidates else "no_exact_or_whitespace_match",
                           candidate_count=len(candidates))
            mappings.append(mapping)
            continue
        run, normalized_start = candidates[0]
        old_offsets = run["normalized_offsets"][normalized_start:normalized_start + len(normalized)]
        segments: list[dict[str, Any]] = []
        for (new_start, new_end), (old_start, old_end) in zip(new_offsets, old_offsets):
            same = text[new_start:new_end] == run["text"][old_start:old_end]
            segment = {"new_start_char": new_start, "new_end_char": new_end,
                       "legacy_start_char": run["start_char"] + old_start,
                       "legacy_end_char": run["start_char"] + old_end,
                       "method": "exact" if same else "whitespace_normalized"}
            if (segments and same and segments[-1]["method"] == "exact"
                    and segments[-1]["new_end_char"] == new_start
                    and segments[-1]["legacy_end_char"] == segment["legacy_start_char"]):
                segments[-1]["new_end_char"] = new_end
                segments[-1]["legacy_end_char"] = segment["legacy_end_char"]
            else:
                segments.append(segment)
        method = "exact" if all(piece["method"] == "exact" for piece in segments) else "whitespace_normalized"
        mapping.update(status="aligned", method=method, legacy_doc_id=legacy_doc,
                       legacy_block_id=run["block_id"], legacy_text_sha256=_sha(run["text"]),
                       legacy_run_start_char=run["start_char"], segments=segments,
                       covered_new_chars=sum(piece["new_end_char"] - piece["new_start_char"] for piece in segments))
        mappings.append(mapping)
    total_chars = sum(item["new_char_count"] for item in mappings)
    covered_chars = sum(item.get("covered_new_chars", 0) for item in mappings)
    counts = {status: sum(item["status"] == status for item in mappings)
              for status in ("aligned", "ambiguous", "unmapped")}
    return {"mappings": mappings, "report": {
        "alignment_version": "doc2skill-legacy-alignment-v1",
        "legacy_chunks_path": str(path.resolve()),
        "legacy_chunks_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "legacy_block_count": len({(run["doc_id"], run["block_id"]) for run in legacy_runs}),
        "block_count": len(mappings), **{f"{key}_blocks": value for key, value in counts.items()},
        "total_new_chars": total_chars, "covered_new_chars": covered_chars,
        "character_coverage": covered_chars / total_chars if total_chars else 0.0,
        "gold_annotations_used": False,
        "policy": "Exact or whitespace-normalized, page-restricted, unique occurrence only; unresolved text receives no credit.",
    }}


def apply_alignment(items: Sequence[Mapping[str, Any]], mappings: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project only returned new-source intervals into frozen legacy coordinates."""
    if isinstance(mappings, Mapping):
        mappings = mappings["mappings"]
    by_block = {str(item["new_block_id"]): item for item in mappings}
    if len(by_block) != len(mappings):
        raise ValueError("duplicate alignment records")
    results = []
    for item in items:
        result = dict(item)
        original = list(item.get("source_spans") or item.get("metadata", {}).get("source_spans") or ())
        result["doc2skill_source_spans"] = original
        result["doc2skill_doc_id"] = item.get("doc_id")
        projected: list[dict[str, Any]] = []
        unresolved = []
        legacy_docs: set[str] = set()
        for span in original:
            identifier = str(span.get("block_id", ""))
            mapping = by_block.get(identifier)
            if mapping is None or mapping.get("status") != "aligned":
                unresolved.append(identifier)
                continue
            if str(item.get("doc_id")) != str(mapping["doc_id"]):
                raise ValueError("retrieval item document does not match its alignment")
            start = int(span.get("source_start_char", span.get("start_char", -1)))
            end = int(span.get("source_end_char", span.get("end_char", -1)))
            if start < 0 or end <= start or end > int(mapping["new_char_count"]):
                raise ValueError("returned source span is outside its aligned block")
            legacy_docs.add(str(mapping["legacy_doc_id"]))
            for piece in mapping["segments"]:
                left, right = max(start, piece["new_start_char"]), min(end, piece["new_end_char"])
                if right <= left:
                    continue
                if piece["method"] == "exact":
                    a = piece["legacy_start_char"] + left - piece["new_start_char"]
                    b = a + right - left
                elif left == piece["new_start_char"] and right == piece["new_end_char"]:
                    a, b = piece["legacy_start_char"], piece["legacy_end_char"]
                else:
                    # A partial normalized whitespace run cannot justify its full old span.
                    continue
                projected.append({"block_id": mapping["legacy_block_id"],
                                  "start_char": a, "end_char": b, "coordinate_space": "block_text"})
        if len(legacy_docs) > 1:
            raise ValueError("one retrieved item cannot map across different documents")
        if legacy_docs:
            result["doc_id"] = next(iter(legacy_docs))
        # Merge only actually covered adjacent/overlapping intervals.
        merged: list[dict[str, Any]] = []
        for span in sorted(projected, key=lambda value: (value["block_id"], value["start_char"], value["end_char"])):
            if merged and merged[-1]["block_id"] == span["block_id"] and span["start_char"] <= merged[-1]["end_char"]:
                merged[-1]["end_char"] = max(merged[-1]["end_char"], span["end_char"])
            else:
                merged.append(dict(span))
        result["source_spans"] = merged
        result.pop("evidence_ids", None)
        result.pop("equivalence_groups", None)
        result["alignment"] = {"status": "aligned" if merged and not unresolved else "partial" if merged else "unmapped",
                               "unresolved_block_ids": sorted(set(unresolved)), "projected_span_count": len(merged)}
        results.append(result)
    return results
