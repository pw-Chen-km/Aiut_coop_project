"""Stable sentence/semantic source atoms and derived chunk mappings.

Atoms are anchored to immutable document/page/block offsets.  Retrieval chunks
are deliberately absent from atom IDs, so a new preprocessing granularity only
requires rebuilding the many-to-many mapping in :func:`map_atoms_to_chunks`.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?P<closing>[\"'’”)]{0,2})(?P<space>\s+)(?=[A-Z0-9\[])"
)
IMPERATIVE_RE = re.compile(
    r"^(?:click|press|select|choose|open|switch|verify|remove|save|apply|wait|set|add|enter|scan|turn)\b",
    re.I,
)


def normalize_atom_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_atom_id(
    doc_id: str,
    block_id: str,
    start_char: int,
    end_char: int,
    quote: str,
    *,
    atom_type: str = "sentence",
    visual_identity: str = "",
) -> str:
    key = "\x1f".join(
        (
            doc_id,
            block_id,
            str(start_char),
            str(end_char),
            normalize_atom_text(quote),
            atom_type,
            visual_identity,
        )
    )
    return f"sa:{_sha256(key)[:24]}"


def atom_equivalence_group(quote: str, atom_type: str = "sentence") -> str:
    return f"eg:atom-{_sha256(atom_type + chr(31) + normalize_atom_text(quote))[:16]}"


def sentence_segments(text: str) -> list[tuple[int, int]]:
    """Return trimmed, exact-offset sentence-like spans.

    Line wrapping is not a boundary.  Punctuation followed by a new sentence is
    a boundary; a blank paragraph is used only when both sides contain text.
    This is conservative because an over-split rule is worse than retaining a
    short compound semantic atom.
    """

    if not text:
        return []
    boundaries = {0, len(text)}
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        boundaries.add(match.start("space"))
        boundaries.add(match.end("space"))
    for match in re.finditer(r"\n[ \t]*\n+", text):
        left = text[: match.start()].rstrip()
        right = text[match.end() :].lstrip()
        if left and right:
            boundaries.add(match.start())
            boundaries.add(match.end())
    ordered = sorted(boundaries)
    raw: list[tuple[int, int]] = []
    for start, end in zip(ordered, ordered[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            raw.append((start, end))
    # Whitespace boundaries can produce punctuation-only fragments; merge them.
    result: list[tuple[int, int]] = []
    for start, end in raw:
        segment = text[start:end]
        if result and (len(normalize_atom_text(segment)) < 12 or not re.search(r"[A-Za-z0-9]", segment)):
            previous_start, _ = result[-1]
            result[-1] = (previous_start, end)
        else:
            result.append((start, end))
    return result or [(0, len(text))]


def infer_atom_type(evidence: Mapping[str, Any], chunk: Mapping[str, Any] | None) -> str:
    """Infer a source-semantic type without depending on retrieval chunking."""

    if evidence.get("role") == "visual" or evidence.get("visual_ref"):
        return "visual_region"
    quote = str(evidence.get("quote", "")).strip()
    if IMPERATIVE_RE.match(quote):
        return "procedure_step"
    return "sentence"


def atomize_evidence(
    evidence: Mapping[str, Any], *, document_sha256: str,
    chunk: Mapping[str, Any] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Split one QA anchor into exact source atoms and p2 evidence templates.

    Returns ``(atom, evidence_template)`` pairs.  The caller assigns question-
    local evidence IDs; the stable ``source_atom_id`` is already populated.
    """

    quote = str(evidence.get("quote", ""))
    if not quote:
        raise ValueError("source evidence requires a quote")
    source_span = evidence.get("source_span")
    chunk_span = evidence.get("span")
    if not isinstance(source_span, Mapping) or not isinstance(chunk_span, Mapping):
        raise ValueError("source evidence requires block and chunk spans")
    if evidence.get("role") == "visual":
        segments = [(0, len(quote))]
    else:
        segments = sentence_segments(quote)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for relative_start, relative_end in segments:
        segment = quote[relative_start:relative_end]
        source_start = int(source_span["start_char"]) + relative_start
        source_end = int(source_span["start_char"]) + relative_end
        chunk_start = int(chunk_span["start_char"]) + relative_start
        chunk_end = int(chunk_span["start_char"]) + relative_end
        atom_type = infer_atom_type(evidence, chunk)
        visual_identity = ""
        if atom_type == "visual_region":
            visual_identity = "|".join(
                (
                    str(evidence.get("figure_id", "")),
                    ",".join(str(value) for value in evidence.get("bbox", ())),
                )
            )
        atom_id = source_atom_id(
            str(evidence["doc_id"]), str(evidence["block_id"]), source_start,
            source_end, segment, atom_type=atom_type, visual_identity=visual_identity,
        )
        atom: dict[str, Any] = {
            "source_atom_id": atom_id,
            "atom_schema_version": "sav1",
            "doc_id": evidence["doc_id"],
            "document_sha256": document_sha256,
            "page_index": evidence["page_index"],
            "page_label": evidence["page_label"],
            "block_id": evidence["block_id"],
            "source_span": {
                "coordinate_space": "block_text",
                "start_char": source_start,
                "end_char": source_end,
            },
            "quote": segment,
            "quote_sha256": _sha256(segment),
            "atom_type": atom_type,
            "equivalence_group": atom_equivalence_group(segment, atom_type),
            "canonical_atom_id": atom_id,
        }
        for field in ("visual_ref", "figure_id", "bbox", "ocr_text"):
            if field in evidence:
                atom[field] = evidence[field]
        template = dict(evidence)
        template.update(
            {
                "source_atom_id": atom_id,
                "source_span": dict(atom["source_span"]),
                "span": {
                    "coordinate_space": "chunk_text",
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                },
                "quote": segment,
                "equivalence_group": atom["equivalence_group"],
            }
        )
        pairs.append((atom, template))
    return pairs


def canonicalize_atoms(atoms: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for value in atoms:
        item = dict(value)
        atom_id = str(item["source_atom_id"])
        previous = unique.get(atom_id)
        if previous is not None and previous != item:
            raise ValueError(f"source atom collision: {atom_id}")
        unique[atom_id] = item
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unique.values():
        groups[str(item["equivalence_group"])].append(item)
    for members in groups.values():
        canonical = min(
            members,
            key=lambda item: (
                "map-user-guide-kanbans" not in str(item["doc_id"]),
                str(item["doc_id"]), int(item["page_index"]),
                int(item["source_span"]["start_char"]),
            ),
        )
        for item in members:
            item["canonical_atom_id"] = canonical["source_atom_id"]
    return sorted(
        unique.values(),
        key=lambda item: (
            str(item["doc_id"]), int(item["page_index"]), str(item["block_id"]),
            int(item["source_span"]["start_char"]), str(item["source_atom_id"]),
        ),
    )


def map_atoms_to_chunks(
    atoms: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a derived many-to-many mapping using block-coordinate overlap."""

    chunks_by_doc_block: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for chunk in chunks:
        for span in chunk.get("source_spans", []):
            chunks_by_doc_block[(str(chunk["doc_id"]), str(span["block_id"]))].append((chunk, span))
    mappings: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for atom in atoms:
        atom_start = int(atom["source_span"]["start_char"])
        atom_end = int(atom["source_span"]["end_char"])
        atom_length = atom_end - atom_start
        found = False
        for chunk, chunk_span in chunks_by_doc_block.get((str(atom["doc_id"]), str(atom["block_id"])), []):
            overlap_start = max(atom_start, int(chunk_span["source_start_char"]))
            overlap_end = min(atom_end, int(chunk_span["source_end_char"]))
            if overlap_end <= overlap_start:
                continue
            found = True
            overlap = overlap_end - overlap_start
            coverage = overlap / atom_length
            mappings.append(
                {
                    "mapping_schema_version": "acmv1",
                    "source_atom_id": atom["source_atom_id"],
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": atom["doc_id"],
                    "block_id": atom["block_id"],
                    "overlap_source_span": {
                        "coordinate_space": "block_text",
                        "start_char": overlap_start,
                        "end_char": overlap_end,
                    },
                    "overlap_chars": overlap,
                    "coverage": coverage,
                    "mapping_kind": "contains" if coverage == 1 else "partial",
                }
            )
        if not found:
            unmapped.append(str(atom["source_atom_id"]))
    if unmapped:
        raise ValueError(f"source atoms have no chunk mapping: {unmapped[:20]}")
    return sorted(mappings, key=lambda item: (str(item["source_atom_id"]), str(item["chunk_id"])))


def atom_union_coverage(
    atom: Mapping[str, Any], retrieved_chunk_ids: set[str],
    mappings: Sequence[Mapping[str, Any]],
) -> float:
    """Coverage of one atom by the union of retrieved current-chunk mappings."""

    atom_start = int(atom["source_span"]["start_char"])
    atom_end = int(atom["source_span"]["end_char"])
    intervals: list[tuple[int, int]] = []
    for item in mappings:
        if (
            item.get("source_atom_id") != atom.get("source_atom_id")
            or item.get("chunk_id") not in retrieved_chunk_ids
        ):
            continue
        span = item.get("overlap_source_span")
        if not isinstance(span, Mapping):
            raise ValueError("atom-chunk mappings require overlap_source_span")
        start = max(atom_start, int(span["start_char"]))
        end = min(atom_end, int(span["end_char"]))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    covered = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered / (atom_end - atom_start)


__all__ = [
    "atom_equivalence_group", "atom_union_coverage", "atomize_evidence",
    "canonicalize_atoms", "map_atoms_to_chunks", "normalize_atom_text",
    "sentence_segments", "source_atom_id",
]
