"""Optimizer-only, exact gold-to-current-source projections.

This is NOT the gold-independent production alignment. Gold lookup stays offline
and cannot be handed to ordinary navigation, retrieval or answer generation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .data import digest, read_jsonl


def normalize_with_offsets(text):
    characters, offsets = [], []
    for match in re.finditer(r"\S|\s+", text):
        token = match.group()
        if token.isspace():
            if not characters or match.end() == len(text):
                continue
            token = " "
        characters.append(token)
        offsets.append((match.start(), match.end()))
    return "".join(characters), offsets


def load_corpus(directory):
    """Read corpus JSONL or the serving database without vectors/model loading."""
    root = Path(directory).resolve()
    kinds = ("documents", "sections", "blocks", "source_units", "chunks")
    if all((root / (kind + ".jsonl")).is_file() for kind in kinds):
        return {kind: read_jsonl(root / (kind + ".jsonl")) for kind in kinds}
    with sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
        return {kind: [json.loads(row[0]) for row in connection.execute(
            f"SELECT payload FROM {kind} ORDER BY ordinal")] for kind in kinds}


def project_chunk_slice(chunk, start, end):
    """Project only exposed characters; unanchored non-whitespace is refused."""
    text = chunk.get("text", "")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Excerpt offsets must lie within the original chunk")
    projected, covered = [], set()
    for span in chunk.get("source_spans", []):
        a, b = span.get("chunk_start_char"), span.get("chunk_end_char")
        x, y = span.get("source_start_char"), span.get("source_end_char")
        if (any(type(v) is not int for v in (a, b, x, y)) or not 0 <= a < b <= len(text)
                or x < 0 or y - x != b - a or not span.get("block_id")
                or not ((type(span.get("page_index")) is int and span["page_index"] >= 0)
                        or (span.get("page_index") is None and span.get("source_type") == "html"))):
            raise ValueError("Chunk source spans have invalid provenance offsets")
        left, right = max(start, a), min(end, b)
        if left < right:
            projected.append({"block_id": span["block_id"], "start_char": x + left - a,
                              "end_char": x + right - a, "page_index": span["page_index"],
                              "coordinate_space": "block_text", "text": text[left:right]})
            covered.update(range(left, right))
    if any(not text[i].isspace() and i not in covered for i in range(start, end)):
        raise ValueError("Excerpt includes non-whitespace without current source provenance")
    if not projected:
        raise ValueError("Excerpt has no current source provenance")
    return projected


def _span_signature(doc_id, spans):
    # A duplicated retrieval chunk is not a second physical source occurrence.
    return (doc_id, tuple((s["block_id"], s["start_char"], s["end_char"], s["page_index"])
                          for s in spans))


def build_oracle(qa_records, corpus, *, legacy_documents=None, source_atoms=()):
    """Return an auditable exact/whitespace gold support sidecar; never write.

    PDF identity is verified against full old/current SHA256 values. Missing
    identity, changed wording/case, unanchored text and multiple occurrences do
    not become usable targets. Adjacent source units may match only where an
    existing canonical chunk explicitly supplies their text/order/provenance.
    Declared source-atom equivalence alternatives are allowed, never invented.
    """
    qa = deepcopy(list(qa_records))
    current_documents = {d["doc_id"]: d for d in corpus.get("documents", [])}
    legacy_documents = legacy_documents if legacy_documents is not None else corpus.get("legacy_documents", [])
    old_documents = {d["doc_id"]: d for d in legacy_documents}
    chunks = deepcopy(list(corpus.get("chunks", [])))
    blocks = {b["block_id"]: b for b in corpus.get("blocks", [])}
    if len(blocks) != len(corpus.get("blocks", [])) or len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate current block/chunk IDs")
    for chunk in chunks:
        chunk["_normalized"], chunk["_offsets"] = normalize_with_offsets(chunk["text"])
        # Verify canonical chunk-to-block text, including the native page.
        for span in chunk.get("source_spans", []):
            block = blocks.get(span.get("block_id"))
            if block is None or block.get("doc_id") != chunk.get("doc_id") or block.get("page_index") != span.get("page_index"):
                raise ValueError("Chunk references an unknown or mismatched current source block")
            a, b = span["chunk_start_char"], span["chunk_end_char"]
            x, y = span["source_start_char"], span["source_end_char"]
            if chunk["text"][a:b] != block["text"][x:y]:
                raise ValueError("Chunk text disagrees with current source block")
    equivalents = defaultdict(list)
    atoms = list(source_atoms) or list(corpus.get("source_atoms", []))
    for atom in atoms:
        if atom.get("equivalence_group"):
            equivalents[atom["equivalence_group"]].append(atom)
    cache = {}

    def locate(anchor):
        key = digest({k: anchor.get(k) for k in ("doc_id", "page_index", "quote", "document_sha256")})
        if key in cache:
            return deepcopy(cache[key])
        doc = anchor.get("doc_id")
        expected = old_documents.get(doc, {}).get("source_sha256") or anchor.get("document_sha256")
        actual = current_documents.get(doc, {}).get("source_sha256")
        if not expected or not re.fullmatch(r"[a-f0-9]{64}", str(expected)) or actual != expected:
            return {"status": "unknown", "reason": "unverified_or_mismatched_pdf_identity", "candidates": []}
        if type(anchor.get("page_index")) is not int or anchor["page_index"] < 0:
            return {"status": "unknown", "reason": "invalid_gold_page", "candidates": []}
        needle, _ = normalize_with_offsets(str(anchor.get("quote", "")))
        if not needle:
            return {"status": "unknown", "reason": "empty_gold_quote", "candidates": []}
        hits = {}
        for chunk in chunks:
            if chunk["doc_id"] != doc:
                continue
            position = chunk["_normalized"].find(needle)
            while position >= 0:
                start = chunk["_offsets"][position][0]
                end = chunk["_offsets"][position + len(needle) - 1][1]
                try:
                    spans = project_chunk_slice(chunk, start, end)
                except ValueError:
                    spans = []
                if spans and all(s["page_index"] == anchor["page_index"] for s in spans):
                    signature = _span_signature(doc, spans)
                    candidate = hits.setdefault(signature, {"doc_id": doc, "document_sha256": actual,
                        "page_index": anchor["page_index"], "source_spans": spans,
                        "section_ids": [], "chunk_ids": [], "chunk_excerpts": [],
                        "match_method": "exact" if chunk["text"][start:end] == anchor["quote"] else "whitespace_normalized"})
                    if chunk.get("section_id") not in candidate["section_ids"]:
                        candidate["section_ids"].append(chunk.get("section_id"))
                    candidate["chunk_ids"].append(chunk["chunk_id"])
                    candidate["chunk_excerpts"].append({"chunk_id": chunk["chunk_id"], "start_char": start,
                        "end_char": end, "text": chunk["text"][start:end],
                        "text_sha256": hashlib.sha256(chunk["text"][start:end].encode()).hexdigest()})
                position = chunk["_normalized"].find(needle, position + 1)
        candidates = sorted(hits.values(), key=digest)
        result = {"status": "mapped" if len(candidates) == 1 else "ambiguous" if candidates else "unknown",
                  "reason": "unique_exact_current_source" if len(candidates) == 1 else
                            "multiple_current_source_occurrences" if candidates else "no_exact_same_pdf_page_match",
                  "candidate_count": len(candidates), "candidates": candidates}
        cache[key] = result
        return deepcopy(result)

    queries = {}
    for row in qa:
        qid = row["qid"]
        if qid in queries:
            raise ValueError("Duplicate QA qid")
        evidence = []
        for gold in row.get("evidence", []):
            direct = locate(gold)
            alternates, targets = [], []
            if direct["status"] == "mapped":
                targets.extend(direct["candidates"])
            for atom in equivalents.get(gold.get("equivalence_group"), []):
                if atom.get("source_atom_id") == gold.get("source_atom_id"):
                    continue
                located = locate(atom)
                alternates.append({"source_atom_id": atom.get("source_atom_id"), **located})
                if located["status"] == "mapped":
                    targets.extend(located["candidates"])
            unique_targets = {digest(t["source_spans"] + [{"doc_id": t["doc_id"]}]): t for t in targets}
            evidence.append({"evidence_id": gold["evidence_id"], "source_atom_id": gold.get("source_atom_id"),
                             "equivalence_group": gold.get("equivalence_group"),
                             "gold": {k: deepcopy(gold.get(k)) for k in ("doc_id", "block_id", "page_index", "quote", "source_span")},
                             "status": "mapped" if targets else direct["status"],
                             "direct": direct, "registered_alternatives": alternates,
                             "targets": list(unique_targets.values())})
        by_evidence = {e["evidence_id"]: e for e in evidence}
        routes = []
        for route in row.get("gold_evidence_sets", []):
            required = []
            for eid in route.get("required_evidence_ids", []):
                if eid not in by_evidence:
                    raise ValueError("Gold route references unknown evidence")
                evidence_key = by_evidence[eid].get("equivalence_group") or eid
                if evidence_key not in required:
                    required.append(evidence_key)
            for group in route.get("required_equivalence_groups", []):
                if group not in required:
                    required.append(group)
            routes.append({"set_id": route.get("set_id"), "required_keys": required})
        if not routes and evidence:
            routes = [{"set_id": "all_gold", "required_keys": sorted({e.get("equivalence_group") or e["evidence_id"] for e in evidence})}]
        available = {e.get("equivalence_group") or e["evidence_id"] for e in evidence if e["targets"]}
        complete = any(route["required_keys"] and set(route["required_keys"]) <= available for route in routes)
        queries[qid] = {"qid": qid, "status": "mapped" if complete else "partial" if available else "unknown",
                        "complete_route_available": complete, "evidence": evidence, "routes": routes}
    counts = Counter(e["status"] for q in queries.values() for e in q["evidence"])
    result = {"schema_version": "qa-skillopt-exact-oracle-v1", "optimizer_only": True,
              "policy": "Verified identical PDF bytes/page; unique exact or whitespace-only current-source occurrence; registered equivalents only; never fuzzy.",
              "qa_sha256": digest(sorted(qa, key=lambda q: q["qid"])),
              "corpus_sha256": digest({kind: corpus.get(kind, []) for kind in ("documents", "blocks", "sections", "chunks")}),
              "queries": queries, "summary": {"query_count": len(queries), "evidence_status_counts": dict(counts),
                  "complete_query_count": sum(q["complete_route_available"] for q in queries.values())}}
    result["oracle_sha256"] = digest(result)
    return result
