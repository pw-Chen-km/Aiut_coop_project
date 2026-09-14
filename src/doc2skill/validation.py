"""Corpus integrity, source coverage and artifact checks independent of QA."""
from __future__ import annotations

import json
from pathlib import Path

from .config import sha256


def validate_corpus(corpus, tokenizer=None, max_tokens=384):
    errors, warnings = [], []
    documents = {r["doc_id"]: r for r in corpus["documents"]}
    sections = {r["section_id"]: r for r in corpus["sections"]}
    blocks = {r["block_id"]: r for r in corpus["blocks"]}
    for kind, field in (("documents", "doc_id"), ("sections", "section_id"), ("blocks", "block_id"),
                        ("source_units", "source_unit_id"), ("chunks", "chunk_id")):
        if len({r[field] for r in corpus[kind]}) != len(corpus[kind]):
            errors.append(f"Duplicate {kind} IDs")
    for section in sections.values():
        if section["doc_id"] not in documents:
            errors.append(f"Unknown document for {section['section_id']}")
        visited = set()
        node = section
        while node.get("parent_id"):
            identity = node["parent_id"]
            if identity in visited or identity not in sections:
                errors.append(f"Invalid/cyclic parent for {section['section_id']}")
                break
            visited.add(identity)
            node = sections[identity]
            if node["doc_id"] != section["doc_id"]:
                errors.append("Cross-document parent")
                break
    intervals = {b: [] for b in blocks}
    for chunk in corpus["chunks"]:
        section = sections.get(chunk["section_id"])
        if not section or section["doc_id"] != chunk["doc_id"]:
            errors.append(f"Invalid chunk section: {chunk['chunk_id']}")
        if chunk["token_count"] > max_tokens:
            errors.append(f"Chunk exceeds configured token budget: {chunk['chunk_id']}")
        if tokenizer and len(tokenizer.encode(chunk["embedding_text"], add_special_tokens=True)) > max_tokens:
            errors.append(f"Actual embedding input exceeds budget: {chunk['chunk_id']}")
        for span in chunk["source_spans"]:
            block = blocks.get(span["block_id"])
            if not block:
                errors.append("Unknown block in chunk")
                continue
            if block.get("build_role", "content") != "content":
                errors.append("Chunk includes excluded or navigation-only content")
            a, b = span["source_start_char"], span["source_end_char"]
            x, y = span["chunk_start_char"], span["chunk_end_char"]
            if not (0 <= a < b <= len(block["text"]) and 0 <= x < y <= len(chunk["text"])):
                errors.append(f"Invalid coordinates: {chunk['chunk_id']}")
            elif block["text"][a:b] != chunk["text"][x:y]:
                errors.append(f"Non-exact source span: {chunk['chunk_id']}")
            else:
                intervals[block["block_id"]].append((a, b))
            if block["doc_id"] != chunk["doc_id"] or block["section_id"] != chunk["section_id"]:
                errors.append("Chunk crosses unrelated document/section")
    uncovered = []
    for identity, block in blocks.items():
        text = block.get("text", "")
        role = block.get("build_role", "content")
        if role not in {"content", "noise", "navigation_only"}:
            errors.append("Unknown build role: " + identity)
        if role in {"noise", "navigation_only"}:
            continue
        if not text.strip():
            continue
        covered = bytearray(len(text))
        for a, b in intervals[identity]:
            covered[a:b] = b"\1" * (b - a)
        if any(not flag and not char.isspace() for char, flag in zip(text, covered)):
            uncovered.append(identity)
    if uncovered:
        errors.append(f"{len(uncovered)} source blocks have unindexed non-whitespace content")
    recovered_count = sum(b.get("kind") == "native_pdf_text_recovery" for b in blocks.values())
    if recovered_count:
        warnings.append(f"{recovered_count} verbatim native-text recovery blocks require structure review; no table semantics inferred")
    for unit in corpus["source_units"]:
        block = blocks.get(unit.get("block_id"))
        if block is None:
            errors.append("Source unit refers to missing block")
            continue
        a, b = unit.get("source_start_char", -1), unit.get("source_end_char", -1)
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(block["text"])):
            errors.append("Source unit has invalid source offsets")
        elif unit.get("text") != block["text"][a:b]:
            errors.append("Source unit text does not match canonical offsets")
        if unit.get("doc_id") != block["doc_id"] or unit.get("section_id") != block["section_id"]:
            errors.append("Source unit document/section differs from canonical block")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "counts": {k: len(v) for k, v in corpus.items() if isinstance(v, list)},
            "page_count": sum(d["page_count"] for d in documents.values()), "uncovered_blocks": uncovered}


def validate_bundle(root):
    from .pipeline import load_corpus
    root = Path(root)
    corpus = load_corpus(root)
    result = validate_corpus(corpus)
    if any("build_role" in b for b in corpus["blocks"]):
        audit_path = root / "structure_audit.json"
        if not audit_path.exists():
            result["errors"].append("Normalized content requires a structure audit")
        else:
            audit = json.loads(audit_path.read_text())
            roles = {b['block_id']: b['role'] for b in audit['block_assignments']}
            if roles != {b['block_id']: b.get('build_role', 'content') for b in corpus['blocks']}:
                result["errors"].append("Structure audit role inventory mismatch")
    status = json.loads((root / "build_status.json").read_text())
    result["build_status"] = status["status"]
    if status["status"] != "complete":
        result["warnings"].append("Metadata/skills build is incomplete; corpus integrity is not navigation readiness")
    for path, digest in status["source_hashes"].items():
        if not Path(path).exists() or sha256(path) != digest:
            result["errors"].append("Original source missing or changed: " + Path(path).name)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest["artifacts"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != digest:
                result["errors"].append("Artifact missing or changed: " + relative)
    else:
        result["warnings"].append("Build is incomplete; final artifact manifest absent")
    if status["status"] == "complete":
        paths = json.loads((root / "skill_paths.json").read_text())
        if paths.get('adaptive_tree'):
            from .adaptive import validate_tree
            from qa_agent.bundle import local_asset
            try:
                tree = validate_tree(json.loads(local_asset(root, paths['adaptive_tree']).read_text()))
                if tree['section_documents'] != {s['section_id']: s['doc_id'] for s in corpus['sections']}:
                    raise ValueError('Navigation tree/corpus mismatch')
                for node in tree['nodes'].values():
                    local_asset(root, node['md_path'])
            except (ValueError, KeyError) as exc:
                result['errors'].append(str(exc))
            result['valid'] = not result['errors']
            result['ready_for_navigation'] = result['valid']
            return result
        if paths.get("hierarchy"):
            from qa_agent.hierarchical_navigation import hierarchy_assets
            try:
                for relative in hierarchy_assets(paths):
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root.resolve()) or not path.is_file():
                        raise ValueError("Missing/unsafe directory MD")
            except (ValueError, KeyError) as exc:
                result["errors"].append("Invalid navigation hierarchy: " + str(exc))
        aliases = paths.get("section_aliases") or {s["section_id"]: s["section_id"] for s in corpus["sections"]}
        inverse = {sid: alias for alias, sid in aliases.items()}
        if set(inverse) != {s["section_id"] for s in corpus["sections"]} or len(inverse) != len(aliases):
            result["errors"].append("Section aliases are not a complete bijection")
        for doc in corpus["documents"]:
            path = (root / paths["documents"][doc["doc_id"]]).resolve()
            if not path.is_relative_to(root.resolve()):
                result["errors"].append("Unsafe skill path")
                continue
            text = path.read_text(encoding="utf-8")
            for section in corpus["sections"]:
                if section["doc_id"] == doc["doc_id"] and inverse.get(section["section_id"], "MISSING_ALIAS") not in text:
                    result["errors"].append("Section absent from document skill: " + section["section_id"])
    result["valid"] = not result["errors"]
    result["ready_for_navigation"] = result["valid"] and status["status"] == "complete"
    return result
