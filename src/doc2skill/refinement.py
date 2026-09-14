"""Offline structure review -> normalized chunks -> reviewed navigation skills.

Uses an existing parsed corpus, not QA or gold answers. Review files can come
from an approved endpoint or the current assistant, with honest provenance.
The original build is immutable. No remote LLM calls are made by this path.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .config import fingerprint, sha256, write_json, write_jsonl, environment_manifest
from .normalization import aliases, normalize_structure, write_structure_report, input_fingerprint


def compile_reviewed_metadata(corpus, original, review, structure_review):
    from .metadata import _validate_metadata
    if review.get("schema") != "navigation-review-v1" or review.get("structure_fingerprint") != fingerprint(structure_review):
        raise ValueError("Navigation review does not match the reviewed structure")
    provenance = review.get("provenance", {})
    if provenance.get("mode") not in {"assistant_authored", "llm_endpoint", "human_reviewed", "test_fixture"}:
        raise ValueError("Navigation review needs truthful provenance")
    maps = aliases(original)
    if set(provenance.get("reviewed_documents", [])) != set(maps["documents"]):
        raise ValueError("Navigation review must cover all documents")
    sections = {s["section_id"]: s for s in corpus["sections"]}
    key_to_sid = {s["review_key"]: s["section_id"] for s in sections.values()}
    blocks = {b["block_id"]: b for b in corpus["blocks"] if b.get("build_role") != "noise"}
    if set(review["sections"]) != set(key_to_sid) or set(review["documents"]) != set(maps["documents"]):
        raise ValueError("Navigation cards must cover exactly the corrected section/document inventory")

    def authored_card(key, seen=None):
        seen = set(seen or ())
        if key in seen or key not in review["sections"]:
            raise ValueError("Invalid/cyclic authored card inheritance")
        seen.add(key)
        card = review["sections"][key]
        if "inherit" not in card:
            return card
        if not card.get("support"):
            raise ValueError("Reused prose requires its own document-local support")
        return {**authored_card(card['inherit'], seen), **{k: v for k, v in card.items() if k != 'inherit'}}

    def descendants(sid):
        found = {sid}
        for child in sections.values():
            if child["parent_id"] == sid:
                found.update(descendants(child["section_id"]))
        return found

    def compile_card(card, allowed):
        for required in ("summary", "navigation_summary", "question_intents", "operations", "not_covered", "support"):
            if required not in card:
                raise ValueError("Missing authored navigation field: " + required)
        if not card["summary"].strip() or not card["navigation_summary"].strip():
            raise ValueError("Empty authored summary")
        refs = []
        for key in card["support"]:
            if key not in maps["blocks"]:
                raise ValueError("Unknown support block in authored metadata")
            original_block = maps["blocks"][key]
            bid = original_block["block_id"]
            if bid not in blocks or bid not in allowed or not original_block["text"].strip():
                raise ValueError("Authored support must be non-noise text inside the section/document scope")
            refs.append({"block_id": bid, "quote": original_block["text"]})
        data = {"summary": card["summary"], "navigation_summary": card["navigation_summary"],
                "navigation_intents": card["question_intents"][:2],
                "operations": card["operations"], "question_intents": card["question_intents"],
                "use_when": card.get("use_when", card["question_intents"]), "not_covered": card["not_covered"],
                "topics": card.get("topics", []), "aliases": card.get("aliases", []),
                "related_sections": [key_to_sid[k] for k in card.get("related_sections", [])],
                "source_refs": refs, "supported_observations": []}
        _validate_metadata(data, blocks, allowed, set(sections))
        return {**data, "question_intents_are_inferred": True, "generation_mode": provenance["mode"]}

    result = {"sections": {}, "documents": {}, "provenance": provenance,
              "coverage": {"section_count": len(sections), "document_count": len(corpus['documents']),
                           "input_fingerprint": input_fingerprint(original),
                           "reviewed_body_characters": sum(len(b['text']) for b in blocks.values()),
                           "verification": "Author-declared full read; exact reference and inventory validation, not semantic proof"}}
    for key, card in review["sections"].items():
        sid = key_to_sid[key]
        scope = descendants(sid)
        allowed = {bid for bid, b in blocks.items() if b["section_id"] in scope}
        result["sections"][sid] = {**compile_card(authored_card(key), allowed), "section_id": sid}
    for dk, card in review["documents"].items():
        did = maps["documents"][dk]["doc_id"]
        allowed = {bid for bid, b in blocks.items() if b["doc_id"] == did}
        result["documents"][did] = {**compile_card(card, allowed),
                                    "section_ids": [sid for sid, s in sections.items() if s["doc_id"] == did]}
    result["overall"] = compile_card(review["overall"], set(blocks))
    return result


def refine_corpus(config, input_dir, review_path, output_dir, *, metadata_review=None, encoder=None, stop_after="complete"):
    from .pipeline import load_corpus, COLLECTIONS
    from .validation import validate_bundle, validate_corpus
    from .embedding import E5Encoder
    from .chunking import build_chunks
    from .storage import Store
    from .skills import write_skills
    source, root = Path(input_dir).resolve(), Path(output_dir).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Refinement requires a separate, non-overlapping version directory")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refinement output exists; choose a new version")
    if stop_after not in {"structure", "complete"}:
        raise ValueError("stop_after must be structure or complete")
    if not (source / "manifest.json").is_file():
        raise ValueError("Refinement requires a checksummed input bundle")
    checks = validate_bundle(source)
    if not checks["valid"]:
        raise ValueError("Input bundle is not intact: " + "; ".join(checks["errors"][:3]))
    original = load_corpus(source)
    # Protect external inputs even if a caller chooses an output near them.
    from .pipeline import _protect_output
    _protect_output(root, config)
    review = json.loads(Path(review_path).read_text())
    corpus, audit = normalize_structure(original, review)
    metadata = None
    if stop_after == "complete":
        if metadata_review:
            metadata = compile_reviewed_metadata(corpus, original, json.loads(Path(metadata_review).read_text()), review)
        else:
            raise ValueError("Complete refinement requires authored navigation metadata; no placeholder summaries")
    previous = json.loads((source / "build_status.json").read_text())
    status = {"status": "building", "started_at": time.time(), "environment": environment_manifest(),
              "completed_stages": [], "source_hashes": previous["source_hashes"],
              "input_bundle": str(source), "input_manifest_sha256": sha256(source / "manifest.json"),
              "build_signature": fingerprint({"input": input_fingerprint(original), "review": review,
                                               "metadata": metadata, "chunking": config.get("chunking", {})}),
              "generation_mode": review["provenance"]["mode"]}
    root.mkdir(parents=True, exist_ok=True)
    try:
        write_json(root / "structure_review.json", review)
        write_json(root / "structure_audit.json", audit)
        write_structure_report(original, corpus, audit, root / "structure_report.md")
        status["completed_stages"].append("normalization")
        ec = {**config["embedding"], "model": previous["embedding"]["model"], "revision": previous["embedding"]["revision"]}
        encoder = encoder or E5Encoder(ec, local_only=True, tokenizer_only=stop_after == "structure")
        corpus["chunks"] = build_chunks(None, corpus, config, encoder.tokenizer)
        for collection in COLLECTIONS:
            write_jsonl(root / f"{collection}.jsonl", corpus[collection])
        validation = validate_corpus(corpus, encoder.tokenizer, config["chunking"].get("max_tokens", 384))
        write_json(root / "corpus_validation.json", validation)
        if not validation["valid"]:
            raise ValueError("Corrected corpus validation failed: " + "; ".join(validation["errors"][:5]))
        status["completed_stages"].append("chunking")
        status["embedding"] = encoder.provenance
        if stop_after == "complete":
            vectors = encoder.encode_documents([c["embedding_text"] for c in corpus["chunks"]])
            with Store.create(root / "corpus.sqlite", {**corpus, "metadata": {"embedding": encoder.provenance}}, vectors):
                pass
            status["completed_stages"].append("index")
            write_json(root / "navigation_review.json", json.loads(Path(metadata_review).read_text()))
            write_json(root / "metadata.json", metadata)
            paths = write_skills(corpus, metadata, root / "skills")
            paths = {**paths, "overall_doc_skill": str(Path(paths["overall_doc_skill"]).relative_to(root)),
                     "navigation_policy": str(Path(paths["navigation_policy"]).relative_to(root)),
                     "documents": {k: str(Path(v).relative_to(root)) for k, v in paths["documents"].items()}}
            write_json(root / "skill_paths.json", paths)
            with sqlite3.connect(root / "corpus.sqlite") as con:
                con.execute("CREATE TABLE navigation_metadata (id TEXT PRIMARY KEY, json TEXT NOT NULL)")
                for kind in ("sections", "documents"):
                    for identity, value in metadata[kind].items():
                        con.execute("INSERT INTO navigation_metadata VALUES (?,?)", (identity, json.dumps(value)))
                con.execute("INSERT INTO navigation_metadata VALUES (?,?)", ("overall", json.dumps(metadata["overall"])))
            status["completed_stages"].append("metadata")
        status["status"] = "complete" if stop_after == "complete" else "structure_complete"
    except Exception as exc:
        status.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        status["source_hashes_unchanged"] = all(Path(p).is_file() and sha256(p) == h for p, h in status["source_hashes"].items())
        input_unchanged = sha256(source / "manifest.json") == status["input_manifest_sha256"] and validate_bundle(source)["valid"]
        status["input_bundle_unchanged"] = input_unchanged
        if not status["source_hashes_unchanged"] or not input_unchanged:
            status["status"] = "blocked"
            status["error"] = {"type": "InputIntegrityError", "message": "Original PDF or input bundle changed during refinement"}
        status["completed_at"] = time.time()
        write_json(root / "build_status.json", status)
    if status["status"] == "blocked":
        raise ValueError("Input changed during refinement")
    write_json(root / "manifest.json", {"build_signature": status["build_signature"], "status": status["status"],
               "embedding": status["embedding"], "artifacts": {str(p.relative_to(root)): sha256(p) for p in root.rglob('*')
                          if p.is_file() and p.name not in {"build_status.json", "manifest.json"}}})
    return status
