"""Resumable per-document native corpus build; does not open dialogue/QA files."""
from __future__ import annotations
import json
import time
from pathlib import Path
from .config import fingerprint, sha256, write_json, write_jsonl, read_jsonl, environment_manifest
from .pipeline import COLLECTIONS


def build_native_corpus(config, *, out_dir=None, stop_after="complete", resume=False, encoder=None,
                        metadata_client=None, resource_guard=None, progress=None, **unused):
    if stop_after != "complete":
        raise ValueError("Native build requires reviewed structure; use qa-benchmarks prepare for the raw import")
    from .amd import AMDMetadataClient, GPUResourceGuard, ResourceBusy
    from .embedding import E5Encoder
    from .native_review import generate_structure_review
    from .normalization import normalize_structure, write_structure_report
    from .metadata import generate_metadata
    from .chunking import build_chunks
    from .storage import Store
    from .validation import validate_corpus
    from .hierarchy import compile_hierarchy, write_hierarchical_skills
    source = Path(config["native_corpus_dir"]).resolve()
    root = Path(out_dir or config["output_dir"]).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Native build must be separate from imported corpus")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Choose a new build directory or --resume")
    source_hashes = {str(source / (k + ".jsonl")): sha256(source / (k + ".jsonl")) for k in COLLECTIONS}
    if (source / "native_spans.jsonl").exists():
        source_hashes[str(source / "native_spans.jsonl")] = sha256(source / "native_spans.jsonl")
    implementation_hashes = {name: sha256(Path(__file__).with_name(name)) for name in (
        "native_pipeline.py", "native_review.py", "normalization.py", "chunking.py", "metadata.py",
        "skills.py", "hierarchy.py", "amd.py", "embedding.py", "storage.py", "validation.py")}
    signature = fingerprint({"sources": source_hashes, "implementation": implementation_hashes,
        "config": {k: v for k, v in config.items() if k != "output_dir"}})
    status_path = root / "build_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status and status.get("build_signature") != signature:
        raise ValueError("Source/config/implementation changed; cannot resume into mixed artifacts")
    root.mkdir(parents=True, exist_ok=True)
    status.pop("error", None)
    status.update(build_signature=signature, source_hashes=source_hashes,
                  implementation_hashes=implementation_hashes, status="building",
                  environment=environment_manifest(), started_at=status.get("started_at", time.time()))
    original = {k: read_jsonl(source / (k + ".jsonl")) for k in COLLECTIONS}
    corpus = {k: [] for k in COLLECTIONS}
    metadata = {"sections": {}, "documents": {}, "provenance": {"mode": "llm_endpoint", "document_calls": []}}
    audits = []
    try:
        client = metadata_client or AMDMetadataClient(config["offline_llm"], root / "llm_audit")
        guard = resource_guard or GPUResourceGuard(config["gpu_guard"], root / "resource_checks")
        if metadata_client is None:
            write_json(root / "endpoint_preflight.json", client.transport.preflight())
        encoder = encoder or E5Encoder(config["embedding"], local_only=True)
        status["embedding"] = encoder.provenance
        write_json(status_path, status)
        for index, doc in enumerate(original["documents"], 1):
            did = doc["doc_id"]
            stage_dir = root / "document_builds" / did.replace(":", "-")
            completion = stage_dir / "complete.json"
            single = {k: [r for r in rows if r.get("doc_id") == did] for k, rows in original.items()}
            if completion.exists():
                saved = json.loads(completion.read_text())
                for relative, digest in saved["artifacts"].items():
                    p = (stage_dir / relative).resolve()
                    if not p.is_relative_to(stage_dir.resolve()) or sha256(p) != digest:
                        raise ValueError("Completed document checkpoint changed")
                normalized = json.loads((stage_dir / "corpus.json").read_text())
                audit = json.loads((stage_dir / "structure_audit.json").read_text())
                meta = json.loads((stage_dir / "metadata.json").read_text())
            else:
                stage_dir.mkdir(parents=True, exist_ok=True)
                guard.check("document:" + did)
                status.update(current_document=did, documents_completed=index - 1)
                write_json(status_path, status)
                review_path = stage_dir / "structure_review.json"
                if review_path.exists():
                    review = json.loads(review_path.read_text())
                    normalized, audit = normalize_structure(single, review)
                else:
                    review, normalized, audit = generate_structure_review(single, client, stage_dir)
                write_json(stage_dir / "structure_audit.json", audit)
                write_structure_report(single, normalized, audit, stage_dir / "structure_report.md")
                normalized["chunks"] = build_chunks(None, normalized, config, encoder.tokenizer)
                check = validate_corpus(normalized, encoder.tokenizer, config["chunking"].get("max_tokens", 384))
                if not check["valid"]:
                    raise ValueError("Normalized native corpus invalid: " + "; ".join(check["errors"][:3]))
                meta = generate_metadata(normalized, client, {**config["metadata"], "cache_dir": str(stage_dir / "metadata_cache")})
                write_json(stage_dir / "corpus.json", normalized)
                write_json(stage_dir / "metadata.json", meta)
                write_json(completion, {"doc_id": did, "artifacts": {
                    str(p.relative_to(stage_dir)): sha256(p) for p in stage_dir.rglob("*") if p.is_file() and p != completion}})
            for kind in COLLECTIONS:
                corpus[kind].extend(normalized[kind])
            metadata["sections"].update(meta["sections"])
            metadata["documents"].update(meta["documents"])
            metadata["provenance"]["document_calls"].append(meta["provenance"])
            audits.append(audit)
            status["documents_completed"] = index
            write_json(status_path, status)
            if progress:
                progress(f"native document {index}/{len(original['documents'])}: {did}")
        hierarchy = compile_hierarchy(corpus, metadata, client, root, guard.check)
        paths = write_hierarchical_skills(corpus, metadata, hierarchy, root / "skills")
        for name in ("overall_doc_skill", "navigation_policy"):
            paths[name] = str(Path(paths[name]).resolve().relative_to(root))
        paths["documents"] = {k: str(Path(v).resolve().relative_to(root)) for k, v in paths["documents"].items()}
        for kind in ("domains", "groups"):
            for row in paths["hierarchy"][kind].values():
                row["skill_path"] = str(Path(row["skill_path"]).relative_to(root))
        write_json(root / "skill_paths.json", paths)
        if config.get("navigation_audit"):
            from .hierarchy import audit_directory_budget
            from transformers import AutoTokenizer
            options = config["navigation_audit"]
            tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_path"], local_files_only=True,
                                                       trust_remote_code=False)
            budget_report = audit_directory_budget(paths, root,
                lambda text: len(tokenizer.encode(text, add_special_tokens=True)), options)
            budget_report["tokenizer_files"] = {p.name: sha256(p) for p in Path(options["tokenizer_path"]).iterdir()
                                                if p.is_file()}
            write_json(root / "navigation_budget_audit.json", budget_report)
            if not budget_report["valid"]:
                raise ValueError("Navigation directory exceeds the reserved 8K request budget; no truncation applied")
        elif metadata_client is None:
            raise ValueError("Production build requires navigation_audit with a local 4B tokenizer")
        for kind in COLLECTIONS:
            write_jsonl(root / (kind + ".jsonl"), corpus[kind])
        write_json(root / "metadata.json", metadata)
        write_json(root / "structure_audit.json", {"block_assignments": [b for a in audits for b in a["block_assignments"]],
            "documents": audits, "before_sections": len(original["sections"]), "after_sections": len(corpus["sections"])})
        write_structure_report(original, corpus, {"provenance": {"mode": "llm_endpoint"}}, root / "structure_report.md")
        if (source / "native_spans.jsonl").exists():
            original_spans = read_jsonl(source / "native_spans.jsonl")
            by_doc = {}
            for b in corpus["blocks"]:
                by_doc.setdefault(b["doc_id"], []).append(b)
            for span in original_spans:
                span["section_ids"] = sorted({b["section_id"] for b in by_doc[span["doc_id"]]
                    if max(span["start_sp"], b["doc_start_char"]) < min(span["end_sp"], b["doc_end_char"])})
            write_jsonl(root / "native_span_sections.jsonl", original_spans)
        database = root / "corpus.sqlite"
        if database.exists():
            with Store(database) as recovered:
                if recovered.records("chunks") != corpus["chunks"] or recovered.metadata.get("embedding") != encoder.provenance:
                    raise ValueError("Existing native index differs from completed corpus")
        else:
            vectors = encoder.encode_documents([c["embedding_text"] for c in corpus["chunks"]])
            with Store.create(database, {**corpus, "metadata": {"embedding": encoder.provenance}}, vectors):
                pass
        status.update(status="complete", completed_stages=["native_import", "normalization", "chunking", "metadata", "hierarchy", "index"],
                      completed_at=time.time(), counts={k: len(v) for k, v in corpus.items()})
    except ResourceBusy as exc:
        status.update(status="paused_resource_busy", error={"type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:
        status.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        status["source_hashes_unchanged"] = all(sha256(p) == h for p, h in source_hashes.items())
        if not status["source_hashes_unchanged"]:
            status["status"] = "blocked"
        write_json(status_path, status)
    if status["status"] == "complete":
        write_json(root / "manifest.json", {"build_signature": signature, "embedding": status["embedding"], "status": "complete",
            "artifacts": {str(p.relative_to(root)): sha256(p) for p in root.rglob("*")
                          if p.is_file() and p.name not in {"manifest.json", "build_status.json"}}})
    return status
