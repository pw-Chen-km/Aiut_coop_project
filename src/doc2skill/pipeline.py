"""Resumable corpus-only compilation. Never reads questions or gold atoms."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import environment_manifest, fingerprint, read_jsonl, sha256, write_json, write_jsonl

COLLECTIONS = ("documents", "sections", "blocks", "source_units", "chunks", "figures", "furniture")


def load_sources(config: dict) -> list[dict]:
    manifest = json.loads(Path(config["source_manifest"]).read_text(encoding="utf-8"))
    registry = {row["source_sha256"]: row for row in read_jsonl(config["document_registry"])} if config.get("document_registry") else {}
    source_dir = Path(config["source_dir"]).resolve()
    documents = []
    for item in manifest["documents"]:
        path = (source_dir / item["filename"]).resolve()
        if not path.is_relative_to(source_dir) or path.suffix.lower() != ".pdf":
            raise ValueError("Source manifest contains unsafe or non-PDF path")
        digest = sha256(path)
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Source checksum/size mismatch: {path.name}")
        document = dict(registry.get(digest, {}))
        document.update({"source_path": str(path), "source_filename": path.name,
                         "source_sha256": digest, "document_version": item["version"],
                         "page_count": item["pages"]})
        document.setdefault("doc_id", "d2s:doc:" + digest[:24])
        document.setdefault("title", path.stem)
        documents.append(document)
    if len({d["doc_id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate document identity in source manifest")
    return documents


def load_corpus(out_dir: str | Path) -> dict:
    root = Path(out_dir)
    return {key: read_jsonl(root / f"{key}.jsonl") for key in COLLECTIONS}


def _protect_output(root: Path, config: dict) -> None:
    source_dir = Path(config["source_dir"]).resolve()
    protected = [source_dir]
    for key in ("source_manifest", "document_registry", "legacy_chunks"):
        if config.get(key):
            protected.append(Path(config[key]).resolve().parent)
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Output must be separate from the source/benchmark directories")


def build_corpus(config: dict, *, out_dir=None, stop_after="complete", resume=False,
                 encoder=None, parser=None, metadata_client=None, progress=None) -> dict:
    """Build new artifacts. Injectable dependencies are for explicit test fixtures only."""
    if config.get("native_corpus_dir"):
        from .native_pipeline import build_native_corpus
        return build_native_corpus(config, out_dir=out_dir, stop_after=stop_after, resume=resume,
                                   encoder=encoder, metadata_client=metadata_client, progress=progress)
    if stop_after not in ("parse", "index", "complete"):
        raise ValueError("stop_after must be parse, index or complete")
    from .embedding import E5Encoder
    from .parsing import parse_pdf
    from .alignment import align_corpus
    from .storage import Store

    root = Path(out_dir or config["output_dir"]).resolve()
    _protect_output(root, config)
    sources = load_sources(config)
    source_hashes = {d["source_path"]: d["source_sha256"] for d in sources}
    # Credentials are not part of config; only their environment variable names are.
    build_config = {k: v for k, v in config.items() if k not in ("offline_llm", "metadata", "runtime", "output_dir")}
    signature = fingerprint({"config": build_config, "sources": sources,
                             "input_registry_hashes": {key: sha256(config[key]) for key in
                                                       ("source_manifest", "document_registry", "legacy_chunks") if config.get(key)}})
    status_path = root / "build_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError("Output already exists; use --resume or choose a new version directory")
    if resume and previous and previous.get("build_signature") != signature:
        raise ValueError("Cannot resume with changed corpus/parser/chunk/embedding configuration")
    if resume and root.exists() and any(root.iterdir()) and not previous:
        raise ValueError("Cannot resume directory without build_status.json")
    if resume:
        for relative, digest in previous.get("checkpoint_artifacts", {}).items():
            artifact = (root / relative).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file() or sha256(artifact) != digest:
                raise ValueError("Resume artifact integrity mismatch: " + relative)
    root.mkdir(parents=True, exist_ok=True)
    status = {**previous, "build_signature": signature, "status": "building", "environment": environment_manifest(),
              "source_hashes": source_hashes, "started_at": previous.get("started_at", time.time()),
              "completed_stages": previous.get("completed_stages", [])}

    def checkpoint(stage):
        if stage not in status["completed_stages"]:
            status["completed_stages"].append(stage)
        status["checkpoint_artifacts"] = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*")
                                          if p.is_file() and p.name not in ("build_status.json", "manifest.json")}
        write_json(status_path, status)
        if progress:
            progress(stage)

    write_json(status_path, status)
    try:
        if "parse" not in status["completed_stages"]:
            encoder = encoder or E5Encoder(config["embedding"], tokenizer_only=stop_after == "parse")
            status["embedding"] = encoder.provenance
            corpus = {key: [] for key in COLLECTIONS}
            audits = []
            for document in sources:
                key = fingerprint({"document": document["source_sha256"], "signature": signature})[:24]
                fragment_file = root / "parsed" / key / "fragment.json"
                if resume and fragment_file.exists():
                    fragment = json.loads(fragment_file.read_text(encoding="utf-8"))
                else:
                    fragment = (parser or parse_pdf)(Path(document["source_path"]), document, config,
                                                     fragment_file.parent, encoder.tokenizer)
                    write_json(fragment_file, fragment)
                if len(fragment.get("documents", [])) != 1 or fragment["documents"][0]["page_count"] != document["page_count"]:
                    raise ValueError("Parsed document/page count differs from source manifest")
                for collection in COLLECTIONS:
                    corpus[collection].extend(fragment.get(collection, []))
                audits.append(fragment.get("audit", {}))
                if progress:
                    progress("parsed " + document["source_filename"])
            for collection in COLLECTIONS:
                write_jsonl(root / f"{collection}.jsonl", corpus[collection])
            write_json(root / "parse_audit.json", audits)
            from .validation import validate_corpus
            validation = validate_corpus(corpus, encoder.tokenizer, config["chunking"].get("max_tokens", 384))
            write_json(root / "corpus_validation.json", validation)
            if not validation["valid"]:
                raise ValueError("Parsed corpus failed integrity checks: " + "; ".join(validation["errors"][:5]))
            checkpoint("parse")
        else:
            corpus = load_corpus(root)
            from .validation import validate_corpus
            if not validate_corpus(corpus, max_tokens=config["chunking"].get("max_tokens", 384))["valid"]:
                raise ValueError("Resumed corpus failed source integrity checks")
        if "alignment" not in status["completed_stages"] and config.get("legacy_chunks"):
            alignment = align_corpus(corpus, config["legacy_chunks"])
            write_json(root / "alignment.json", alignment)
            checkpoint("alignment")
        from .reporting import write_inspection_report
        alignment_path = root / "alignment.json"
        write_inspection_report(corpus, root, json.loads(alignment_path.read_text()) if alignment_path.exists() else None)
        if stop_after != "parse" and "index" not in status["completed_stages"]:
            embed_config = {**config["embedding"], "revision": status["embedding"]["revision"]}
            if encoder is None or getattr(encoder, "model", True) is None:
                encoder = E5Encoder(embed_config)
            database = root / "corpus.sqlite"
            if database.exists():
                # A published index can outlive a crash immediately before the checkpoint.
                # Verify its complete corpus identity; never overwrite it during resume.
                with Store(database) as recovered:
                    for collection in ("documents", "sections", "blocks", "source_units", "chunks", "figures"):
                        if recovered.records(collection) != corpus[collection]:
                            raise ValueError("Interrupted index does not match the parsed corpus")
                    if recovered.metadata.get("embedding") != encoder.provenance:
                        raise ValueError("Interrupted index embedding identity is not verifiable")
            else:
                vectors = encoder.encode_documents([c["embedding_text"] for c in corpus["chunks"]])
                with Store.create(database, {**corpus, "metadata": {"embedding": encoder.provenance}}, vectors):
                    pass
            status["embedding"] = encoder.provenance
            checkpoint("index")
        if stop_after == "complete":
            from .llm import ChatClient
            from .metadata import generate_metadata
            from .skills import write_skills
            generation_settings = {k: v for k, v in config["offline_llm"].items() if k not in
                                   ("approved_private_data", "zero_data_retention_confirmed", "api_key_env", "timeout_seconds")}
            meta_signature = fingerprint({"llm": generation_settings, "metadata": config["metadata"]})
            if status.get("metadata_signature") and status["metadata_signature"] != meta_signature:
                raise ValueError("Metadata configuration changed; build a new version")
            if "metadata" not in status["completed_stages"]:
                client = metadata_client or ChatClient(config["offline_llm"])
                status["metadata_signature"] = meta_signature
                write_json(status_path, status)
                metadata = generate_metadata(corpus, client, {**config["metadata"], "cache_dir": str(root / "metadata_cache")})
                write_json(root / "metadata.json", metadata)
                paths = write_skills(corpus, metadata, root / "skills")
                paths = {"overall_doc_skill": str(Path(paths["overall_doc_skill"]).resolve().relative_to(root)),
                         "navigation_policy": str(Path(paths["navigation_policy"]).resolve().relative_to(root)),
                         "documents": {k: str(Path(v).resolve().relative_to(root)) for k, v in paths["documents"].items()},
                         "section_aliases": paths.get("section_aliases", {})}
                write_json(root / "skill_paths.json", paths)
                status["metadata_signature"] = meta_signature
                # Store metadata separately without rebuilding or changing the vector matrix.
                import sqlite3
                connection = sqlite3.connect(root / "corpus.sqlite")
                connection.execute("CREATE TABLE IF NOT EXISTS navigation_metadata (id TEXT PRIMARY KEY, json TEXT NOT NULL)")
                for kind in ("sections", "documents"):
                    for identity, value in metadata[kind].items():
                        connection.execute("INSERT OR REPLACE INTO navigation_metadata VALUES (?,?)", (identity, json.dumps(value)))
                connection.execute("INSERT OR REPLACE INTO navigation_metadata VALUES (?,?)", ("overall", json.dumps(metadata["overall"])))
                connection.commit()
                connection.close()
                checkpoint("metadata")
        status["status"] = "complete" if stop_after == "complete" else f"{stop_after}_complete"
        status["completed_at"] = time.time()
    except Exception as exc:
        status["status"] = "blocked"
        status["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        status["source_hashes_unchanged"] = all(Path(p).is_file() and sha256(p) == digest for p, digest in source_hashes.items())
        if not status["source_hashes_unchanged"]:
            status["status"] = "blocked"
            status["error"] = {"type": "SourceIntegrityError", "message": "Source changed during build"}
        write_json(status_path, status)
        if not status["source_hashes_unchanged"]:
            raise ValueError("Source changed during build; artifacts must not be used")
    artifact_hashes = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*")
                       if p.is_file() and p.name not in ("manifest.json", "build_status.json")}
    write_json(root / "manifest.json", {"build_signature": signature, "embedding": status.get("embedding"),
                                      "status": status["status"], "artifacts": artifact_hashes})
    return status
