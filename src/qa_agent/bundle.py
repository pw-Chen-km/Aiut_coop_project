"""Portable, private serving snapshots independent of original PDF locations.

An export copies checked assets, never modifies the immutable offline build.
Runtime validation checks only snapshot-local files; PDFs/Docling are not required.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import shutil

from doc2skill.config import sha256, write_json

SCHEMA = "qa-serving-bundle-v1"
ADAPTIVE_SCHEMA = 'qa-serving-bundle-v2'
REQUIRED = {"corpus.sqlite", "corpus.vectors.npy", "skill_paths.json"}


def local_asset(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Asset path must be a nonempty relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("Asset path must stay inside the bundle")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("Missing or unsafe bundle asset: " + relative)
    return candidate


def _skill_assets(paths: dict) -> set[str]:
    if not isinstance(paths.get("documents"), dict) or not paths["documents"]:
        raise ValueError("Missing document skills")
    assets = {paths["overall_doc_skill"], paths["navigation_policy"], *paths["documents"].values()}
    from .hierarchical_navigation import hierarchy_assets
    assets.update(hierarchy_assets(paths))
    assets.update(paths.get('node_skills', {}).values())
    if any(not isinstance(p, str) or not p.endswith(".md") for p in assets):
        raise ValueError("Navigation assets must be Markdown")
    return assets


def validate_serving_bundle(directory) -> dict:
    root = Path(directory).resolve()
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {SCHEMA, ADAPTIVE_SCHEMA}:
        raise ValueError("Expected a portable QA serving bundle; first run export-bundle")
    assets = manifest.get("artifacts", {})
    if not isinstance(assets, dict) or not REQUIRED <= assets.keys():
        raise ValueError("Serving manifest is incomplete")
    for relative, digest in assets.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid asset hash")
        if sha256(local_asset(root, relative)) != digest:
            raise ValueError("Serving asset checksum mismatch: " + relative)
    paths = json.loads((root / "skill_paths.json").read_text(encoding="utf-8"))
    if not _skill_assets(paths) <= assets.keys():
        raise ValueError("Every runtime MD must be checksummed")
    if paths.get('adaptive_tree'):
        from doc2skill.adaptive import validate_tree
        if paths['adaptive_tree'] not in assets:
            raise ValueError('Navigation registry must be checksummed')
        tree = validate_tree(json.loads(local_asset(root, paths['adaptive_tree']).read_text()))
        if paths.get('node_skills') != {n: v['md_path'] for n, v in tree['nodes'].items()}:
            raise ValueError('Node skill paths differ from registry')
    embedding = manifest.get("embedding", {})
    if not embedding.get("model") or not re.fullmatch(r"[0-9a-f]{40}", embedding.get("revision", "")):
        raise ValueError("Serving bundle requires a pinned embedding model")
    return manifest


def export_bundle(source_dir, output_dir) -> dict:
    from doc2skill.validation import validate_bundle
    from doc2skill.storage import Store

    source, target = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Choose a separate, non-overlapping serving directory")
    if target.exists():
        raise FileExistsError("Serving output exists; choose a new version")
    check = validate_bundle(source)
    if not check["valid"] or not check["ready_for_navigation"]:
        raise ValueError("Export requires an intact completed skill build")
    original_manifest = json.loads((source / "manifest.json").read_text())
    status = json.loads((source / "build_status.json").read_text())
    paths = json.loads((source / "skill_paths.json").read_text())
    assets = REQUIRED | _skill_assets(paths)
    if paths.get('adaptive_tree'):
        assets.add(paths['adaptive_tree'])
    if (source / 'metadata_support.json').exists():
        assets.add('metadata_support.json')
    if (source / "metadata.json").exists():
        assets.add("metadata.json")  # Human inspection link from the existing MD cards.
    for optional in ("native_span_sections.jsonl", "navigation_budget_audit.json"):
        if (source / optional).exists():
            assets.add(optional)
    if not assets <= original_manifest["artifacts"].keys():
        raise ValueError("Serving assets must be covered by the original manifest")
    files = {relative: local_asset(source, relative) for relative in assets}
    # Validate the actual database before copying, not just the JSON registries.
    with Store(source / "corpus.sqlite") as store:
        counts = {"documents": len(store.documents), "sections": len(store.sections), "chunks": len(store.chunks)}
        if store.metadata.get("embedding") != status["embedding"]:
            raise ValueError("Stored embedding identity disagrees with the build")
    target.mkdir(parents=True, exist_ok=False)
    for relative, path in files.items():
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if sha256(dest) != original_manifest["artifacts"][relative]:
            raise ValueError("Source changed during export; incomplete export must not be used")
    manifest = {"schema_version": ADAPTIVE_SCHEMA if paths.get('adaptive_tree') else SCHEMA, "private_data": True,
                "source_build_signature": original_manifest["build_signature"],
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "embedding": status["embedding"], "counts": counts,
                "artifacts": {relative: sha256(target / relative) for relative in sorted(assets)}}
    write_json(target / "bundle_manifest.json", manifest)
    return validate_serving_bundle(target)
