"""Content-addressed manifests for reproducible dataset artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import canonical_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory tree without embedding absolute paths."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries = []
    for item in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": item.relative_to(source).as_posix(),
                "sha256": sha256_file(item),
                "size_bytes": item.stat().st_size,
            }
        )
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def _record_count(path: Path) -> int | None:
    if path.suffix.lower() != ".jsonl":
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_manifest(
    paths: Iterable[str | Path],
    *,
    dataset_version: str,
    base_dir: str | Path | None = None,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("at least one artifact is required")
    base = Path(base_dir).resolve() if base_dir is not None else None
    entries: list[dict[str, Any]] = []
    for source in sources:
        resolved = source.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if base is not None:
            try:
                name = resolved.relative_to(base).as_posix()
            except ValueError as exc:
                raise ValueError(f"artifact is outside base_dir: {resolved}") from exc
        else:
            name = source.name
        entry: dict[str, Any] = {
            "path": name,
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
        count = _record_count(resolved)
        if count is not None:
            entry["record_count"] = count
        entries.append(entry)
    entries.sort(key=lambda item: item["path"])
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("manifest paths must be unique; provide a shared base_dir")
    payload = {
        "manifest_version": "1",
        "dataset_version": dataset_version,
        "schema_version": schema_version,
        "files": entries,
    }
    payload["dataset_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def verify_manifest(root: str | Path, manifest: Mapping[str, Any]) -> list[str]:
    """Return deterministic verification errors; an empty list means success."""

    errors: list[str] = []
    base = Path(root).resolve()
    files = manifest.get("files")
    if not isinstance(files, list):
        return ["manifest.files must be an array"]
    for entry in files:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            errors.append("manifest file entry must contain path")
            continue
        target = (base / entry["path"]).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            errors.append(f"manifest path escapes root: {entry['path']}")
            continue
        if not target.is_file():
            errors.append(f"missing artifact: {entry['path']}")
            continue
        if entry.get("sha256") != sha256_file(target):
            errors.append(f"sha256 mismatch: {entry['path']}")
        if entry.get("size_bytes") != target.stat().st_size:
            errors.append(f"size mismatch: {entry['path']}")
        expected_count = entry.get("record_count")
        if expected_count is not None and expected_count != _record_count(target):
            errors.append(f"record_count mismatch: {entry['path']}")
    unsigned = {
        key: value for key, value in manifest.items() if key != "dataset_sha256"
    }
    expected_dataset_hash = hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    if manifest.get("dataset_sha256") != expected_dataset_hash:
        errors.append("dataset_sha256 mismatch")
    return errors
