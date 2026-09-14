"""Configuration and reproducibility helpers; never persist credential values."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream) if path.suffix == ".toml" else json.load(stream)
    def reject_credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "api_key_env" and re.search(r"api_?key|secret|authorization|access_token|refresh_token|password|credential", key, re.I):
                    raise ValueError("Configuration must reference environment variables, never credential literals")
                reject_credentials(child)
        elif isinstance(value, list):
            for child in value:
                reject_credentials(child)
    reject_credentials(config)
    for section in ("parser", "chunking", "embedding", "offline_llm", "metadata", "runtime"):
        config.setdefault(section, {})
    if config["chunking"].get("max_tokens", 384) > 512:
        raise ValueError("chunking.max_tokens must not exceed the default E5 context of 512")
    # Credentials belong in environment variables, not checked-in config or manifests.
    for section in ("offline_llm", "runtime"):
        if any(key in config[section] for key in ("api_key", "token", "authorization", "secret")):
            raise ValueError(f"{section}: use api_key_env, not a credential literal")
    return config


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def environment_manifest() -> dict:
    import importlib.metadata
    versions = {}
    for name in ("docling", "docling-core", "transformers", "torch", "numpy", "qegs"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
