"""Small reproducibility helpers for append-only private experiment outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def new_directory(path, *, protected=()):
    path = Path(path).resolve()
    for value in protected:
        source = Path(value).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Output must not overlap immutable input directories")
    path.mkdir(parents=True, exist_ok=False)
    return path
