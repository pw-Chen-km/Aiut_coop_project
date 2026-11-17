"""Record a LOCAL GGUF's exact hash for a run; does not start a model or experiment."""
import argparse
from pathlib import Path

from _bootstrap import ROOT
from topic_rag.io import digest_file, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/qwen-checksum.json")
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            parser.error("file is not a GGUF binary (possibly a Git LFS pointer)")
    write_json(args.output, {"filename": path.name, "size_bytes": path.stat().st_size,
                             "sha256": digest_file(path)})
    print(args.output.resolve())
