"""Explicit public data download, pinned to one upstream commit."""
import json
from urllib.request import urlopen

from _bootstrap import ROOT
from topic_rag.io import digest_file, write_json, write_jsonl

REVISION = "97808f3e5fd16ede40bbff6c2949af8139b2eb7b"


def main():
    import pyarrow.parquet as pq
    raw = ROOT / "data/emanual/raw"
    raw.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/galileo-ai/ragbench/resolve/{REVISION}"
    for source, dest in (("emanual/test-00000-of-00001.parquet", "test.parquet"),
                         ("README.md", "DATASET_CARD.md")):
        with urlopen(f"{url}/{source}", timeout=60) as response:
            (raw / dest).write_bytes(response.read())
    write_jsonl(raw / "test.jsonl", pq.read_table(raw / "test.parquet").to_pylist())
    manifest = {"dataset": "galileo-ai/ragbench", "subset": "emanual", "split": "test",
                "revision": REVISION, "license": "CC-BY-4.0",
                "files": {name: digest_file(raw / name) for name in ("test.parquet", "test.jsonl", "DATASET_CARD.md")}}
    write_json(raw / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

