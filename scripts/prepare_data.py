"""Normalize downloaded JSONL and chunk it. No model inference or network."""
import argparse
import json

from _bootstrap import ROOT
from topic_rag.datasets.emanual import normalize
from topic_rag.indexing import load_tokenizer, make_chunks, validate_chunks
from topic_rag.io import digest_file, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()
    root = ROOT / "data/emanual"
    report = normalize(root / "raw/test.jsonl", root / "processed")
    sources = read_jsonl(root / "processed/sources.jsonl")
    tokenizer_path = ROOT / "assets/e5-tokenizer/tokenizer.json"
    chunks = make_chunks(sources, load_tokenizer(tokenizer_path), args.max_tokens)
    validate_chunks(sources, chunks)
    write_jsonl(root / "processed/chunks.jsonl", chunks)
    report.update(chunk_count=len(chunks), max_e5_tokens=args.max_tokens,
                  tokenizer_sha256=digest_file(tokenizer_path),
                  chunk_policy="sentence_boundary_preferred_exact_contiguous_no_overlap",
                  max_observed_e5_tokens=max(c["embedding_tokens"] for c in chunks))
    write_json(root / "processed/validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

