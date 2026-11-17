"""Offline exact-span chunking and E5 assets. No model is loaded at import."""
from pathlib import Path
import importlib.metadata
import json

import numpy as np

from topic_rag.io import digest_file, read_jsonl, write_json, write_jsonl


def load_tokenizer(path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def make_chunks(sources, tokenizer, max_tokens=384):
    chunks = []
    for source in sources:
        text, start = source["text"], 0
        title = source["title"]
        while start < len(text):
            def fits(end):
                return len(tokenizer.encode("passage: " + title + "\n" + text[start:end]).ids) <= max_tokens
            lo, hi = start + 1, len(text)
            if not fits(lo):
                raise ValueError("title leaves no room for chunk text")
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid - 1
            end = lo
            # Prefer a complete sentence, but never discard an overlong sentence.
            boundaries = [s["end"] for s in source["sentence_spans"] if start < s["end"] <= end]
            if end < len(text) and boundaries:
                end = max(boundaries)
            chunk_text = text[start:end]
            embedding_text = "passage: " + title + "\n" + chunk_text
            if len(tokenizer.encode(embedding_text).ids) > max_tokens:
                raise ValueError("chunk exceeds token budget")
            chunks.append({
                "chunk_id": f"{source['source_id']}:{start}:{end}", "source_id": source["source_id"],
                "title": title, "text": chunk_text, "source_start": start, "source_end": end,
                "retrieval_text": title + "\n" + chunk_text, "embedding_text": embedding_text,
                "embedding_tokens": len(tokenizer.encode(embedding_text).ids),
            })
            start = end
    return chunks


def validate_chunks(sources, chunks):
    by_source = {row["source_id"]: row["text"] for row in sources}
    for sid, text in by_source.items():
        rows = sorted([c for c in chunks if c["source_id"] == sid], key=lambda c: c["source_start"])
        cursor = 0
        for chunk in rows:
            start, end = chunk["source_start"], chunk["source_end"]
            if start != cursor or not start < end <= len(text) or chunk["text"] != text[start:end]:
                raise ValueError(f"chunk coverage/alignment failure: {sid}")
            cursor = end
        if cursor != len(text):
            raise ValueError(f"missing source content: {sid}")
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    if any(c["source_id"] not in by_source for c in chunks):
        raise ValueError("unknown chunk source")


class E5Encoder:
    def __init__(self, model_dir, threads=4):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).cpu().eval()

    def encode(self, texts, batch_size=8):
        import torch
        output = []
        for start in range(0, len(texts), batch_size):
            selected = texts[start:start + batch_size]
            if any(len(self.tokenizer.encode(t)) > 512 for t in selected):
                raise ValueError("E5 input exceeds 512 tokens; no silent truncation")
            batch = self.tokenizer(selected, padding=True, truncation=False, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(mean, dim=1)
            output.append(vectors.cpu().numpy().astype(np.float32))
        return np.concatenate(output)

    def query(self, text):
        return self.encode(["query: " + text])[0]


def model_fingerprint(path):
    # Hash weights as well as tokenizer/config: a path or model name is not a version.
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    return {str(p.relative_to(root)): digest_file(p) for p in files}


def build_index(chunks_path, model_dir, output_dir):
    if output_dir.exists():
        raise ValueError("index directory already exists; choose a new version")
    chunks = read_jsonl(chunks_path)
    encoder = E5Encoder(model_dir)
    vectors = encoder.encode([c["embedding_text"] for c in chunks])
    output_dir.mkdir(parents=True)
    np.save(output_dir / "vectors.npy", vectors, allow_pickle=False)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    manifest = {
        "chunk_count": len(chunks), "input_chunks_sha256": digest_file(chunks_path),
        "e5_files": model_fingerprint(model_dir),
        "encoder": {"pooling": "attention_mask_mean_then_l2_normalize", "device": "cpu",
                    "threads": 4, "input_prefix": "passage: ", "dimension": vectors.shape[1]},
        "implementation_sha256": digest_file(Path(__file__)),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "numpy")},
        "files": {name: digest_file(output_dir / name) for name in ("vectors.npy", "chunks.jsonl")},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_index(index_dir, chunks_path, model_dir):
    manifest = json.loads((index_dir / "manifest.json").read_text())
    if digest_file(chunks_path) != manifest["input_chunks_sha256"]:
        raise ValueError("stale index: chunks changed")
    for name, digest in manifest["files"].items():
        if digest_file(index_dir / name) != digest:
            raise ValueError(f"index checksum mismatch: {name}")
    if model_fingerprint(model_dir) != manifest["e5_files"]:
        raise ValueError("query encoder differs from index encoder")
    if digest_file(Path(__file__)) != manifest.get("implementation_sha256"):
        raise ValueError("indexing implementation changed; build a new index version")
    versions = {name: importlib.metadata.version(name) for name in manifest["dependencies"]}
    if versions != manifest["dependencies"]:
        raise ValueError("embedding dependencies changed; use the recorded environment or rebuild index")
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    vectors = np.load(index_dir / "vectors.npy", allow_pickle=False)
    return chunks, vectors, manifest

