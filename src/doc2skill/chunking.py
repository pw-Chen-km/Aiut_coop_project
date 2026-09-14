"""Docling HybridChunker with exact canonical spans and full embedding budgets."""
from __future__ import annotations

import re
from typing import Any

from .structure import stable_id


class ChunkingError(RuntimeError):
    pass


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def embedding_text(path: list[str], text: str) -> str:
    heading = " > ".join(path)
    return f"passage: {heading}\n{text}"


def _normalised_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars, offsets = [], []
    for match in re.finditer(r"\s+|\S", text):
        chars.append(" " if match[0].isspace() else match[0])
        offsets.append((match.start(), match.end()))
    return "".join(chars), offsets


def align_segments(canonical: str, segments: list[str]) -> list[tuple[int, int]]:
    """Resolve HybridChunker whitespace normalization without fabricating offsets."""
    normalized, offsets = _normalised_offsets(canonical)
    cursor, result = 0, []
    for segment in segments:
        needle = re.sub(r"\s+", " ", segment).strip()
        if not needle:
            continue
        start = normalized.find(needle, cursor)
        if start < 0 or normalized[cursor:start].strip():
            raise ChunkingError("HybridChunker text cannot be exactly aligned to canonical source; refusing guessed spans")
        end = start + len(needle)
        result.append((offsets[start][0], offsets[end - 1][1]))
        cursor = end
    if normalized[cursor:].strip():
        raise ChunkingError("HybridChunker omitted canonical source text")
    return result


def budget_spans(text: str, path: list[str], tokenizer: Any, max_tokens: int) -> list[tuple[int, int]]:
    """Split without tokenizer.decode or truncation; all source characters survive."""
    if token_count(tokenizer, embedding_text(path, "")) >= max_tokens:
        raise ChunkingError("Section heading/passage prefix exhausts the embedding token budget")
    result, start = [], 0
    while start < len(text):
        if token_count(tokenizer, embedding_text(path, text[start:])) <= max_tokens:
            result.append((start, len(text)))
            break
        low, high, best = start + 1, len(text), start
        while low <= high:
            mid = (low + high) // 2
            if token_count(tokenizer, embedding_text(path, text[start:mid])) <= max_tokens:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        if best == start:
            raise ChunkingError("A source character cannot fit the full embedding input budget")
        # Prefer a nearby whitespace boundary, but never remove any characters.
        boundaries = list(re.finditer(r"\s+", text[start:best]))
        if boundaries and boundaries[-1].end() > (best - start) // 2:
            best = start + boundaries[-1].end()
        while best > start and token_count(tokenizer, embedding_text(path, text[start:best])) > max_tokens:
            best -= 1
        if best == start:
            raise ChunkingError("Cannot make a bounded exact source span")
        result.append((start, best))
        start = best
    return result


def _hybrid_segments(blocks: list[dict], tokenizer: Any, max_tokens: int) -> list[str]:
    try:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
        from docling_core.types.doc import DoclingDocument, DocItemLabel
    except ImportError as exc:
        raise ChunkingError("Docling HybridChunker is required; install docling-core[chunking]. No fallback chunker is used.") from exc

    class CanonicalTokenizer(BaseTokenizer):
        backend: Any
        limit: int

        def count_tokens(self, text: str) -> int:
            return len(self.backend.encode(text, add_special_tokens=False))

        def get_max_tokens(self) -> int:
            return self.limit

        def get_tokenizer(self) -> Any:
            return self.backend

    projection = DoclingDocument(name="canonical-section-projection")
    for block in blocks:
        # Plain canonical text intentionally avoids table triplet serialization,
        # which cannot be treated as a character-exact source span.
        projection.add_text(label=DocItemLabel.TEXT, text=block["text"])
    chunker = HybridChunker(tokenizer=CanonicalTokenizer(backend=tokenizer, limit=max_tokens), merge_peers=True)
    return [chunk.text for chunk in chunker.chunk(dl_doc=projection)]


def build_chunks(docling_document: Any, fragment: dict, config: dict, tokenizer: Any) -> list[dict]:
    """Chunk a per-section canonical Docling projection, never cross sections.

    ``docling_document`` is retained in the interface to emphasize the genuine
    parser origin; raw table serialization is not used for evidence mapping.
    """
    options = config.get("chunking", config)
    max_tokens = int(options.get("max_tokens", 384))
    model_limit = getattr(tokenizer, "model_max_length", max_tokens)
    if isinstance(model_limit, int) and 0 < model_limit < max_tokens:
        raise ChunkingError(f"Configured budget {max_tokens} exceeds tokenizer model limit {model_limit}")
    if max_tokens <= 0:
        raise ChunkingError("max_tokens must be positive")
    tokenizer_identity = {
        "name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                    or config.get("embedding", {}).get("revision"),
        "configured_model": config.get("embedding", {}).get("model"),
    }
    by_id = {block["block_id"]: block for block in fragment["blocks"]}
    chunks = []
    for section in fragment["sections"]:
        blocks = [by_id[bid] for bid in section["block_ids"]
                  if by_id[bid]["text"].strip() and by_id[bid].get("build_role", "content") == "content"]
        if not blocks:
            continue
        path = section["path"]
        prefix_tokens = token_count(tokenizer, embedding_text(path, ""))
        body_budget = max_tokens - prefix_tokens
        if body_budget <= 0:
            raise ChunkingError(f"Section {section['section_id']} path exhausts max_tokens")
        segments = _hybrid_segments(blocks, tokenizer, body_budget)
        canonical = "\n".join(block["text"] for block in blocks)
        offsets, cursor = [], 0
        for block in blocks:
            offsets.append((cursor, cursor + len(block["text"]), block))
            cursor += len(block["text"]) + 1
        for h_start, h_end in align_segments(canonical, segments):
            for local_start, local_end in budget_spans(canonical[h_start:h_end], path, tokenizer, max_tokens):
                start, end = h_start + local_start, h_start + local_end
                text = canonical[start:end]
                source_spans = []
                for b_start, b_end, block in offsets:
                    left, right = max(start, b_start), min(end, b_end)
                    if left < right:
                        source_spans.append({"block_id": block["block_id"], "page_index": block["page_index"],
                                             "source_start_char": left - b_start, "source_end_char": right - b_start,
                                             "chunk_start_char": left - start, "chunk_end_char": right - start})
                        if block.get("source_type") == "html":
                            source_spans[-1].update(source_type="html",
                                doc_start_char=block["doc_start_char"] + left - b_start,
                                doc_end_char=block["doc_start_char"] + right - b_start)
                embedded = embedding_text(path, text)
                count = token_count(tokenizer, embedded)
                if count > max_tokens:
                    raise ChunkingError("Full embedding input exceeded budget after splitting")
                chunks.append({"chunk_id": stable_id("chk", "hybrid-canonical-v1", section["section_id"],
                                                     max_tokens, tokenizer_identity,
                                                     source_spans, text),
                               "doc_id": section["doc_id"], "section_id": section["section_id"],
                               "text": text, "embedding_text": embedded, "token_count": count,
                               "source_spans": source_spans, "chunker": "docling-hybrid-canonical-v1",
                               "max_tokens": max_tokens, "tokenizer": tokenizer_identity})
    return chunks
