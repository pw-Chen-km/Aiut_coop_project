from pathlib import Path

from topic_rag.indexing import validate_chunks
from topic_rag.io import digest_file, read_jsonl
from topic_rag.taxonomy import Taxonomy, validate_labels


def validate_dataset(config):
    root = Path(config["dataset_dir"])
    sources = read_jsonl(root / "sources.jsonl")
    chunks = read_jsonl(root / "chunks.jsonl")
    questions = read_jsonl(root / "questions.jsonl")
    taxonomy = Taxonomy.load(config["taxonomy"])
    labels = validate_labels(sources, read_jsonl(config["labels"]), taxonomy)
    validate_chunks(sources, chunks)
    by_id = {s["source_id"]: s["text"] for s in sources}
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("empty or duplicate questions")
    for q in questions:
        if not q["gold_evidence"]:
            raise ValueError("question without evidence")
        for e in q["gold_evidence"]:
            if e["source_id"] not in by_id or by_id[e["source_id"]][e["start"]:e["end"]] != e["text"]:
                raise ValueError("evidence/source alignment failure")
    files = [root / name for name in ("sources.jsonl", "chunks.jsonl", "questions.jsonl", "validation.json")]
    files += [Path(config[k]) for k in ("taxonomy", "labels", "jev_prompt", "answer_prompt")]
    return {"sources": len(sources), "chunks": len(chunks), "questions": len(questions),
            "labels": len(labels), "topic_options": taxonomy.options,
            "hashes": {str(p): digest_file(p) for p in files}}

