"""RAGBench E-Manual: exact sentence-key to source-span mapping.

RAGBench has generated responses and model-annotated relevance. These are
reference responses / annotated evidence, not human QA ground truth.
"""
from collections import defaultdict

from topic_rag.io import digest_file, digest_text, read_jsonl, write_json, write_jsonl


def normalize(input_path, output_dir):
    sources, valid_rows, invalid_rows = {}, [], []
    rows = read_jsonl(input_path)
    for row in rows:
        row_id = row["id"]
        key_to_evidence = {}
        errors = []
        documents, sentence_groups = row["documents"], row["documents_sentences"]
        if len(documents) != len(sentence_groups):
            errors.append("documents_sentences_count_mismatch")
        for doc_index, text in enumerate(documents):
            source_id = "src-" + digest_text(text)[:16]
            sentences = sentence_groups[doc_index] if doc_index < len(sentence_groups) else []
            source = sources.setdefault(source_id, {
                "source_id": source_id, "text": text, "text_sha256": digest_text(text),
                "title": sentences[0][1] if sentences else "Untitled passage",
                "title_origin": "first_dataset_sentence", "section_path": [],
                "sentence_spans": [],
            })
            if source["text"] != text:
                raise ValueError("source hash collision")
            cursor = 0
            for key, sentence in sentences:
                # Do not parse a key prefix (keys are 0a, 2e, ...); use the
                # explicit documents_sentences membership supplied by the data.
                start = text.find(sentence, cursor)
                if not sentence or start < 0 or key in key_to_evidence:
                    errors.append(f"invalid_sentence:{key}")
                    continue
                end = start + len(sentence)
                cursor = end
                span = {"start": start, "end": end}
                if span not in source["sentence_spans"]:
                    source["sentence_spans"].append(span)
                key_to_evidence[key] = {
                    "evidence_id": f"{source_id}:{start}:{end}",
                    "source_id": source_id, "start": start, "end": end, "text": sentence,
                }
        gold_keys = row.get("all_relevant_sentence_keys", [])
        missing = [key for key in gold_keys if key not in key_to_evidence]
        if missing:
            errors.append("unmapped_evidence_key")
        if not gold_keys or not row.get("question", "").strip():
            errors.append("empty_question_or_evidence")
        if errors:
            invalid_rows.append({"row_id": row_id, "reasons": sorted(set(errors)), "unmapped_keys": missing})
            continue
        evidence = {key_to_evidence[key]["evidence_id"]: key_to_evidence[key] for key in gold_keys}
        valid_rows.append({
            "row_id": row_id, "question": row["question"], "gold_evidence": list(evidence.values()),
            "gold_sentence_keys": gold_keys,
            "reference": {
                "row_id": row_id, "response": row.get("response", ""),
                "generation_model": row.get("generation_model_name"),
                "annotation_model": row.get("annotating_model_name"),
                "adherence_score": row.get("adherence_score"),
                "sentence_support_information": row.get("sentence_support_information"),
                "unsupported_response_sentence_keys": row.get("unsupported_response_sentence_keys"),
            },
        })
    # Multiple generated responses to the exact same question are not independent
    # test questions. Preserve every valid reference; union annotated evidence.
    groups = defaultdict(list)
    for row in valid_rows:
        groups[row["question"]].append(row)
    questions = []
    for question, group in sorted(groups.items()):
        evidence = {e["evidence_id"]: e for row in group for e in row["gold_evidence"]}
        questions.append({
            "question_id": "q-" + digest_text(question)[:16], "question": question,
            "raw_row_ids": [row["row_id"] for row in group],
            "reference_responses": [row["reference"] for row in group],
            "reference_type": "generated_ragbench_response_not_human_gold",
            "gold_evidence": sorted(evidence.values(), key=lambda e: e["evidence_id"]),
            "gold_source_ids": sorted({e["source_id"] for e in evidence.values()}),
        })
    for source in sources.values():
        source["sentence_spans"].sort(key=lambda s: (s["start"], s["end"]))
    write_jsonl(output_dir / "sources.jsonl", sorted(sources.values(), key=lambda r: r["source_id"]))
    write_jsonl(output_dir / "questions.jsonl", questions)
    write_jsonl(output_dir / "validated_rows.jsonl", valid_rows)
    report = {
        "raw_rows": len(rows), "valid_rows": len(valid_rows), "invalid_rows": invalid_rows,
        "unique_questions": len(questions), "source_count": len(sources),
        "input_sha256": digest_file(input_path),
        "mapping": "exact_sentence_key_membership_then_verbatim_source_span",
        "corpus_scope": "union_of_supplied_test_passages_not_full_original_manuals",
        "deduplication": "identical_question_string; evidence_union; all_valid_references_retained",
    }
    write_json(output_dir / "validation.json", report)
    return report

