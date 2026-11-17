"""Explicitly invoked experiment; default CLI only validates and prints a plan."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import random
import time
import uuid

from topic_rag.evaluation import evidence_metrics, summarize
from topic_rag.generation import AnswerGenerator
from topic_rag.indexing import E5Encoder, load_index
from topic_rag.io import digest_file, read_jsonl, write_json, write_jsonl
from topic_rag.model_client import ModelClient
from topic_rag.retrieval import Retriever
from topic_rag.taxonomy import Taxonomy, validate_labels
from topic_rag.topic_scoring import JevScorer
from topic_rag.validation import validate_dataset


def plan(config):
    validation = validate_dataset(config)
    return {"status": "prepared_not_run", "validation": validation,
            "variants": config["variants"], "candidate_k": 20, "answer_final_k": 5,
            "recall_at": [5, 10, 20], "soft_weights": [1 / 3] * 3,
            "requires_explicit_execute": True,
            "index_present": (Path(config["index_dir"]) / "manifest.json").is_file(),
            "llm_requests_sent": 0}


def run(config, skip_answer=False):
    validation = validate_dataset(config)
    dataset = Path(config["dataset_dir"])
    chunks, vectors, index_manifest = load_index(Path(config["index_dir"]), dataset / "chunks.jsonl", config["e5_model_dir"])
    taxonomy = Taxonomy.load(config["taxonomy"])
    labels = validate_labels(read_jsonl(dataset / "sources.jsonl"), read_jsonl(config["labels"]), taxonomy)
    retriever = Retriever(chunks, vectors, labels, taxonomy)
    encoder = E5Encoder(config["e5_model_dir"])
    client = ModelClient(config["model_endpoint"], config["timeout_seconds"])
    rows = []
    try:
        identity = client.identity()
        expected = config["expected_model_marker"].casefold()
        if expected not in (identity["id"] + " " + str(identity.get("model_path", ""))).casefold():
            raise ValueError("server model identity does not match expected_model_marker")
        scorer = JevScorer(client, taxonomy, Path(config["jev_prompt"]).read_text(),
                           config["n_probs"], config["max_n_probs"])
        generator = None if skip_answer else AnswerGenerator(
            client, identity["id"], Path(config["answer_prompt"]).read_text(),
            config["context_tokens"], config["answer_max_tokens"])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        output = Path(config["results_dir"]) / run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
        manifest = {"status": "running", "identity": identity, "data": validation,
                    "index": index_manifest, "skip_answer": skip_answer,
                    "shared_topic_scores": True, "seed": config["seed"],
                    "implementation": {str(p.relative_to(Path(__file__).parent)): digest_file(p)
                                       for p in sorted(Path(__file__).parent.rglob("*.py"))}}
        write_json(output / "manifest.json", manifest)
        rng = random.Random(config["seed"])
        try:
            for question in read_jsonl(dataset / "questions.jsonl"):
                if client.identity() != identity:
                    raise ValueError("served model changed during experiment")
                started = time.perf_counter()
                vector = encoder.query(question["question"])
                embedding_ms = (time.perf_counter() - started) * 1000
                # One distribution shared by hard and soft; charge its measured
                # latency to each method's hypothetical standalone request.
                decision, decision_error = None, None
                try:
                    decision = scorer.score(question["question"])
                except Exception as exc:
                    decision_error = f"{type(exc).__name__}: {exc}"
                with (output / "topic_scores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"question_id": question["question_id"],
                                             "decision": decision, "error": decision_error}) + "\n")
                variants = list(config["variants"])
                rng.shuffle(variants)
                for variant in variants:
                    row = {**question, "variant": variant, "model": identity,
                           "answer_final_k": 5, "candidate_k": 20, "error": None}
                    try:
                        if variant != "baseline" and decision_error:
                            raise ValueError(decision_error)
                        start = time.perf_counter()
                        results, trace = retriever.retrieve(
                            question["question"], vector, variant,
                            decision["probabilities"] if decision else None, candidate_k=20, output_k=20)
                        retrieval_ms = (time.perf_counter() - start) * 1000
                        # Preserve retrieval evidence even if answer generation fails.
                        row.update({"answer_context": results[:5], "retrieved_chunks": results,
                                    "evidence": evidence_metrics(results, question["gold_evidence"]),
                                    "retrieval_trace": trace,
                                    "topic_probabilities": decision["probabilities"] if variant != "baseline" else None})
                        start = time.perf_counter()
                        generated = generator.generate(question["question"], results[:5]) if generator else {
                            "response_type": "not_run", "answer": None, "sources": []}
                        answer_ms = (time.perf_counter() - start) * 1000
                        topic_ms = decision["elapsed_ms"] if variant != "baseline" else 0.0
                        row.update({"generated_answer": generated["answer"],
                                    "response_type": generated["response_type"], "cited_sources": generated["sources"],
                                    "answer_context": results[:5], "retrieved_chunks": results,
                                    "evidence": evidence_metrics(results, question["gold_evidence"]),
                                    "retrieval_trace": trace,
                                    "topic_probabilities": decision["probabilities"] if variant != "baseline" else None,
                                    "latency_ms": {"query_embedding": embedding_ms, "topic_scoring": topic_ms,
                                                   "retrieval": retrieval_ms, "answer": answer_ms,
                                                   "total": embedding_ms + topic_ms + retrieval_ms + answer_ms},
                                    "answer_review": {"score": None, "reason": None}})
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["response_type"] = "error"
                    rows.append(row)
                    with (output / f"{variant}.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"completed {question['question_id']}", flush=True)
            if client.identity() != identity:
                raise ValueError("served model changed during experiment")
            manifest["status"] = "complete_with_errors" if any(r["error"] for r in rows) else "complete"
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(output / "manifest.json", manifest)
            summaries = summarize(rows)
            write_json(output / "summary.json", summaries)
            write_jsonl(output / "answer_review.jsonl", [
                {"question_id": r["question_id"], "variant": r["variant"],
                 "question": r["question"], "reference_responses": r["reference_responses"],
                 "gold_evidence": r["gold_evidence"], "generated_answer": r.get("generated_answer"),
                 "error": r.get("error"), "score": None, "reason": None}
                for r in rows])
            if summaries:
                with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
        return output
    finally:
        client.close()

