"""Blind, append-only evaluation of frozen baseline answers with the AMD judge.

The online answers and their retrieval traces are immutable inputs.  Candidate
method names, traces and retrieval metadata are never included in the judge
prompt.  Exact lexical metrics and exact oracle-backed evidence metrics are
computed locally; only semantic answer correctness uses the LLM.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

from doc2skill.config import fingerprint, sha256, read_jsonl, write_json
from qa_skillopt.data import digest
from qa_skillopt.io import digest_text
from qa_skillopt.judge import ANSWER_RUBRIC, AnswerJudge, JudgmentUnavailable
from qa_skillopt.transport import OLLAMA_DIGEST, OLLAMA_MODEL, OllamaTransport, TransportError

from qa_experiments import METHODS


SCHEMA_VERSION = "qa-experiments-amd-judge-v2"
JUDGE_POLICY = {
    "candidate_order": "qid-major; per-qid method order sorted by SHA256(qid|method)",
    "candidate_status_not_ok": "hard=0 without LLM call",
    "judge_uncertain_or_invalid_schema": "unscored; never retried or converted to zero",
    "transport_failure": "stop and resume the exact same frozen run later",
    "judge_view": "question, generated answer, reference answer, claims, original evidence only",
    "attempts_per_candidate": 1,
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def answer_references(answer):
    if isinstance(answer, str):
        return [answer]
    if not isinstance(answer, dict) or not isinstance(answer.get("canonical"), str):
        raise ValueError("Reference answer requires a canonical string")
    values = [answer["canonical"]]
    aliases = answer.get("aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(x, str) for x in aliases):
        raise ValueError("Reference aliases must be strings")
    return list(dict.fromkeys(values + aliases))


def answer_tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE)


def lexical_scores(candidate, answer):
    if not isinstance(candidate, str) or not candidate.strip():
        return {"normalized_exact_match": 0.0, "token_f1": 0.0}
    predicted = answer_tokens(candidate)
    exact, best = 0.0, 0.0
    for reference in answer_references(answer):
        expected = answer_tokens(reference)
        exact = max(exact, float(predicted == expected))
        if not predicted or not expected:
            f1 = float(predicted == expected)
        else:
            common = Counter(predicted) & Counter(expected)
            overlap = sum(common.values())
            precision, recall = overlap / len(predicted), overlap / len(expected)
            f1 = 0.0 if not overlap else 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return {"normalized_exact_match": exact, "token_f1": best}


def _context_chunk_ids(result):
    identifiers = []
    for item in result.get("context_items", []):
        if isinstance(item.get("chunk_id"), str):
            identifiers.append(item["chunk_id"])
        for chunk_id in item.get("chunk_ids", []):
            if isinstance(chunk_id, str):
                identifiers.append(chunk_id)
    return list(dict.fromkeys(identifiers))


def _oracle_route_chunks(query):
    chunks = {}
    for evidence in query.get("evidence", []):
        key = evidence.get("equivalence_group") or evidence.get("evidence_id")
        if not key:
            continue
        chunks.setdefault(key, set())
        for target in evidence.get("targets", []):
            chunks[key].update(x for x in target.get("chunk_ids", []) if isinstance(x, str))
    return chunks


def route_coverage(query, chunk_ids):
    """Best exact gold route coverage; only valid for a complete oracle query."""
    if not query.get("complete_route_available"):
        return None
    selected, by_key = set(chunk_ids), _oracle_route_chunks(query)
    outcomes = []
    for route in query.get("routes", []):
        required = route.get("required_keys", [])
        if required and all(by_key.get(key) for key in required):
            hits = sum(bool(selected & by_key[key]) for key in required)
            outcomes.append((hits / len(required), hits == len(required)))
    if not outcomes:
        raise ValueError("Complete oracle query has no scoreable route")
    return {"recall": max(x[0] for x in outcomes), "complete": any(x[1] for x in outcomes)}


def retrieval_scores(result, query, ks=(1, 3, 5, 10, 20)):
    ranked = list(dict.fromkeys(x for x in result.get("retrieved_chunk_ids", []) if isinstance(x, str)))
    output = {"oracle_status": query.get("status"), "oracle_scoreable": bool(query.get("complete_route_available")),
              "retrieved_count": len(ranked), "context_chunk_count": len(_context_chunk_ids(result))}
    if not query.get("complete_route_available"):
        return output
    output["candidate"] = {str(k): route_coverage(query, ranked[:k]) for k in ks}
    output["context"] = route_coverage(query, _context_chunk_ids(result))
    return output


def _validate_inputs(experiment, prepared):
    experiment, prepared = Path(experiment).resolve(), Path(prepared).resolve()
    frozen = _read(experiment / "frozen.json")
    body = dict(frozen)
    if fingerprint({k: v for k, v in body.items() if k != "experiment_id"}) != frozen.get("experiment_id"):
        raise ValueError("Frozen experiment identity changed")
    methods = frozen.get("active_methods", list(METHODS))
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Invalid frozen active methods")
    job = _read(experiment / "job_status.json")
    phase_manifest = _read(prepared / "test_manifest.json")
    dataset_path = prepared / "test.jsonl"
    if phase_manifest.get("sha256") != sha256(dataset_path):
        raise ValueError("Test payload hash changed")
    qa = read_jsonl(dataset_path)
    if not qa or len({x["qid"] for x in qa}) != len(qa) or {x["qid"] for x in qa} != set(phase_manifest.get("qids", [])):
        raise ValueError("Test payload differs from its qid manifest")
    if (job.get("status") != "inference_complete_unscored" or job.get("records") != len(qa) * len(methods)
            or job.get("experiment_id") != frozen["experiment_id"]):
        raise ValueError("Inference experiment is not complete and unscored")
    questions = {x["qid"]: x["question"] for x in qa}
    results, hashes = {}, {}
    for method in methods:
        path = experiment / "runs" / "test" / method / "answers.jsonl"
        run_manifest = _read(path.parent / "run_manifest.json")
        run_id = fingerprint({"identity": frozen["experiment_id"], "method": method, "phase": "test"})
        if (run_manifest.get("run_id") != run_id or run_manifest.get("identity") != frozen["experiment_id"]
                or run_manifest.get("method") != method or run_manifest.get("phase") != "test"
                or run_manifest.get("count") != len(qa)):
            raise ValueError("Results do not belong to the frozen run: " + method)
        rows = read_jsonl(path)
        if any(x.get("run_id") != run_id for x in rows):
            raise ValueError("Mixed result run identity: " + method)
        if len(rows) != len(qa) or len({x["qid"] for x in rows}) != len(qa):
            raise ValueError("Incomplete or duplicate results: " + method)
        if {x["qid"]: x["question"] for x in rows} != questions:
            raise ValueError("Method questions differ from sealed test: " + method)
        results[method] = {x["qid"]: x for x in rows}
        hashes[method] = sha256(path)
    oracle_path = prepared / "oracle_test.json"
    oracle = _read(oracle_path)
    if oracle.get("phase") != "test" or oracle.get("preparation_id") != phase_manifest.get("preparation_id"):
        raise ValueError("Test oracle identity mismatch")
    if phase_manifest.get("oracle_file_sha256") != sha256(oracle_path):
        raise ValueError("Test oracle file hash changed")
    # prepare() appends phase/preparation_id after build_oracle() seals its body;
    # the phase manifest above independently seals the complete on-disk file.
    embedded = {k: v for k, v in oracle.items() if k not in {"oracle_sha256", "phase", "preparation_id"}}
    if oracle.get("oracle_sha256") != digest(embedded):
        raise ValueError("Test oracle hash changed")
    if set(oracle.get("queries", {})) != set(questions):
        raise ValueError("Oracle qids differ from test qids")
    return qa, results, oracle, {
        "experiment_id": frozen["experiment_id"], "preparation_id": phase_manifest["preparation_id"],
        "methods": methods, "question_count": len(qa), "frozen_sha256": sha256(experiment / "frozen.json"),
        "run_manifests_sha256": {m: sha256(experiment / "runs/test" / m / "run_manifest.json") for m in methods},
        "test_sha256": sha256(dataset_path), "test_manifest_sha256": sha256(prepared / "test_manifest.json"),
        "oracle_sha256": sha256(oracle_path), "answer_files_sha256": hashes,
    }


def _manifest(identity):
    value = {"schema_version": SCHEMA_VERSION, **identity,
             "model": OLLAMA_MODEL, "model_digest": OLLAMA_DIGEST,
             "answer_rubric_sha256": digest_text(ANSWER_RUBRIC), "policy": JUDGE_POLICY,
             "human_reviewed": False, "independent_model": True,
             "note": "The judge is independent from the online Qwen 4B generator, but remains an automatic model judge."}
    value["evaluation_id"] = fingerprint(value)
    return value


def _task_order(qa, methods=METHODS):
    tasks = []
    for item in sorted(qa, key=lambda x: x["qid"]):
        ordered = sorted(methods, key=lambda m: hashlib.sha256((item["qid"] + "|" + m).encode()).hexdigest())
        tasks.extend((method, item) for method in ordered)
    return tasks


def run_judge(experiment, prepared, output, *, usage_window_confirmed=False, transport=None):
    experiment, prepared, output = map(lambda x: Path(x).resolve(), (experiment, prepared, output))
    if output == experiment or output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError("Judge output must not overlap the immutable inference experiment")
    if output == prepared or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError("Judge output must not overlap the immutable prepared dataset")
    qa, results, oracle, identity = _validate_inputs(experiment, prepared)
    expected_manifest = _manifest(identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if _read(manifest_path) != expected_manifest:
            raise ValueError("Resume requires the identical judge/input/rubric identity")
    else:
        write_json(manifest_path, expected_manifest)
    rows_path = output / "judgments.jsonl"
    prior = read_jsonl(rows_path) if rows_path.exists() else []
    if any(x.get("evaluation_id") != expected_manifest["evaluation_id"] for x in prior):
        raise ValueError("Judgment file mixes evaluation versions")
    done = {(x["method"], x["qid"]) for x in prior}
    if len(done) != len(prior):
        raise ValueError("Duplicate judgment rows")
    if not done <= {(m, q["qid"]) for m in identity["methods"] for q in qa}:
        raise ValueError("Unknown method/qid in judgments")
    transport = transport or OllamaTransport(usage_window_confirmed=usage_window_confirmed,
                                               audit_dir=output / "offline_calls", timeout=600)
    if not usage_window_confirmed and transport.__class__ is OllamaTransport:
        raise ValueError("AMD usage window must be explicitly confirmed")
    preflight = transport.preflight() if hasattr(transport, "preflight") else {"model": transport.model}
    write_json(output / "preflight.json", preflight) if not (output / "preflight.json").exists() else None
    judge = AnswerJudge(transport)
    with (output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, (method, item) in enumerate(_task_order(qa, identity["methods"]), 1):
            key = method, item["qid"]
            if key in done:
                continue
            prediction = results[method][item["qid"]]
            lexical = lexical_scores(prediction.get("answer"), item["answer"])
            record = {"schema_version": SCHEMA_VERSION, "evaluation_id": expected_manifest["evaluation_id"],
                      "method": method, "qid": item["qid"], "question_type": item["question_type"],
                      "visual_diagnostic": bool(item.get("visual_diagnostic")),
                      "candidate_status": prediction.get("status"),
                      "candidate_sha256": digest({"qid": item["qid"], "answer": prediction.get("answer"),
                                                   "status": prediction.get("status")}),
                      "lexical": lexical,
                      "retrieval": retrieval_scores(prediction, oracle["queries"][item["qid"]]),
                      "human_reviewed": False}
            if prediction.get("status") != "ok" or not isinstance(prediction.get("answer"), str) or not prediction["answer"].strip():
                record.update(judge_status="not_called_invalid_candidate", hard=0, verdict="invalid_candidate",
                              reason="The frozen online run did not produce a complete candidate answer.", judge_called=False)
            else:
                try:
                    judgment = judge.score(item, {"status": "ok", "answer": prediction["answer"]})
                    record.update(judge_status="scored", hard=judgment["hard"], verdict=judgment["verdict"],
                                  complete=judgment["complete"], contradiction=judgment["contradiction"],
                                  reference_consistent=judgment["reference_consistent"], reason=judgment["reason"],
                                  provenance=judgment["provenance"], judge_called=True)
                except JudgmentUnavailable as exc:
                    record.update(judge_status="unscored", hard=None, verdict="unscored", reason=str(exc), judge_called=True)
                except TransportError:
                    raise
            _write_jsonl_record(rows_path, record)
            done.add(key)
            print(json.dumps({"completed": len(done), "total": len(qa) * len(results), "qid": item["qid"],
                              "candidate": index, "status": record["judge_status"]}), flush=True)
    # Detect any mutation that occurred while judging before reporting success.
    _, _, _, final_identity = _validate_inputs(experiment, prepared)
    if final_identity != identity:
        raise ValueError("Frozen inference or dataset inputs changed during judging")
    report = summarize_judgments(experiment, prepared, output)
    write_json(output / "job_status.json", {"status": "judge_complete", "records": len(done),
               "evaluation_id": expected_manifest["evaluation_id"], "judge_calls": report["judge_calls"],
               "finished_unix": time.time()})
    return report


def _mean(values):
    return sum(values) / len(values) if values else None


def _wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    p, d = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _aggregate(rows, qa_by_id):
    scored = [x for x in rows if x["hard"] is not None]
    correct = sum(x["hard"] for x in scored)
    by_type = {}
    for question_type in sorted({x["question_type"] for x in rows}):
        subset = [x for x in rows if x["question_type"] == question_type]
        eligible = [x for x in subset if x["hard"] is not None]
        by_type[question_type] = {"records": len(subset), "scored": len(eligible),
                                  "correct": sum(x["hard"] for x in eligible),
                                  "accuracy": _mean([x["hard"] for x in eligible])}
    scoreable = [x for x in rows if x["retrieval"]["oracle_scoreable"]]
    retrieval = {"denominator": len(scoreable), "oracle_unscoreable": len(rows) - len(scoreable)}
    for k in (1, 3, 5, 10, 20):
        retrieval[f"EvidenceSetRecall@{k}"] = _mean([x["retrieval"]["candidate"][str(k)]["recall"] for x in scoreable])
        retrieval[f"CompleteEvidence@All-Hops@{k}"] = _mean([float(x["retrieval"]["candidate"][str(k)]["complete"]) for x in scoreable])
    retrieval["ContextEvidenceSetRecall"] = _mean([x["retrieval"]["context"]["recall"] for x in scoreable])
    retrieval["ContextCompleteEvidence@All-Hops"] = _mean([float(x["retrieval"]["context"]["complete"]) for x in scoreable])
    multihop = [x for x in scoreable if x["question_type"] == "multihop"]
    retrieval["multihop_denominator"] = len(multihop)
    retrieval["MultiHopCompleteEvidence@20"] = _mean([float(x["retrieval"]["candidate"]["20"]["complete"]) for x in multihop])
    joint = [x for x in scoreable if x["hard"] is not None]
    retrieval["JointSuccess@20_denominator"] = len(joint)
    retrieval["JointSuccess@20"] = _mean([float(x["hard"] == 1 and x["retrieval"]["candidate"]["20"]["complete"]) for x in joint])
    retrieval["ContextJointSuccess_denominator"] = len(joint)
    retrieval["ContextJointSuccess"] = _mean([float(x["hard"] == 1 and x["retrieval"]["context"]["complete"]) for x in joint])
    return {"records": len(rows), "judge_calls": sum(bool(x["judge_called"]) for x in rows),
            "judge_statuses": dict(Counter(x["judge_status"] for x in rows)),
            "scored": len(scored), "correct": correct, "accuracy": _mean([x["hard"] for x in scored]),
            "accuracy_wilson95": _wilson(correct, len(scored)),
            "coverage": len(scored) / len(rows) if rows else None,
            "correct_over_observed_lower_bound": correct / len(rows) if rows else None,
            "normalized_exact_match": _mean([x["lexical"]["normalized_exact_match"] for x in rows]),
            "token_f1": _mean([x["lexical"]["token_f1"] for x in rows]),
            "by_question_type": by_type, "retrieval": retrieval}


def summarize_judgments(experiment, prepared, output):
    experiment, prepared, output = map(lambda x: Path(x).resolve(), (experiment, prepared, output))
    qa, results, oracle, identity = _validate_inputs(experiment, prepared)
    manifest = _read(output / "manifest.json")
    if manifest != _manifest(identity):
        raise ValueError("Judge manifest does not match immutable inputs")
    rows = read_jsonl(output / "judgments.jsonl") if (output / "judgments.jsonl").exists() else []
    if len({(x["method"], x["qid"]) for x in rows}) != len(rows):
        raise ValueError("Duplicate judgment rows")
    qa_by_id = {x["qid"]: x for x in qa}
    if any(x.get("evaluation_id") != manifest["evaluation_id"] or x.get("method") not in results
           or x.get("qid") not in qa_by_id for x in rows):
        raise ValueError("Unknown or mixed evaluation record")
    methods = {method: _aggregate([x for x in rows if x["method"] == method], qa_by_id)
               for method in results}
    by_key = {(x["method"], x["qid"]): x for x in rows}
    pairwise = {}
    for left in results:
        for right in results:
            if left >= right:
                continue
            pairs = [(by_key.get((left, qid)), by_key.get((right, qid))) for qid in qa_by_id]
            pairs = [(a, b) for a, b in pairs if a and b and a["hard"] is not None and b["hard"] is not None]
            pairwise[left + "__vs__" + right] = {"denominator": len(pairs),
                "left_wins": sum(a["hard"] > b["hard"] for a, b in pairs),
                "ties": sum(a["hard"] == b["hard"] for a, b in pairs),
                "right_wins": sum(a["hard"] < b["hard"] for a, b in pairs)}
    report = {"schema_version": SCHEMA_VERSION, "evaluation_id": manifest["evaluation_id"],
              "label": "prototype_independent_amd_qwen35b_llm_as_judge_not_sme",
              "records": len(rows), "expected_records": len(qa) * len(results),
              "judge_calls": sum(bool(x["judge_called"]) for x in rows),
              "invalid_candidates_not_called": sum(x["judge_status"] == "not_called_invalid_candidate" for x in rows),
              "oracle": oracle["summary"], "methods": methods, "pairwise": pairwise,
              "limitations": [
                  "Automatic LLM-as-judge labels are not SME approval or human ground truth.",
                  "Evidence metrics exclude queries without a complete exact oracle; see the per-run denominators.",
                  "A-RAG retrieval IDs are ordered exposure events, not a conventional one-shot ranked list.",
                  "Visual-diagnostic questions are judged from textual reference material; no image is shown to the judge.",
              ]}
    write_json(output / "summary.json", report)
    return report
