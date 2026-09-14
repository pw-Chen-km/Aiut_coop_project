"""Deployable local QA commands. Batch input is projected to qid/question only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .bundle import export_bundle, validate_serving_bundle
from .factory import create_session, load_online_config, model_identity


def _parser():
    parser = argparse.ArgumentParser(prog="qa-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    comparison = sub.add_parser('compare', help='Inspect controlled trees and optionally evaluate common QA')
    comparison.add_argument('--source', required=True)
    comparison.add_argument('--llm', required=True)
    comparison.add_argument('--corpus2skill', required=True)
    comparison.add_argument('--output-dir', required=True)
    comparison.add_argument('--qa')
    comparison.add_argument('--config')
    export = sub.add_parser("export-bundle", help="Offline: export a completed build for private transfer")
    export.add_argument("--artifact-dir", required=True)
    export.add_argument("--out-dir", required=True)
    check = sub.add_parser("validate-bundle", help="Check a serving bundle without PDFs or models")
    check.add_argument("--bundle-dir", required=True)
    info = sub.add_parser("model-info", help="Read local GGUF/tokenizer hashes for configuration")
    info.add_argument("--gguf", required=True)
    info.add_argument("--tokenizer", required=True)
    for name in ("ask", "batch", "doctor", "prepare-embedding"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "ask":
            command.add_argument("question")
            command.add_argument("--output", help="New JSON file; omit to print answer and citations")
        if name == "batch":
            command.add_argument("--input", required=True, help="JSONL with qid and question (QA files supported)")
            command.add_argument("--output", required=True, help="New private run directory")
            command.add_argument("--limit", type=int, default=0, help="0 means all; use dev input for tuning")
    return parser


def read_questions(path, limit=0):
    if limit < 0:
        raise ValueError("limit must not be negative")
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            qid, question = value.get("qid"), value.get("question")
            if not isinstance(qid, str) or not qid.strip() or qid in seen:
                raise ValueError("Every question needs a unique nonempty qid")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Every question must be nonempty text")
            seen.add(qid)
            rows.append({"qid": qid, "question": question})
            if "history" in value:
                from .dialogue import validate_history
                rows[-1]["history"] = validate_history(value["history"])
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Input contains no questions")
    return rows


def answer_record(result):
    """Compact output for future answer evaluation; gold labels are never copied."""
    keys = ("schema_version", "qid", "question", "status", "answer", "answerable", "citations",
            "scope", "retrieval_scopes", "rounds_used", "max_rounds", "stop_reason",
            "missing_information", "error", "elapsed_seconds", "response_type", "retrieval_query")
    return {key: result[key] for key in keys if key in result}


def _new_output(path, config):
    target = Path(path).resolve()
    bundle = Path(config["bundle_dir"]).resolve()
    if target == bundle or target.is_relative_to(bundle) or bundle.is_relative_to(target):
        raise ValueError("Run output must be separate from the immutable serving bundle")
    if target.exists():
        raise FileExistsError("Output exists; choose a new run path")
    return target


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == 'compare':
            from .comparison import compare_bundles
            from doc2skill.config import load_config
            result = compare_bundles({k: getattr(args, k) for k in ('source', 'llm', 'corpus2skill')},
                args.output_dir, qa_config=load_config(args.config) if args.config else None, qa_path=args.qa)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['all_three_accepted'] and result.get('all_three_qa_completed', True) else 1
        if args.command == "export-bundle":
            manifest = export_bundle(args.artifact_dir, args.out_dir)
            print(json.dumps({"status": "ok", "private_bundle": args.out_dir, "counts": manifest["counts"]}))
            return 0
        if args.command == "validate-bundle":
            manifest = validate_serving_bundle(args.bundle_dir)
            print(json.dumps({"status": "ok", "counts": manifest["counts"], "embedding": manifest["embedding"]}))
            return 0
        if args.command == "model-info":
            print(json.dumps(model_identity(args.gguf, args.tokenizer), indent=2))
            return 0
        config = load_online_config(args.config)
        if args.command == "prepare-embedding":
            # Explicit offline-preparation action; public model weights only, never corpus uploads.
            from doc2skill.embedding import E5Encoder
            bundle = validate_serving_bundle(config["bundle_dir"])
            encoder = E5Encoder({**config["embedding"], "model": bundle["embedding"]["model"],
                                 "revision": bundle["embedding"]["revision"]}, local_only=False)
            print(json.dumps({"status": "ok", "embedding": encoder.provenance}))
            return 0
        output = _new_output(args.output, config) if getattr(args, "output", None) else None
        questions = read_questions(args.input, args.limit) if args.command == "batch" else None
        with create_session(config) as session:
            if args.command == "doctor":
                print(json.dumps({"status": "ready", "model_inference_tested": False,
                                  "note": "Local files/config checked; llama-server readiness requires ask",
                                  "manifest": session.manifest}, indent=2))
                return 0
            if args.command == "ask":
                result = session.agent.answer(args.question)
                if output:
                    _write_new(output, {**result, "run_manifest": session.manifest})
                print(json.dumps(answer_record(result), ensure_ascii=False, indent=2))
                return 0 if result["status"] == "ok" else 2
            output.mkdir(parents=True, exist_ok=False)
            started, failed, answered, abstained = time.monotonic(), 0, 0, 0
            rechecked, recovered, clarified = 0, 0, 0
            _write_new(output / "run_manifest.json", session.manifest)
            with (output / "answers.jsonl").open("x", encoding="utf-8") as answers, (output / "traces.jsonl").open("x", encoding="utf-8") as traces:
                for index, row in enumerate(questions, 1):
                    result = session.agent.answer(row["question"], qid=row["qid"],
                                                  **({"history": row["history"]} if "history" in row else {}))
                    failed += result["status"] != "ok"
                    answered += result["status"] == "ok" and result["answerable"] is True
                    clarified += result["status"] == "ok" and result.get("response_type") == "clarify"
                    abstained += result["status"] == "ok" and result["answerable"] is False and result.get("response_type") != "clarify"
                    rechecked += result.get("rounds_used", 1) > 1
                    recovered += (result.get("rounds_used", 1) > 1 and result["status"] == "ok"
                                  and result["answerable"] is True)
                    answers.write(json.dumps(answer_record(result), ensure_ascii=False) + "\n")
                    traces.write(json.dumps(result, ensure_ascii=False) + "\n")
                    answers.flush()
                    traces.flush()
                    print(f"{index}/{len(questions)} {row['qid']}: {result['status']}", file=sys.stderr, flush=True)
            summary = {"status": "completed", "total": len(questions), "failed": failed,
                       "answered": answered, "insufficient_context": abstained, "clarifications": clarified,
                       "rechecked": rechecked, "answered_after_recheck": recovered,
                       "elapsed_seconds": time.monotonic() - started, "answer_accuracy": None,
                       "note": "Predictions only; no ground-truth grading was performed"}
            _write_new(output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 2 if failed else 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
