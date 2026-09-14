"""Persistent, isolated subprocess protocol. Input contains questions, never GT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from .transport import LocalTransport

def normalize(method, question, result, transport, elapsed):
    calls = transport.calls
    errors = [c for c in calls if c["status"] != "ok"]
    tool_errors = result.get("tool_checks", [])
    status = result.get("status", "runtime_error")
    if errors:
        status = errors[0]["status"]
    answer = result.get("answer")
    if status == "ok" and (not isinstance(answer, str) or not answer.strip()):
        status = "empty_answer"
    contexts = result.get("all_context_items", result.get("context_items", []))
    sent_messages = [m for c in calls if c.get("request_sent") for m in c["request"]["messages"]]
    if method == "arag":
        presented = {m.get("content") for m in sent_messages if m["role"] == "tool"}
        contexts = [c for c in contexts if c["text"] in presented]
    elif method in ("naive", "navigation"):
        contexts = []
        for message in sent_messages:
            if message.get("role") == "user":
                try:
                    content = json.loads(message.get("content", ""))
                    if isinstance(content, dict):
                        contexts.extend(content.get("original_passages", []))
                except (TypeError, ValueError):
                    pass
    elif not any(c.get("request_sent") for c in calls):
        contexts = []
    items = result.get("all_items", result.get("items", []))
    ids = result.get("retrieved_chunk_ids", list(dict.fromkeys(x["chunk_id"] for x in items)))
    tokens = {key: sum(c.get("usage", {}).get(key, 0) for c in calls)
              for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {"method": method, "qid": question["qid"], "question": question["question"],
            "question_type": question.get("question_type"), "visual_diagnostic": question.get("visual_diagnostic", False),
            "answer": answer if status == "ok" else None, "status": status,
            "retrieved_chunk_ids": ids, "context_items": contexts,
            "calls": calls, "tokens": tokens, "elapsed_seconds": elapsed,
            "stop_reason": result.get("stop_reason") if status == "ok" else status,
            "native_result": result, "judge_requested": False,
            "tool_error_events": tool_errors + [e for c in calls for e in c.get("tool_validation_errors", [])],
            "exposure_definition": "content submitted to Qwen; transport timeouts have uncertain processing"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Native progress never corrupts JSONL protocol.
    random.seed(42)
    import numpy as np
    import torch
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        raise RuntimeError("Baseline worker must not access CUDA; model is an external server")
    from .adapters import ExistingQA, ARAG, Linear
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "online.json").read_text())
    transport = LocalTransport(config["runtime"])
    try:
        if args.method in ("naive", "navigation"):
            adapter = ExistingQA(root, args.method, transport)
        elif args.method == "arag":
            settings = manifest.get("settings", {}).get("arag", {})
            # The preparation manifest, not an environment override, owns the budget.
            max_loops = settings.get("max_loops", 15)
            if type(max_loops) is not int or not 1 <= max_loops <= 15:
                raise ValueError("Invalid sealed A-RAG max_loops")
            adapter = ARAG(root, manifest["environments"]["arag"], transport, build=args.build, max_loops=max_loops)
        elif args.method == "linear":
            adapter = Linear(root, manifest["environments"]["linear"], transport,
                             manifest["settings"]["linear"], build=args.build)
        else:
            raise ValueError("Unknown method")
        capabilities = adapter.capability_check() if args.method == "arag" else None
        protocol.write(json.dumps({"ready": True, "capabilities": capabilities}) + "\n")
        protocol.flush()
        if not args.build:
            for line in sys.stdin:
                question = json.loads(line)
                allowed = {"qid", "question", "question_type", "persona", "visual_diagnostic"}
                if set(question) - allowed:
                    raise ValueError("Worker input may only contain question fields, not GT")
                transport.calls = []
                start = time.monotonic()
                try:
                    result = adapter.answer(question)
                except Exception as exc:
                    traceback.print_exc()
                    result = {"status": getattr(exc, "status", "runtime_error"), "answer": None,
                              "error": {"type": type(exc).__name__, "message": str(exc)}}
                response = normalize(args.method, question, result, transport, time.monotonic()-start)
                protocol.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
                protocol.flush()
        adapter.close()
    except Exception as exc:
        traceback.print_exc()
        protocol.write(json.dumps({"ready": False, "error": type(exc).__name__, "message": str(exc)}) + "\n")
        protocol.flush()
        raise

if __name__ == "__main__":
    main()
