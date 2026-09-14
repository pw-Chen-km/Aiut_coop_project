"""Version-checked coordinator; sequential workers and append-only results."""
from __future__ import annotations

from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from doc2skill.config import fingerprint, sha256, read_jsonl, write_json
from . import METHODS
from .data import read, tree_hashes, verify_preparation, verify_originals, installed_packages
from .transport import LocalTransport

def code_hashes(project):
    return {str(p.relative_to(project)): sha256(p) for name in ("qa_experiments", "qa_agent", "doc2skill")
            for p in sorted((Path(project) / "src" / name).rglob("*")) if p.is_file() and p.suffix in (".py", ".md")}

def receive(process, timeout):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError("Worker did not respond before orchestration timeout")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Worker exited; see private worker log")
    return json.loads(line)

def start_worker(root, method, log, build=False, arag_max_loops=None):
    manifest = read(root / "manifest.json")
    project = Path(manifest["project"])
    python = manifest["environments"].get(method, {}).get("python", str(project / ".venv/bin/python"))
    env = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONHASHSEED": "42",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false"}
    validate_loop_budget(manifest, method, arag_max_loops)
    env.pop("QURSOR_ARAG_MAX_LOOPS", None)
    command = [python, "-m", "qa_experiments.worker", "--root", str(root), "--method", method]
    if build:
        command.append("--build")
    return subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, text=True, bufsize=1)

def verify_dependencies(manifest):
    if installed_packages(Path(manifest["project"]) / ".venv/bin/python") != manifest["runtime_packages"]:
        raise ValueError("Existing QA runtime packages changed")
    for method, env in manifest["environments"].items():
        if installed_packages(env["python"]) != env["installed_packages"]:
            raise ValueError("Isolated dependency versions changed: " + method)
        if tree_hashes(env["repo"]) != env["source_hashes"]:
            raise ValueError("Upstream source changed: " + method)
        if tree_hashes(env["model"]) != env["model_hashes"]:
            raise ValueError("Embedding files changed: " + method)

def doctor(root):
    root = Path(root).resolve()
    manifest = verify_preparation(root)
    verify_dependencies(manifest)
    verify_originals(manifest)
    from doc2skill.runtime import validate_runtime_identity
    config = read(root / "online.json")
    validate_runtime_identity(config["runtime"])
    transport = LocalTransport(config["runtime"])
    models = transport._request("/v1/models")
    if config["runtime"]["model"] not in {m["id"] for m in models["data"]}:
        raise ValueError("Qwen server model identity mismatch")
    props = transport._request("/props")
    if props["default_generation_settings"]["n_ctx"] != 8192:
        raise ValueError("Server context differs from experiment")
    if props.get("model_path") != config["runtime"]["gguf_path"]:
        raise ValueError("Server uses a different GGUF path")
    return {"ok": True, "model": config["runtime"]["model"], "context_tokens": 8192,
            "test_count": 120, "generation_requested": False, "judge_requested": False,
            "gpu_state": gpu_state(), "indexes": {m: (root / "indexes" / m).exists() for m in ("arag", "linear")}}

def gpu_state():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

def build_index(root, method):
    root = Path(root).resolve()
    if method not in ("arag", "linear"):
        raise ValueError("Only native baselines need a new index")
    manifest = verify_preparation(root)
    verify_dependencies(manifest)
    with (root / (method + ".index.log")).open("x") as log:
        proc = start_worker(root, method, log, build=True)
        try:
            status = receive(proc, 3600)
            proc.wait(timeout=30)
            if not status.get("ready") or proc.returncode:
                raise RuntimeError(str(status))
            write_json(root / (method + ".index-manifest.json"), {"method": method,
                "artifacts": tree_hashes(root / "indexes" / method), "capabilities": status.get("capabilities")})
            return status
        except Exception as exc:
            write_json(root / (method + ".blocked.json"), {"status": "blocked", "stage": "index",
                "method": method, "error": str(exc), "code": code_hashes(Path(manifest["project"]))})
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)

def verify_indexes(root, methods=("arag", "linear")):
    for method in methods:
        if method not in ("arag", "linear"):
            continue
        manifest = read(root / (method + ".index-manifest.json"))
        if tree_hashes(root / "indexes" / method) != manifest["artifacts"]:
            raise ValueError("Frozen index changed: " + method)

def validate_loop_budget(manifest, method, requested):
    sealed = manifest.get("settings", {}).get("arag", {}).get("max_loops", 15)
    if type(sealed) is not int or not 1 <= sealed <= 15:
        raise ValueError("Invalid sealed A-RAG max_loops")
    if requested is not None and (method != "arag" or type(requested) is not int or requested != sealed):
        raise ValueError("Changed loop budget requires a new preparation and smoke/freeze")


def freeze(root, methods=None):
    root = Path(root).resolve()
    manifest = verify_preparation(root)
    verify_dependencies(manifest)
    methods = list(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Select nonempty unique known methods before freezing")
    smoke, blocked, active = {}, {}, []
    code = code_hashes(Path(manifest["project"]))
    smoke_identity = fingerprint({"preparation": manifest["preparation_id"], "code": code_hashes(Path(manifest["project"]))})
    for method in methods:
        smoke_path = root / "runs/smoke" / smoke_identity[:16] / method / "answers.jsonl"
        block_path = root / (method + ".blocked.json")
        if not smoke_path.exists():
            report = read(block_path) if block_path.exists() else {}
            if report.get("code") != code:
                raise ValueError("Method has not completed a smoke attempt for this code: " + method)
            blocked[method] = report
            continue
        items = read_jsonl(smoke_path)
        expected = {r["qid"] for r in read_jsonl(root / "smoke.questions.jsonl")}
        if {r["qid"] for r in items} != expected:
            raise ValueError("Smoke is incomplete; resume before freezing: " + method)
        smoke[method] = {"path": str(smoke_path.relative_to(root)), "sha256": sha256(smoke_path)}
        if any(r["status"] != "ok" for r in items):
            blocked[method] = {"status": "blocked", "stage": "smoke", "statuses": dict(Counter(r["status"] for r in items)),
                               "smoke": smoke[method]}
            continue
        if method == "arag" and not any(c["response"]["choices"][0]["message"].get("tool_calls") for r in items for c in r["calls"] if c["status"] == "ok"):
            blocked[method] = {"status": "blocked", "stage": "smoke", "reason": "native_tool_calls_not_demonstrated", "smoke": smoke[method]}
            continue
        active.append(method)
    if not active:
        raise ValueError("No method passed compatibility smoke")
    verify_indexes(root, active)
    frozen = {"preparation_id": manifest["preparation_id"], "code": code_hashes(Path(manifest["project"])),
              "indexes": {m: sha256(root / (m + ".index-manifest.json")) for m in active if m in ("arag", "linear")},
              "active_methods": active, "blocked_methods": blocked, "requested_methods": methods,
              "smoke": smoke, "test_has_not_run": not (root / "runs/test").exists(),
              "judge_requested": False, "time_unix": time.time()}
    if not frozen["test_has_not_run"]:
        raise ValueError("Cannot re-freeze after opening test execution")
    frozen["experiment_id"] = fingerprint(frozen)
    with (root / "frozen.json").open("x") as f:
        json.dump(frozen, f, indent=2)
    return frozen

def verify_frozen(root, manifest):
    frozen = read(root / "frozen.json")
    check = dict(frozen)
    ident = check.pop("experiment_id")
    if fingerprint(check) != ident or frozen["preparation_id"] != manifest["preparation_id"]:
        raise ValueError("Frozen experiment identity changed")
    if code_hashes(Path(manifest["project"])) != frozen["code"]:
        raise ValueError("Code changed after freeze; do not mix test versions")
    for method, digest in frozen["indexes"].items():
        if sha256(root / (method + ".index-manifest.json")) != digest:
            raise ValueError("Frozen index manifest changed")
    verify_indexes(root, frozen["indexes"])
    return frozen

def run_method(root, method, phase, *, arag_max_loops=None):
    root = Path(root).resolve()
    if method not in METHODS or phase not in ("smoke", "test"):
        raise ValueError("Unknown method/phase")
    manifest = verify_preparation(root)
    verify_dependencies(manifest)
    validate_loop_budget(manifest, method, arag_max_loops)
    doctor(root)
    if phase == "test":
        frozen = verify_frozen(root, manifest)
        if method not in frozen.get("active_methods", METHODS):
            raise ValueError("Method is blocked by frozen compatibility checks: " + method)
        identity = frozen["experiment_id"]
    else:
        identity = fingerprint({"preparation": manifest["preparation_id"], "code": code_hashes(Path(manifest["project"]))})
    rows = read_jsonl(root / (phase + ".questions.jsonl"))
    output = (root / "runs" / phase / identity[:16] / method) if phase == "smoke" else (root / "runs" / phase / method)
    output.mkdir(parents=True, exist_ok=True)
    run_id = fingerprint({"identity": identity, "method": method, "phase": phase})
    run_manifest = {"run_id": run_id, "identity": identity, "phase": phase, "method": method,
                    "question_sha256": sha256(root / (phase + ".questions.jsonl")), "count": len(rows)}
    existing = output / "run_manifest.json"
    if existing.exists() and read(existing) != run_manifest:
        raise ValueError("Resume requires exactly the same source/code/index/model configuration")
    if not existing.exists():
        write_json(existing, run_manifest)
    answers = output / "answers.jsonl"
    prior = read_jsonl(answers) if answers.exists() else []
    if any(r["run_id"] != run_id for r in prior) or len({r["qid"] for r in prior}) != len(prior):
        raise ValueError("Mixed or duplicate output records")
    expected = {r["qid"]: r["question"] for r in rows}
    if any(expected.get(r["qid"]) != r["question"] for r in prior):
        raise ValueError("Output record belongs to a different question")
    done = {r["qid"] for r in prior}
    if len(done) == len(rows):
        return {"method": method, "completed": len(done), "resumed": True}
    with (root / "inference.lock").open("a") as lock, (output / "worker.log").open("a") as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = start_worker(root, method, log, arag_max_loops=arag_max_loops if method == "arag" else None)
        try:
            ready = receive(proc, 900)
            if not ready.get("ready"):
                if phase == "smoke":
                    write_json(root / (method + ".blocked.json"), {"status": "blocked", "stage": "startup",
                        "method": method, "error": ready, "code": code_hashes(Path(manifest["project"]))})
                raise RuntimeError("Compatibility blocked: " + str(ready))
            write_json(output / "worker_capabilities.json", ready)
            with answers.open("a") as stream:
                for row in rows:
                    if row["qid"] in done:
                        continue
                    before = gpu_state()
                    proc.stdin.write(json.dumps(row) + "\n")
                    proc.stdin.flush()
                    result = receive(proc, 5400)
                    if result.get("qid") != row["qid"] or result.get("question") != row["question"]:
                        raise RuntimeError("Worker protocol mismatch")
                    result.update(run_id=run_id, gpu_state_before=before)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add(row["qid"])
                    print(json.dumps({"method": method, "phase": phase, "completed": len(done), "total": len(rows),
                                      "qid": row["qid"], "status": result["status"]}), flush=True)
            proc.stdin.close()
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
    verify_originals(manifest)
    if phase == "test":
        verify_frozen(root, manifest)
    return {"method": method, "completed": len(done)}

def summarize(root, phase="test"):
    root = Path(root)
    report = {"label": "prototype_shared_gpu_4b_8k_unscored", "judge_requested": False, "phase": phase, "methods": {}}
    frozen = read(root / "frozen.json") if (root / "frozen.json").exists() else {}
    report["blocked_methods"] = frozen.get("blocked_methods", {})
    for method in frozen.get("active_methods", METHODS) if phase == "test" else METHODS:
        if phase == "smoke":
            manifest = verify_preparation(root)
            identity = fingerprint({"preparation": manifest["preparation_id"], "code": code_hashes(Path(manifest["project"]))})
            path = root / "runs" / phase / identity[:16] / method / "answers.jsonl"
        else:
            path = root / "runs" / phase / method / "answers.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        report["methods"][method] = {"records": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
            "inference_seconds": sum(r["elapsed_seconds"] for r in rows),
            "llm_calls": sum(c.get("request_sent", False) for r in rows for c in r["calls"]),
            "tool_error_events": sum(len(r.get("tool_error_events", [])) for r in rows),
            "prompt_tokens": sum(r["tokens"]["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["tokens"]["completion_tokens"] for r in rows),
            "visual_diagnostic_count": sum(bool(r.get("visual_diagnostic")) for r in rows)}
    write_json(root / (phase + ".summary.json"), report)
    return report
