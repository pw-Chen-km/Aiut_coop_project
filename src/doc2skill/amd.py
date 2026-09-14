"""Offline JSON client using the existing pinned, serialized AMD transport."""
from __future__ import annotations
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from .config import write_json
from .llm import LLMError, validate_schema
from qa_skillopt.transport import OllamaTransport, OLLAMA_MODEL, OLLAMA_DIGEST


class ResourceBusy(RuntimeError):
    pass


class GPUResourceGuard:
    """Read-only snapshots on the configured AMD host; no service mutations."""
    def __init__(self, config, audit_dir, *, snapshot=None, sleep=time.sleep):
        self.config, self.audit_dir = config, Path(audit_dir)
        self.snapshot_fn, self.sleep = snapshot, sleep
        self.max_utilization = float(config.get("max_utilization", 10))
        self.min_free_gib = float(config.get("min_free_gib", 20))
        if not 0 < self.max_utilization <= 100 or not 20 <= self.min_free_gib:
            raise ValueError("GPU limits require utilization in (0,100] and at least 20 GiB headroom")

    def snapshot(self):
        if self.snapshot_fn:
            return self.snapshot_fn()
        identity = Path(self.config["identity"]).expanduser()
        host, user = self.config.get("host", ""), self.config.get("user", "")
        port = self.config.get("port", 22)
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user)
                or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError("gpu_guard requires an explicit SSH host, user and valid port")
        code = r'''import subprocess,json,re,pathlib
socket=subprocess.check_output(["ss","-ltnp","sport = :11437"],text=True)
pids=re.findall(r"pid=(\d+)",socket)
if len(set(pids))!=1: raise RuntimeError("Cannot identify existing listener")
pid=pids[0]
allowed={"ROCR_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","OLLAMA_HOST","OLLAMA_NUM_PARALLEL"}
rows=pathlib.Path("/proc/"+pid+"/environ").read_bytes().decode().split(chr(0))
env={r.split("=",1)[0]:r.split("=",1)[1] for r in rows if "=" in r and r.split("=",1)[0] in allowed}
gpu=env.get("ROCR_VISIBLE_DEVICES",env.get("HIP_VISIBLE_DEVICES",""))
if not gpu.isdigit() or env.get("OLLAMA_HOST")!="127.0.0.1:11437": raise RuntimeError("Ambiguous GPU mapping")
raw=json.loads(subprocess.check_output(["rocm-smi","--showuse","--showmeminfo","vram","--json"],text=True))
card=raw["card"+gpu]
print(json.dumps({"gpu":int(gpu),"listener_pid":int(pid),"gpu_binding":env,
"utilization":float(card["GPU use (%)"]),"free_vram_bytes":int(card["VRAM Total Memory (B)"])-int(card["VRAM Total Used Memory (B)"])}))'''
        command = ["ssh", "-i", str(identity), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", "-p", str(port),
                   user + "@" + host, "python3 -c " + shlex.quote(code)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise ResourceBusy("GPU mapping/load cannot be verified; no inference requested")
        return json.loads(result.stdout)

    def check(self, stage):
        samples = [self.snapshot()]
        self.sleep(5)
        samples.append(self.snapshot())
        result = {"stage": stage, "time": time.time(), "samples": samples,
            "limits": {"max_utilization": self.max_utilization, "min_free_gib": self.min_free_gib},
            "safe": all(s["utilization"] < self.max_utilization and s["free_vram_bytes"] >= self.min_free_gib * 1024 ** 3 for s in samples)
                and samples[0]["gpu"] == samples[1]["gpu"]}
        write_json(self.audit_dir / (str(time.time_ns()) + ".json"), result)
        if not result["safe"]:
            raise ResourceBusy("Target GPU exceeds configured utilization/headroom limits; checkpoint retained")
        return result


class AMDMetadataClient:
    def __init__(self, config, audit_dir, *, transport=None):
        self.model = OLLAMA_MODEL
        self.max_input_chars = int(config.get("max_input_chars", 48000))
        self.max_output_tokens = int(config.get("max_output_tokens", 4096))
        self.config = {"endpoint": "http://127.0.0.1:11437", "model": self.model, "digest": OLLAMA_DIGEST,
                       "temperature": 0, "max_output_tokens": self.max_output_tokens}
        self.transport = transport or OllamaTransport(usage_window_confirmed=config.get("usage_window_confirmed") is True,
            timeout=float(config.get("timeout_seconds", 300)), audit_dir=audit_dir)

    def complete(self, messages, schema, stage, *, no_repair=False):
        started = time.monotonic()
        working = [*messages, {"role": "user", "content": "Return only JSON matching this schema:\n" + json.dumps(schema)}]
        digest = hashlib.sha256(json.dumps(working, ensure_ascii=False).encode()).hexdigest()
        usage = {}
        for attempt in range(1 if no_repair else 2):
            if len(json.dumps(working, ensure_ascii=False)) > self.max_input_chars:
                raise LLMError("Offline input exceeds limit; split, do not truncate")
            raw, counts = self.transport.native_chat(working, self.max_output_tokens, role="corpus:" + stage)
            for k, v in counts.items():
                usage[k] = usage.get(k, 0) + v
            try:
                # Only a whole JSON/code-fenced JSON response is accepted, not substring extraction.
                content = re.sub(r"\A```(?:json)?\s*([\s\S]*?)\s*```\Z", r"\1", raw.strip())
                data = json.loads(content)
                validate_schema(data, schema)
                return {"data": data, "model": self.model, "usage": usage,
                    "elapsed_seconds": time.monotonic() - started,
                    "provenance": {"prompt_sha256": digest, "attempts": attempt + 1, "stage": stage,
                                   "model_digest": OLLAMA_DIGEST}}
            except (ValueError, TypeError) as exc:
                if no_repair or attempt:
                    raise LLMError("Offline JSON/schema failed after bounded repair") from exc
                working += [{"role": "assistant", "content": raw},
                            {"role": "user", "content": "Invalid JSON/schema. Return the complete required JSON only."}]
        raise LLMError("No structured output")
