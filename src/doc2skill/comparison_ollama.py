"""Pinned local Ollama JSON client with rendered-prompt GGUF token accounting.

This adapter is deliberately separate from the legacy AMD/SkillOPT transport.
It neither downloads model weights nor substitutes a tokenizer or an endpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .llm import InvalidOutputError, LLMError, is_loopback_endpoint, validate_schema


class ComparisonTransportError(LLMError):
    """Transport, model identity, or incomplete-generation failure."""


class InputCapacityError(ComparisonTransportError):
    def __init__(self, stage: str, input_tokens: int, output_tokens: int, context_tokens: int):
        self.details = {"stage": stage, "input_tokens": input_tokens,
                        "output_reserve": output_tokens, "context_length": context_tokens}
        super().__init__(f"{stage}: input {input_tokens} + output reserve {output_tokens} "
                         f"exceeds loaded context {context_tokens}; no truncation or retry")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory atomic replacement also works on Windows.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class GGUFTokenizer:
    """Vocabulary-only llama.cpp CLI; no inference model or service is created."""
    def __init__(self, executable, gguf_path, *, expected_build="10091", timeout=60):
        self.executable, self.gguf_path = Path(executable).resolve(), Path(gguf_path).resolve()
        self.timeout, self.expected_build = timeout, str(expected_build)
        if not self.executable.is_file() or not self.gguf_path.is_file():
            raise ComparisonTransportError("Exact local GGUF and llama-tokenize executable are required")
        result = self._run([str(self.executable), "--version"])
        version = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if not re.search(r"version:\s*" + re.escape(self.expected_build) + r"\b", version):
            raise ComparisonTransportError("llama-tokenize does not match the pinned build")
        self.identity = {"implementation": "llama.cpp", "build": self.expected_build,
                         "version": version.strip(), "executable_sha256": _file_sha256(self.executable),
                         "gguf_path": str(self.gguf_path), "gguf_sha256": _file_sha256(self.gguf_path),
                         "add_special": True, "parse_special": True}

    def _run(self, command, input_bytes=None):
        result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=self.timeout,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise ComparisonTransportError(f"Exact GGUF tokenizer failed (exit {result.returncode})")
        return result

    def __call__(self, rendered_prompt: str) -> int:
        # --stdin receives UTF-8 bytes verbatim, without Windows newline conversion.
        result = self._run([str(self.executable), "-m", str(self.gguf_path), "--stdin", "--ids",
                            "--no-escape", "--offline", "--log-disable"],
                           rendered_prompt.encode("utf-8"))
        try:
            tokens = json.loads(result.stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ComparisonTransportError("Tokenizer did not return a complete token ID list") from exc
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ComparisonTransportError("Tokenizer returned invalid token IDs")
        return len(tokens)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ComparisonTransportError("Local Ollama redirects are forbidden")


@contextmanager
def _serial(path, timeout):
    """Cross-process advisory lock for this local endpoint on Windows and POSIX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline, locked = time.monotonic() + timeout, False
        try:
            while not locked:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ComparisonTransportError("Timed out waiting for serialized local inference")
                    time.sleep(.05)
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ComparisonOllamaClient:
    """Complete text-only JSON requests using one explicitly configured local model.

    Completed requests are cached by their complete input, schema, stage and
    execution identity. Errors are retained, never automatically regenerated.
    ``requester`` and ``tokenizer`` injection exists for offline contract tests.
    """
    def __init__(self, config, audit_dir, *, tokenizer=None, requester=None):
        self.config = dict(config)
        self.local_only = True
        self.metrics = {"generation_calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "total_tokens": 0, "elapsed_seconds": 0.0,
                        "calibration_calls": 0, "calibration_prompt_tokens": 0,
                        "calibration_completion_tokens": 0}
        self.endpoint = str(config["endpoint"]).rstrip("/")
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme != "http" or not is_loopback_endpoint(self.endpoint) or parsed.path
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("Comparison inference requires an explicit loopback HTTP endpoint")
        self.model, self.model_digest = str(config["model"]), str(config["digest"]).removeprefix("sha256:")
        if not re.fullmatch("[a-f0-9]{64}", self.model_digest):
            raise ValueError("The full model digest must be pinned")
        self.think = config.get("think", False)
        self.schema_mode = config.get("schema_mode", "native")
        if self.schema_mode not in {"native", "prompt"}:
            raise ValueError("schema_mode must be native or prompt")
        self.generation_mode = config.get("generation_mode", "chat")
        if self.generation_mode not in {"chat", "harmony_final"}:
            raise ValueError("generation_mode must be chat or harmony_final")
        if self.generation_mode == "harmony_final" and (self.think is not False or self.schema_mode != "native"):
            raise ValueError("harmony_final requires think=false and schema_mode=native")
        if (config.get("temperature", 0) != 0 or not (self.think is False
                or type(self.think) is str and self.think in {"low", "medium", "high"})):
            raise ValueError("Controlled comparison requires temperature=0 and an explicit supported thinking mode")
        self.context_tokens = int(config.get("context_tokens", 32768))
        self.max_output_tokens = int(config.get("max_output_tokens", 4096))
        self.timeout = float(config.get("timeout_seconds", 900))
        self.version = str(config.get("ollama_version", "0.32.5"))
        self.keep_alive = config.get("keep_alive", "30m")
        if min(self.context_tokens, self.max_output_tokens, self.timeout) <= 0:
            raise ValueError("Model context, output reservation, and timeout must be positive")
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("Output reservation must leave room for an input")
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(tempfile.gettempdir()) / ("navigation-comparison-" + _digest(self.endpoint)[:16] + ".lock")
        self._requester = requester
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self.tokenizer = tokenizer or GGUFTokenizer(config["tokenizer_executable"], config["gguf_path"],
                                                   expected_build=config.get("tokenizer_build", "10091"))
        if not callable(self.tokenizer) or not isinstance(getattr(self.tokenizer, "identity", None), dict):
            raise ValueError("A tokenizer with verifiable identity is required")
        manifest_path = config.get("tokenizer_source_manifest")
        if manifest_path:
            source = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if (source.get("gguf_sha256") != self.tokenizer.identity["gguf_sha256"]
                    or source.get("model_digest") != self.model_digest):
                raise ComparisonTransportError("Tokenizer source manifest does not match the local vocabulary and pinned model")
            for key in ("remote_model_path", "remote_blob_sha256", "model_info_sha256"):
                if not isinstance(source.get(key), str) or not source[key]:
                    raise ComparisonTransportError("Tokenizer source manifest lacks a remote model binding")
            self.tokenizer.identity = {**self.tokenizer.identity,
                **{key: source[key] for key in ("remote_model_path", "remote_blob_sha256", "model_info_sha256")},
                **{key: source[key] for key in ("metadata_only", "tensor_count", "gguf_writer_version",
                                                "architecture_mapping", "pretokenizer_mapping") if key in source},
                "source_manifest_sha256": _file_sha256(manifest_path)}
        self._preflight = None
        self._calibrated = False
        self._last_rendered = None

    def _request(self, path, payload=None):
        if self._requester:
            return self._requester(path, payload)
        request = Request(self.endpoint + path, data=None if payload is None else _json(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", "")
            except (ValueError, UnicodeDecodeError):
                detail = ""
            # An explicit server error is evidence, not an invitation to retry.
            raise ComparisonTransportError(f"Ollama {path} HTTP {exc.code}: {detail}") from None
        except (URLError, TimeoutError, ValueError) as exc:
            raise ComparisonTransportError(f"Ollama {path} failed: {type(exc).__name__}; no retry") from exc
        if not isinstance(data, dict) or data.get("error"):
            raise ComparisonTransportError(f"Ollama {path} returned an error: {data.get('error') if isinstance(data, dict) else 'invalid JSON object'}")
        return data

    def _model_row(self, inventory, *, loaded):
        rows = [row for row in inventory.get("models", []) if row.get("name", row.get("model")) == self.model]
        if not rows and loaded:
            return None
        if len(rows) != 1 or str(rows[0].get("digest", "")).removeprefix("sha256:") != self.model_digest:
            raise ComparisonTransportError("Pinned local model is missing or its digest changed")
        if loaded and rows[0].get("context_length") != self.context_tokens:
            raise ComparisonTransportError(f"Loaded context must be {self.context_tokens}, got {rows[0].get('context_length')}")
        return rows[0]

    def _loaded(self):
        self._model_row(self._request("/api/tags"), loaded=False)
        row = self._model_row(self._request("/api/ps"), loaded=True)
        if row is None:
            raise ComparisonTransportError("Pinned model is no longer loaded; run preflight to load and verify it")
        return row

    def preflight(self, *, load=True, calibrate=True):
        with _serial(self.lock_path, self.timeout):
            version = self._request("/api/version").get("version")
            if version != self.version:
                raise ComparisonTransportError(f"Expected Ollama {self.version}, got {version}")
            self._model_row(self._request("/api/tags"), loaded=False)
            identity = self.tokenizer.identity
            remote_fields = ("remote_model_path", "remote_blob_sha256", "model_info_sha256")
            remote = any(key in identity for key in remote_fields)
            if remote and not all(identity.get(key) for key in remote_fields):
                raise ComparisonTransportError("Remote tokenizer identity is incomplete")
            show = self._request("/api/show", {"model": self.model, **({"verbose": True} if remote else {})})
            if (self.generation_mode == "harmony_final"
                    and show.get("model_info", {}).get("general.architecture") not in {"gptoss", "gpt-oss"}):
                raise ComparisonTransportError("harmony_final is only supported for verified GPT-OSS model metadata")
            match = re.search(r"(?m)^FROM\s+(.+)$", show.get("modelfile", ""))
            if not match:
                raise ComparisonTransportError("Ollama did not identify the local GGUF backing this model")
            reported_text = match.group(1).strip().strip('"')
            if remote:
                reported = PurePosixPath(reported_text)
                if reported != PurePosixPath(identity["remote_model_path"]):
                    raise ComparisonTransportError("Remote GGUF differs from the tokenizer source model")
                if reported.name != "sha256-" + identity["remote_blob_sha256"]:
                    raise ComparisonTransportError("Remote GGUF blob identity differs from the recorded source")
                if not show.get("model_info") or _digest(show["model_info"]) != identity["model_info_sha256"]:
                    raise ComparisonTransportError("Remote model metadata differs from the local tokenizer source")
            else:
                reported = Path(reported_text).resolve()
                expected_path = Path(identity["gguf_path"]).resolve()
                if reported != expected_path:
                    raise ComparisonTransportError("Tokenizer GGUF differs from the configured Ollama model")
                if reported.name.startswith("sha256-") and reported.name[7:] != identity["gguf_sha256"]:
                    raise ComparisonTransportError("Local GGUF content does not match its Ollama blob digest")
            loaded = self._model_row(self._request("/api/ps"), loaded=True)
            if loaded is None and load:
                self._request("/api/generate", {"model": self.model, "stream": False,
                    "keep_alive": self.keep_alive, "options": {"num_ctx": self.context_tokens}})
                loaded = self._loaded()
            if loaded is None:
                raise ComparisonTransportError("The pinned model is not loaded")
            self._preflight = {"endpoint": self.endpoint, "model": self.model, "digest": self.model_digest,
                               "version": version, "context_length": loaded["context_length"],
                               "tokenizer": identity, "temperature": 0, "think": self.think,
                               "schema_mode": self.schema_mode,
                               "generation_mode": self.generation_mode,
                               "truncate": False, "shift": False,
                               "modelfile_sha256": _digest(show.get("modelfile", ""))}
            if calibrate and not self._calibrated:
                messages = [{"role": "system", "content": "Return the JSON requested by the user."},
                            {"role": "user", "content": 'Return exactly {"ok":true}. Unicode test: 安裝 Linux.'}]
                schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                          "required": ["ok"], "additionalProperties": False}
                working = self._messages(messages, schema)
                payload = self._payload(working, schema, output_tokens=64)
                measurement = self._measure(payload, "tokenizer-calibration", output_tokens=64)
                generation_endpoint, generation_payload = self._generation_request(payload)
                _write(self.audit_dir / "calibration.request.json", {"payload": generation_payload,
                       "endpoint": generation_endpoint, "render_payload": payload,
                       "measurement": measurement, "rendered_prompt": self._last_rendered})
                envelope = self._adapt_response(self._request(generation_endpoint, generation_payload), generation_endpoint)
                _write(self.audit_dir / "calibration.response.json", envelope)
                self.metrics["calibration_calls"] += 1
                self.metrics["calibration_prompt_tokens"] += envelope.get("prompt_eval_count", 0)
                self.metrics["calibration_completion_tokens"] += envelope.get("eval_count", 0)
                self._validate_envelope(envelope, measurement, "tokenizer-calibration")
                self._calibrated = True
                self._preflight["calibration"] = {"measurement": measurement, "response": envelope,
                                                   "matches_server_prompt_eval_count": True}
            _write(self.audit_dir / "preflight.json", self._preflight)
            return dict(self._preflight)

    @staticmethod
    def _messages(messages, schema):
        if not messages or any(not isinstance(row, dict) or set(row) != {"role", "content"}
                or row["role"] not in {"system", "user", "assistant"} or not isinstance(row["content"], str)
                for row in messages):
            raise ValueError("Only complete text messages with role and content are supported")
        return [dict(row) for row in messages] + [{"role": "user", "content":
            "Return only complete JSON matching this schema:\n" + _json(schema)}]

    def _payload(self, working, schema, *, output_tokens=None):
        return {"model": self.model, "messages": working,
                **({"format": schema} if self.schema_mode == "native" else {}), "stream": False,
                "think": self.think, "truncate": False, "shift": False, "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "seed": 42, "num_ctx": self.context_tokens,
                            "num_predict": output_tokens or self.max_output_tokens}}

    def _measure(self, payload, stage, *, output_tokens=None):
        loaded = self._loaded()
        rendered = self._request("/api/chat", {**payload, "_debug_render_only": True})
        prompt = rendered.get("_debug_info", {}).get("rendered_template")
        if not isinstance(prompt, str) or not prompt:
            raise ComparisonTransportError("Ollama did not expose the complete rendered prompt")
        original_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.generation_mode == "harmony_final":
            if not prompt.endswith("<|start|>assistant"):
                raise ComparisonTransportError("GPT-OSS rendered prompt lacks the exact expected open assistant header")
            # Standard Harmony channel framing; the source/system text is unchanged.
            prompt += "<|channel|>final<|message|>"
        count = self.tokenizer(prompt)
        if type(count) is not int or count <= 0:
            raise ComparisonTransportError("Invalid exact token count")
        output = output_tokens or self.max_output_tokens
        result = {"input_tokens": count, "context_length": loaded["context_length"],
                  "output_reserve": output, "available_input_tokens": loaded["context_length"] - output,
                  "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "ollama_rendered_prompt_sha256": original_prompt_sha256,
                  "appended_channel_header": "<|channel|>final<|message|>" if self.generation_mode == "harmony_final" else ""}
        self._last_rendered = prompt
        if count + output > loaded["context_length"]:
            raise InputCapacityError(stage, count, output, loaded["context_length"])
        return result

    def _generation_request(self, chat_payload):
        if self.generation_mode == "chat":
            return "/api/chat", chat_payload
        if not isinstance(self._last_rendered, str) or not self._last_rendered.endswith("<|channel|>final<|message|>"):
            raise ComparisonTransportError("A measured final-channel prompt is required before raw generation")
        return "/api/generate", {"model": self.model, "prompt": self._last_rendered, "raw": True,
            "format": chat_payload["format"], "think": False, "stream": False,
            "truncate": False, "shift": False, "keep_alive": self.keep_alive, "options": chat_payload["options"]}

    @staticmethod
    def _adapt_response(response, endpoint):
        if endpoint == "/api/chat":
            return response
        return {**response, "message": {"role": "assistant", "content": response.get("response"),
                                         "thinking": response.get("thinking", "")},
                "_comparison_endpoint": endpoint, "_comparison_original_response": response}

    def count_request(self, messages, schema):
        if self._preflight is None:
            self.preflight()
        with _serial(self.lock_path, self.timeout):
            return self._measure(self._payload(self._messages(messages, schema), schema), "count-request")

    def _validate_envelope(self, envelope, measurement, stage):
        if envelope.get("model") != self.model or envelope.get("done") is not True:
            raise ComparisonTransportError(f"{stage}: response identity or completion status is invalid")
        if envelope.get("done_reason") != "stop":
            raise ComparisonTransportError(f"{stage}: generation ended with {envelope.get('done_reason')}; incomplete output rejected")
        message = envelope.get("message", {})
        if ((self.think is False and message.get("thinking")) or message.get("tool_calls")
                or not isinstance(message.get("content"), str)):
            raise ComparisonTransportError(f"{stage}: unexpected thinking, tools, or missing text")
        for field in ("prompt_eval_count", "eval_count"):
            if type(envelope.get(field)) is not int or envelope[field] < 0:
                raise ComparisonTransportError(f"{stage}: server token accounting is missing")
        if envelope["prompt_eval_count"] != measurement["input_tokens"]:
            raise ComparisonTransportError(f"{stage}: exact tokenizer count {measurement['input_tokens']} "
                                           f"differs from server {envelope['prompt_eval_count']}")
        return message["content"]

    def complete(self, messages, schema, stage, *, no_repair=False):
        if self._preflight is None or not self._calibrated:
            self.preflight()
        execution = {"preflight": {key: value for key, value in self._preflight.items() if key != "calibration"},
                     "max_output_tokens": self.max_output_tokens, "seed": 42, "client_contract": 1}
        key = _digest({"messages": messages, "schema": schema, "stage": stage,
                       "execution": execution, "no_repair": no_repair})
        directory = self.audit_dir / "requests" / key
        with _serial(self.lock_path, self.timeout):
            self._loaded()
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                validate_schema(result["data"], schema)
                self.metrics["cache_hits"] += 1
                return {**result, "cached": True}
            if (directory / "error.json").exists():
                error = json.loads((directory / "error.json").read_text(encoding="utf-8"))
                if error["error_type"] == "InvalidOutputError":
                    restored = InvalidOutputError(error["message"])
                    restored.raw_content = error.get("raw_content", "")
                    raise restored
                if error["error_type"] == "InputCapacityError":
                    details = error["details"]
                    raise InputCapacityError(details["stage"], details["input_tokens"],
                                             details["output_reserve"], details["context_length"])
                raise ComparisonTransportError(f"Recorded request failure at {stage}: {error['message']}")
            started, usage = time.monotonic(), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            working = self._messages(messages, schema)
            _write(directory / "input.json", {"stage": stage, "messages": messages, "schema": schema,
                                               "execution": execution, "request_key": key})
            try:
                for attempt in range(1 if no_repair else 2):
                    prefix = directory / f"attempt-{attempt + 1}"
                    response_path = prefix.with_suffix(".response.json")
                    request_path = prefix.with_suffix(".request.json")
                    payload = self._payload(working, schema)
                    measurement = self._measure(payload, stage)
                    generation_endpoint, generation_payload = self._generation_request(payload)
                    if request_path.exists() and not response_path.exists():
                        raise ComparisonTransportError("Interrupted request has no recorded response; refusing an untracked duplicate generation")
                    if response_path.exists():
                        envelope = json.loads(response_path.read_text(encoding="utf-8"))
                    else:
                        _write(request_path, {"payload": generation_payload, "endpoint": generation_endpoint,
                                              "render_payload": payload, "measurement": measurement,
                                              "rendered_prompt": self._last_rendered, "time_unix": time.time()})
                        envelope = self._adapt_response(self._request(generation_endpoint, generation_payload), generation_endpoint)
                        _write(response_path, envelope)
                        self.metrics["generation_calls"] += 1
                        self.metrics["prompt_tokens"] += envelope.get("prompt_eval_count", 0)
                        self.metrics["completion_tokens"] += envelope.get("eval_count", 0)
                        self.metrics["total_tokens"] = self.metrics["prompt_tokens"] + self.metrics["completion_tokens"]
                    raw = self._validate_envelope(envelope, measurement, stage)
                    usage["prompt_tokens"] += envelope["prompt_eval_count"]
                    usage["completion_tokens"] += envelope["eval_count"]
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    try:
                        # Complete JSON or a complete fenced JSON document only.
                        content = re.sub(r"\A```(?:json)?\s*([\s\S]*?)\s*```\Z", r"\1", raw.strip())
                        data = json.loads(content)
                        validate_schema(data, schema)
                    except (ValueError, TypeError) as exc:
                        if no_repair or attempt:
                            invalid = InvalidOutputError(f"{stage}: invalid JSON/schema after {attempt + 1} attempt(s)")
                            invalid.raw_content = raw
                            raise invalid from exc
                        working += [{"role": "assistant", "content": raw}, {"role": "user", "content":
                            "The previous response did not match the JSON schema. Return the complete required JSON only. "
                            "Keep the supplied evidence unchanged. Validation error: " + str(exc)}]
                        continue
                    result = {"data": data, "model": self.model, "usage": usage,
                              "elapsed_seconds": time.monotonic() - started, "raw_content": raw, "cached": False,
                              "thinking_content": envelope.get("message", {}).get("thinking", ""),
                              "provenance": {"stage": stage, "attempts": attempt + 1, "prompt_sha256": key,
                                             "model_digest": self.model_digest, "audit_dir": str(directory.resolve()),
                                             "tokenizer": self.tokenizer.identity, "measurement": measurement}}
                    _write(directory / "result.json", result)
                    self.metrics["elapsed_seconds"] += time.monotonic() - started
                    return result
            except Exception as exc:
                _write(directory / "error.json", {"stage": stage, "error_type": type(exc).__name__,
                                                   "message": str(exc), "details": getattr(exc, "details", {}),
                                                   "raw_content": getattr(exc, "raw_content", ""),
                                                   "usage": usage, "elapsed_seconds": time.monotonic() - started})
                self.metrics["elapsed_seconds"] += time.monotonic() - started
                raise
        raise ComparisonTransportError("No complete JSON response")
