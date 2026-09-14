"""Native, serial Ollama transport for the pinned local-only SkillOpt runner.

No service setup, model loading, pulling, or server configuration belongs here.
The private upstream transport shim preserves SkillOpt's actual optimizer code.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Any, Callable, Iterator
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler


OLLAMA_BASE_URL = "http://127.0.0.1:11437"
OLLAMA_MODEL = "qwen3.6:35b-a3b-bf16"
OLLAMA_DIGEST = "94061ddd23a7de9c902f7bc468455aff03564d26e929b7e2e9ce62c5e6a3a492"
CONTEXT_RESERVE_TOKENS = 512


class TransportError(RuntimeError):
    """A transport contract failed; never fall back to another provider."""


class _AbortUpstreamCall(BaseException):
    """Escape upstream's broad Exception fallbacks, then restore a normal error.

    SkillOpt catches model exceptions and otherwise treats them as absent patches.
    This private carrier exists only inside our installed transport context.
    """
    def __init__(self, error: TransportError):
        self.error = error


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError("Ollama redirects are forbidden")


@dataclass
class OllamaTransport:
    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    model_digest: str = OLLAMA_DIGEST
    usage_window_confirmed: bool = False
    timeout: float = 300.0
    lock_timeout: float = 300.0
    lock_path: Path | str = field(default_factory=lambda: Path(tempfile.gettempdir()) / "qa-skillopt-ollama-11437.lock")
    opener: Callable[..., Any] | None = field(default=None, repr=False)
    audit_dir: Path | str | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.base_url != OLLAMA_BASE_URL:
            raise ValueError("Only the approved execution-host loopback endpoint on port 11437 is allowed")
        if self.model != OLLAMA_MODEL or self.model_digest.removeprefix("sha256:") != OLLAMA_DIGEST:
            raise ValueError("The model name and full digest must match the pinned existing model")
        self.model_digest = OLLAMA_DIGEST
        if self.timeout <= 0 or self.lock_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.lock_path = Path(self.lock_path).absolute()
        # Ignore environment proxy configuration: prompts must stay on loopback.
        if self.opener is None:
            self.opener = build_opener(ProxyHandler({}), _NoRedirect()).open

    @contextmanager
    def _serial(self) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TransportError("Timed out waiting for the shared Ollama generation lock")
                    time.sleep(0.05)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _request(self, path: str, payload: dict | None = None, *, timeout: float | None = None) -> dict:
        if (self.base_url != OLLAMA_BASE_URL or self.model != OLLAMA_MODEL
                or self.model_digest != OLLAMA_DIGEST):
            raise TransportError("Transport identity changed after initialization")
        if payload is not None and self.usage_window_confirmed is not True:
            raise TransportError("Generation requires an explicitly confirmed usage window")
        allowed = {"/api/version", "/api/tags", "/api/ps"} if payload is None else {"/api/chat"}
        if path not in allowed:
            raise TransportError("Unapproved Ollama API path")
        request = Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            assert self.opener is not None
            with self.opener(request, timeout=timeout or self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            # Do not include HTTP response bodies, prompts, or credentials in errors.
            raise TransportError(f"Ollama {path} request failed ({type(exc).__name__})") from exc
        if not isinstance(data, dict) or data.get("error"):
            raise TransportError(f"Ollama {path} returned an invalid or error response")
        return data

    def _check_model(self, data: dict, *, loaded: bool) -> dict:
        models = data.get("models")
        if not isinstance(models, list):
            raise TransportError("Ollama model inventory is malformed")
        matches = [row for row in models if isinstance(row, dict) and
                   (row.get("name") == self.model or row.get("model") == self.model)]
        if len(matches) != 1:
            state = "already loaded" if loaded else "available"
            raise TransportError(f"Pinned model must be {state}; no model load or pull will be requested")
        digest = str(matches[0].get("digest", "")).removeprefix("sha256:")
        if digest != self.model_digest:
            raise TransportError("Ollama model digest changed")
        if loaded and (type(matches[0].get("context_length")) is not int or matches[0]["context_length"] <= 0):
            raise TransportError("Loaded model context_length is missing or invalid; generation is forbidden")
        return matches[0]

    def preflight(self) -> dict:
        """Read-only doctor: never generate, even when usage is unconfirmed."""
        version = self._request("/api/version", timeout=10)
        self._check_model(self._request("/api/tags", timeout=10), loaded=False)
        loaded = self._check_model(self._request("/api/ps", timeout=10), loaded=True)
        return {"base_url": self.base_url, "model": self.model, "digest": self.model_digest,
                "version": version.get("version"), "loaded": True, "context_length": loaded["context_length"],
                "generation_requested": False}

    def native_chat(self, messages: list[dict], max_tokens: int, role: str = "target",
                    *, timeout: float | None = None) -> tuple[str, dict[str, int]]:
        if self.usage_window_confirmed is not True:
            raise TransportError("Generation requires an explicitly confirmed usage window")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not messages or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                               or not isinstance(m.get("content"), str)
                               or set(m) - {"role", "content"} for m in messages):
            raise TransportError("Only explicit text messages are supported; tools must not be silently discarded")
        if not isinstance(role, str) or not role:
            raise ValueError("A diagnostic role is required")
        payload = {"model": self.model, "messages": [dict(m) for m in messages],
                   "think": False, "stream": False,
                   "options": {"temperature": 0, "num_predict": max_tokens}}
        audit = None
        started = time.monotonic()
        if self.audit_dir is not None:
            from .io import write_new
            audit = Path(self.audit_dir) / uuid.uuid4().hex
            write_new(audit / "request.json", {"role": role, "payload": payload,
                      "model_digest": self.model_digest, "time_unix": time.time()})
        with self._serial():
            # Check under the cross-process lock before every generation, not just doctor.
            self._check_model(self._request("/api/tags", timeout=10), loaded=False)
            loaded = self._check_model(self._request("/api/ps", timeout=10), loaded=True)
            # UTF-8 bytes conservatively bound text tokens; reserve additional
            # chat-template framing per message. Never silently truncate or set num_ctx.
            input_upper_bound = len(json.dumps(payload["messages"], ensure_ascii=False).encode("utf-8"))
            reserved = CONTEXT_RESERVE_TOKENS + 32 * len(payload["messages"])
            if input_upper_bound + max_tokens + reserved > loaded["context_length"]:
                raise TransportError("Serialized input plus output and framing reserve exceeds the loaded context_length")
            data = self._request("/api/chat", payload, timeout=timeout)
        if audit is not None:
            safe_data = {**data, "message": {k: v for k, v in data.get("message", {}).items() if k != "thinking"}}
            write_new(audit / "response.json", {"data": safe_data, "thinking_present": bool(data.get("message", {}).get("thinking")),
                      "elapsed_seconds": time.monotonic() - started})
        if data.get("model") != self.model or data.get("done") is not True:
            raise TransportError("Ollama returned the wrong model or an incomplete response")
        if data.get("done_reason") != "stop":
            raise TransportError("Ollama response was truncated or did not terminate normally")
        message = data.get("message")
        if not isinstance(message, dict) or message.get("thinking") or message.get("tool_calls"):
            raise TransportError("Thinking or tool output is forbidden in this text-only transport")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise TransportError("Ollama returned empty text")
        counts = [data.get("prompt_eval_count"), data.get("eval_count")]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
            raise TransportError("Ollama did not return valid token accounting")
        return content, {"prompt_tokens": counts[0], "completion_tokens": counts[1],
                         "total_tokens": sum(counts)}

    @contextmanager
    def install_upstream_backend(self) -> Iterator[None]:
        """Version-guarded private transport shim; requires load_upstream first."""
        from .upstream import verify_loaded_upstream
        verify_loaded_upstream()
        from skillopt.model import qwen_backend

        original = qwen_backend._post_chat_completion

        def native_post_impl(payload: dict, timeout: float | None, config: Any) -> dict:
            if payload.get("model") != self.model or config.base_url.rstrip("/") != self.base_url:
                raise TransportError("Upstream attempted an unapproved endpoint or model")
            if set(payload) - {"model", "messages", "max_tokens", "temperature"}:
                raise TransportError("Unsupported upstream request options")
            if payload.get("temperature") != 0 or config.enable_thinking:
                raise TransportError("Upstream generation settings violated the thinking/temperature contract")
            role = "optimizer" if config is qwen_backend.OPTIMIZER_CONFIG else "target"
            content, usage = self.native_chat(payload["messages"], payload["max_tokens"], role, timeout=timeout)
            return {"choices": [{"message": {"role": "assistant", "content": content},
                                  "finish_reason": "stop"}], "usage": usage}

        def native_post(payload: dict, timeout: float | None, config: Any) -> dict:
            try:
                return native_post_impl(payload, timeout, config)
            except TransportError as exc:
                raise _AbortUpstreamCall(exc) from exc

        qwen_backend._post_chat_completion = native_post
        try:
            yield
        except _AbortUpstreamCall as exc:
            raise exc.error from exc
        finally:
            qwen_backend._post_chat_completion = original
