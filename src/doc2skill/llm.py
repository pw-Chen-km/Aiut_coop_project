"""Small, explicit OpenAI-compatible client; never selects a model or endpoint."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    """A request failed without silently substituting generated content."""


class InvalidOutputError(LLMError, ValueError):
    """Malformed model output, distinct from transport and budget failures."""


def validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Validate the JSON-schema subset used by the local tool interfaces."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_schema(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: no anyOf alternative matched")
    expected = schema.get("type")
    checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (float, int)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if expected and (expected not in checks or not checks[expected](value)):
        raise ValueError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: invalid constant")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{path}: missing required fields {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"{path}: unexpected properties")
        for key, child in value.items():
            if key in properties:
                validate_schema(child, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path}: too many items")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ValueError(f"{path}: duplicate items")
        for index, child in enumerate(value):
            validate_schema(child, schema.get("items", {}), f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path}: string is too long")


def is_loopback_endpoint(endpoint: str) -> bool:
    host = urllib.parse.urlsplit(endpoint).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LLMError("LLM endpoint redirects are disabled")


class ChatClient:
    def __init__(self, config: dict, local_only: bool = False):
        if not isinstance(config, dict) or not config.get("endpoint") or not config.get("model"):
            raise ValueError("An explicit LLM endpoint and model are required")
        if any(config.get(key) for key in ("api_key", "token", "secret", "authorization")):
            raise ValueError("Store secrets in an environment variable named by api_key_env")
        self.config = dict(config)
        self.model = str(config["model"])
        endpoint = str(config["endpoint"]).rstrip("/")
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("LLM endpoint must be an explicit HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Endpoint must not contain credentials, query parameters or fragments")
        if local_only and not is_loopback_endpoint(endpoint):
            raise ValueError("Query-time LLM endpoint must be loopback; cloud fallback is prohibited")
        if not local_only and config.get("approved_private_data") is not True:
            raise ValueError("Offline generation requires approved_private_data=true")
        if not is_loopback_endpoint(endpoint) and config.get("zero_data_retention_confirmed") is not True:
            raise ValueError("Remote private-data processing requires zero_data_retention_confirmed=true")
        if not is_loopback_endpoint(endpoint) and parsed.scheme != "https":
            raise ValueError("Remote offline endpoints require HTTPS")
        self.endpoint = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
        self.local_only = local_only
        self.timeout = float(config.get("timeout_seconds", 120))
        self.max_output_tokens = int(config.get("max_output_tokens", 1200))
        self.max_input_chars = int(config.get("max_input_chars", 12000))
        self.context_tokens = int(config.get("context_window_tokens", config.get("context_tokens", 32768)))
        if min(self.timeout, self.max_output_tokens, self.max_input_chars, self.context_tokens) <= 0:
            raise ValueError("LLM budgets and timeout must be positive")
        self.token_counter = config.get("token_counter")
        if self.token_counter is not None and not callable(self.token_counter):
            raise ValueError("token_counter must be callable")
        key_name = config.get("api_key_env")
        self._api_key = os.environ.get(str(key_name), "") if key_name else ""
        if key_name and not self._api_key:
            raise ValueError("Configured api_key_env is not set")
        if not is_loopback_endpoint(endpoint) and not self._api_key:
            raise ValueError("Remote endpoint requires a secret from an explicit api_key_env")
        # Disable environment proxies: local runtime must not send traffic elsewhere.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _check_budget(self, messages: list[dict], schema: dict) -> None:
        serialized = json.dumps({"messages": messages, "schema": schema}, ensure_ascii=False)
        if len(serialized) > self.max_input_chars:
            raise LLMError("Input exceeds max_input_chars; split content rather than truncate")
        # UTF-8 byte count is a conservative fallback, not a chars/4 approximation.
        count = self.token_counter(serialized) if self.token_counter else len(serialized.encode("utf-8"))
        if count + self.max_output_tokens + 256 > self.context_tokens:
            raise LLMError("Input plus output reserve exceeds context_window_tokens")

    def complete(self, messages: list[dict], schema: dict, stage: str, *, no_repair: bool = False) -> dict:
        attempts = 1 if no_repair else 2
        working = list(messages)
        started = time.monotonic()
        usage: dict[str, float] = {}
        prompt_hash = hashlib.sha256(json.dumps({"messages": messages, "schema": schema},
                                               sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        raw = ""
        response_model = self.model
        for attempt in range(attempts):
            self._check_budget(working, schema)
            token_field = self.config.get("token_limit_field", "max_tokens")
            if token_field not in ("max_tokens", "max_completion_tokens"):
                raise ValueError("Unsupported token_limit_field")
            payload = {"model": self.model, "messages": working,
                       token_field: self.max_output_tokens, "store": False,
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "structured_result", "strict": True, "schema": schema}}}
            if self.config.get("reasoning_effort") is not None:
                payload["reasoning_effort"] = self.config["reasoning_effort"]
            if self.config.get("temperature") is not None:
                payload["temperature"] = float(self.config["temperature"])
            if self.config.get("chat_template_kwargs") is not None:
                if not self.local_only or self.config["chat_template_kwargs"] != {"enable_thinking": False}:
                    raise ValueError("Only explicit local non-thinking template configuration is supported")
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = "Bearer " + self._api_key
            request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    envelope = json.loads(response.read(4_000_001))
            except urllib.error.HTTPError as exc:
                raise LLMError(f"LLM HTTP request failed (status {exc.code})") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise LLMError(f"LLM transport/response failed ({type(exc).__name__})") from None
            for key, val in envelope.get("usage", {}).items():
                if isinstance(val, (int, float)):
                    usage[key] = usage.get(key, 0) + val
            response_model = envelope.get("model") or self.model
            if envelope.get("choices") and envelope["choices"][0].get("finish_reason") in {
                "length", "content_filter", "tool_calls", "function_call"
            }:
                raise LLMError("Model did not finish a complete answer at " + stage)
            try:
                raw = envelope["choices"][0]["message"]["content"]
                if not isinstance(raw, str):
                    raise ValueError("message content must be JSON text")
                data = json.loads(raw)
                validate_schema(data, schema)
            except (KeyError, IndexError, TypeError, ValueError):
                if attempt + 1 == attempts:
                    raise InvalidOutputError(f"Invalid structured output at {stage} after {attempt + 1} attempt(s)") from None
                # One bounded format repair; no request replay or provider fallback.
                working = list(messages) + [{"role": "assistant", "content": str(raw)[:1000]},
                    {"role": "user", "content": "Return only JSON matching the given schema. "
                     "The previous response was invalid. Do not change the supplied evidence."}]
                continue
            safe_raw = raw.replace(self._api_key, "[REDACTED]") if self._api_key else raw
            return {"data": data, "usage": usage, "elapsed_seconds": time.monotonic() - started,
                    "model": response_model, "raw_content": safe_raw[:2000],
                    "provenance": {"prompt_sha256": prompt_hash, "stage": stage,
                                   "attempts": attempt + 1, "requested_model": self.model,
                                   "local_only": self.local_only}}
        raise LLMError("Unreachable incomplete LLM result")
