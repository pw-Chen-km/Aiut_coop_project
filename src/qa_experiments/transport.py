"""The only baseline HTTP transport: pinned local Qwen, exact server tokenizer.

No OpenAI credentials, environment proxies, redirects, providers or judge calls.
The formatting/tokenization endpoints do not perform language-model generation.
"""
from __future__ import annotations

import copy
import io
import json
import socket
import time
import urllib.error
import urllib.request


class ExperimentError(RuntimeError):
    status = "runtime_error"

class BudgetError(ExperimentError):
    status = "budget_failure"

class OutputError(ExperimentError):
    status = "invalid_output"

class ToolError(OutputError):
    status = "invalid_tool_call"

class TimeoutError(ExperimentError):
    status = "timeout"

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ExperimentError("Endpoint redirects are prohibited")


class LocalTransport:
    def __init__(self, runtime, *, request=None):
        self.config = dict(runtime)
        if runtime["endpoint"] != "http://127.0.0.1:18084/v1":
            raise ValueError("Only the approved local Qwen endpoint is allowed")
        if runtime["model"] != "Qwen3.5-4B-Q4_K_M":
            raise ValueError("Unexpected online model")
        if runtime.get("chat_template_kwargs") != {"enable_thinking": False}:
            raise ValueError("Non-thinking mode must be explicit")
        if runtime.get("context_tokens") != 8192 or runtime.get("max_input_tokens") != 7000:
            raise ValueError("Experiment context must be 8192 / input 7000")
        self.base = "http://127.0.0.1:18084"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self._request = request or self._http
        self.calls = []

    def _http(self, path, payload=None):
        if path not in {"/apply-template", "/tokenize", "/props", "/v1/models", "/v1/chat/completions"}:
            raise ValueError("Unknown endpoint")
        req = urllib.request.Request(self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=float(self.config.get("timeout_seconds", 300))) as response:
                return json.load(response)
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError("Local Qwen request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise TimeoutError("Local Qwen request timed out") from exc
            raise ExperimentError("Local Qwen HTTP/transport failure") from exc

    def count_text(self, text):
        return len(self._request("/tokenize", {"content": text, "add_special": False})["tokens"])

    def count_request(self, payload):
        inputs = {"messages": payload["messages"], "add_generation_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}}
        if payload.get("tools"):
            inputs.update(tools=payload["tools"], tool_choice=payload.get("tool_choice", "auto"))
        rendered = self._request("/apply-template", inputs)
        prompt = rendered["prompt"]
        return len(self._request("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"])

    def send(self, payload):
        if payload.get("model") != self.config["model"] or payload.get("temperature") != 0:
            raise ValueError("Request changed the frozen model or temperature")
        if payload.get("chat_template_kwargs") != {"enable_thinking": False}:
            raise ValueError("Request must disable thinking")
        maximum = payload.get("max_tokens")
        if type(maximum) is not int or not 1 <= maximum <= 768:
            raise ValueError("Invalid output budget")
        call = {"request": copy.deepcopy(payload), "status": "pending", "request_sent": False}
        self.calls.append(call)
        start = time.monotonic()
        try:
            count = self.count_request(payload)
            call["input_tokens"] = count
            if count > 7000 or count + maximum + 256 > 8192:
                raise BudgetError("Complete template/messages/tools exceed frozen input budget")
            call["request_sent"] = True
            envelope = self._request("/v1/chat/completions", payload)
            call["response"] = envelope
            call["usage"] = envelope.get("usage", {})
            if envelope.get("model") not in (None, self.config["model"]):
                raise OutputError("Server returned a different model")
            choice = envelope["choices"][0]
            if choice.get("finish_reason") == "length":
                raise OutputError("Output token limit reached before a complete response")
            if choice.get("finish_reason") not in ("stop", "tool_calls"):
                raise OutputError("Unexpected completion stop reason")
            message = choice["message"]
            if message.get("tool_calls"):
                allowed = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in payload.get("tools", [])}
                for tool in message["tool_calls"]:
                    function = tool.get("function", {})
                    if not tool.get("id") or not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                        raise ToolError("Malformed tool-call envelope")
                    try:
                        if function["name"] not in allowed:
                            raise ValueError("Unknown tool")
                        args = json.loads(function["arguments"])
                        schema = allowed[function["name"]]
                        if not isinstance(args, dict) or not set(schema.get("required", [])) <= args.keys():
                            raise ValueError()
                        for key, value in args.items():
                            expected = schema.get("properties", {}).get(key, {}).get("type")
                            if expected == "string" and not isinstance(value, str): raise ValueError()
                            if expected == "integer" and (type(value) is not int or value < 1): raise ValueError()
                            if expected == "array" and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)): raise ValueError()
                    except (ValueError, KeyError, TypeError):
                        # Preserve the native agent's error-feedback/recovery
                        # loop. Its registry can only execute registered tools.
                        call.setdefault("tool_validation_errors", []).append({"status": "invalid_tool_call",
                            "tool_call_id": tool["id"], "tool": function["name"]})
            else:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise OutputError("Empty answer")
                if any(marker in content for marker in ("<tool_call>", "[TOOL_CALLS]", "<|tool_call|>")):
                    raise ToolError("Unparsed tool serialization is not an answer")
            call["status"] = "ok"
            return envelope
        except Exception as exc:
            call.update(status=getattr(exc, "status", "runtime_error"), error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            call["elapsed_seconds"] = time.monotonic() - start

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        payload = {"model": self.config["model"], "messages": messages, "temperature": 0,
                   "max_tokens": max_tokens or 768, "stream": False, "store": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        response = self.send(payload)
        return {"message": response["choices"][0]["message"], "usage": response.get("usage", {}),
                "cost": 0.0, "raw_response": response}


class RecordingOpener:
    """Wrap existing structured ChatClient without changing its prompts/repairs."""
    def __init__(self, transport):
        self.transport = transport

    def open(self, request, timeout=None):
        if request.full_url != self.transport.base + "/v1/chat/completions":
            raise ValueError("Unexpected ChatClient destination")
        return io.BytesIO(json.dumps(self.transport.send(json.loads(request.data))).encode())
