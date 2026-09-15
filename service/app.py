"""HTTP API and browser test console for the A navigation skill."""
from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
sys.path.insert(0, str(ROOT / "src"))

from qa_agent.bundle import validate_serving_bundle


def _session_factory(config):
    """Select the runtime matching the immutable embedding in the A bundle."""
    manifest = validate_serving_bundle(config["bundle_dir"])
    if manifest.get("embedding", {}).get("encoder") in {"qwen3_remote", "bge"}:
        from qa_agent.factory import create_comparison_session
        return create_comparison_session
    from qa_agent.factory import create_session
    return create_session


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, relative: str) -> None:
        path = (WEB_ROOT / relative).resolve()
        if not path.is_relative_to(WEB_ROOT.resolve()) or not path.is_file():
            self._json(404, {"error": "not_found"})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            self._static("index.html")
        elif path == "/styles.css":
            self._static("styles.css")
        elif path == "/app.js":
            self._static("app.js")
        elif path == "/health/live":
            self._json(200, {"status": "ok"})
        elif path == "/health/ready":
            try:
                validate_serving_bundle(os.environ["QA_BUNDLE_DIR"])
                self._json(200, {"status": "ready"})
            except Exception as exc:
                self._json(503, {"status": "not_ready", "error": str(exc)})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/v1/answer":
            self._json(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 1_000_000:
                raise ValueError("request body must contain at most 1 MB of JSON")
            row = json.loads(self.rfile.read(size))
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question must be nonempty text")
            from qa_agent.factory import load_online_config

            config = load_online_config(os.environ["QA_CONFIG"])
            with _session_factory(config)(config) as session:
                arguments = {"qid": row.get("qid", "interactive")}
                if "history" in row:
                    arguments["history"] = row["history"]
                result = session.agent.answer(question.strip(), **arguments)
            self._json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            self._json(500, {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    address = (os.getenv("QA_HOST", "0.0.0.0"), int(os.getenv("QA_PORT", "8080")))
    print(f"Navigation RAG test console: http://127.0.0.1:{address[1]}", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
