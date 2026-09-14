"""Minimal HTTP adapter for the A navigation skill.

The request and response payloads are the same qid/question JSON used by
qa-agent. Algorithm code remains in src/ and can be replaced independently.
"""
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qa_agent.bundle import validate_serving_bundle

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health/live": self._send(200, {"status":"ok"})
        elif self.path == "/health/ready":
            try: validate_serving_bundle(os.environ["QA_BUNDLE_DIR"]); self._send(200, {"status":"ready"})
            except Exception as e: self._send(503, {"status":"not_ready", "error":str(e)})
        else: self._send(404, {"error":"not_found"})
    def do_POST(self):
        if self.path != "/v1/answer": self._send(404, {"error":"not_found"}); return
        try:
            n=int(self.headers.get("Content-Length", "0")); row=json.loads(self.rfile.read(n)); q=row.get("question")
            if not isinstance(q,str) or not q.strip(): raise ValueError("question must be nonempty text")
            from qa_agent.factory import create_session, load_online_config
            with create_session(load_online_config(os.environ["QA_CONFIG"])) as session:
                result=session.agent.answer(q, qid=row.get("qid","interactive"), history=row.get("history")) if "history" in row else session.agent.answer(q, qid=row.get("qid","interactive"))
            self._send(200, result)
        except Exception as e: self._send(500, {"status":"failed", "error_type":type(e).__name__, "message":str(e)})

if __name__ == "__main__":
    ThreadingHTTPServer((os.getenv("QA_HOST","0.0.0.0"), int(os.getenv("QA_PORT","8080"))), Handler).serve_forever()
