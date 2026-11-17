"""Read-only client for an already-running llama.cpp server."""
import httpx


class ModelClient:
    def __init__(self, endpoint, timeout=180, client=None):
        self.endpoint = endpoint.rstrip("/").removesuffix("/v1")
        self.client = client or httpx.Client(timeout=timeout)

    def get(self, path):
        response = self.client.get(self.endpoint + path)
        response.raise_for_status()
        return response.json()

    def post(self, path, body):
        response = self.client.post(self.endpoint + path, json=body)
        response.raise_for_status()
        return response.json()

    def identity(self):
        data = self.get("/v1/models").get("data", [])
        if len(data) != 1 or not data[0].get("id"):
            raise ValueError("exactly one active model is required")
        props = self.get("/props")
        return {"id": data[0]["id"], "model_path": props.get("model_path"),
                "build_info": props.get("build_info"), "chat_template": props.get("chat_template")}

    def render(self, messages):
        payload = self.post("/apply-template", {
            "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("server does not expose its chat-template prompt")
        # A Qwen template ignoring enable_thinking must fail rather than score
        # candidates inside an open reasoning block.
        if prompt.rfind("<think>") > prompt.rfind("</think>"):
            raise ValueError("server chat template left thinking enabled")
        return prompt

    def tokenize(self, text):
        tokens = self.post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        if not tokens or any(type(t) is not int for t in tokens):
            raise ValueError("tokenize endpoint did not return token IDs")
        return tokens

    def close(self):
        self.client.close()

