"""One answer prompt and Qwen model for all retrieval conditions."""
import json


class AnswerGenerator:
    def __init__(self, model, model_id, system_prompt, context_tokens=8192, max_tokens=512):
        self.model, self.model_id, self.system = model, model_id, system_prompt
        self.context_tokens, self.max_tokens = context_tokens, max_tokens

    def generate(self, query, chunks):
        if not chunks:
            return {"response_type": "abstain", "answer": "I can't answer this", "sources": []}
        passages = [{"citation_id": f"C{i:03d}", "text": c["text"]} for i, c in enumerate(chunks, 1)]
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": json.dumps({"question": query, "passages": passages})}]
        prompt_tokens = len(self.model.tokenize(self.model.render(messages)))
        if prompt_tokens + self.max_tokens > self.context_tokens:
            raise ValueError("answer context exceeds configured model budget; no passage truncation")
        citation_ids = [p["citation_id"] for p in passages]
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "response_type": {"type": "string", "enum": ["answer", "clarify", "abstain"]},
                "answer": {"type": "string"},
                "citation_ids": {"type": "array", "uniqueItems": True,
                                 "items": {"type": "string", "enum": citation_ids}},
            }, "required": ["response_type", "answer", "citation_ids"],
        }
        payload = self.model.post("/v1/chat/completions", {
            "model": self.model_id, "messages": messages, "temperature": 0, "seed": 0,
            "max_tokens": self.max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        })
        if payload["choices"][0].get("finish_reason") == "length":
            raise ValueError("answer generation hit token limit")
        result = json.loads(payload["choices"][0]["message"]["content"])
        kind, cited = result.get("response_type"), result.get("citation_ids")
        if kind not in {"answer", "clarify", "abstain"} or not isinstance(cited, list):
            raise ValueError("invalid answer schema")
        if not isinstance(result.get("answer"), str) or not result["answer"].strip():
            raise ValueError("blank answer")
        if any(c not in citation_ids for c in cited) or len(set(cited)) != len(cited):
            raise ValueError("unknown or duplicate citation")
        if (kind == "answer") != bool(cited):
            raise ValueError("answer requires citations; clarify/abstain cannot have citations")
        sources = [{"citation_id": p["citation_id"], **chunk}
                   for p, chunk in zip(passages, chunks) if p["citation_id"] in cited]
        return {"response_type": kind, "answer": "I can't answer this" if kind == "abstain" else result["answer"],
                "sources": sources, "prompt_tokens": prompt_tokens}

