"""Verbatim conversation input, bounded dense query and dialogue response mode."""
from __future__ import annotations
import copy
import hashlib
import json
from doc2skill.llm import InvalidOutputError, validate_schema


def validate_history(history):
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    result, last_turn = [], None
    for row in history:
        if (not isinstance(row, dict) or set(row) - {"role", "content", "turn_id"}
                or row.get("role") not in {"user", "agent", "assistant"}
                or not isinstance(row.get("content"), str) or not row["content"].strip()):
            raise ValueError("History accepts role/content and optional turn_id only")
        if "turn_id" in row:
            turn = row["turn_id"]
            if type(turn) is not int or (last_turn is not None and turn <= last_turn):
                raise ValueError("History turn IDs must be chronological")
            last_turn = turn
        result.append(copy.deepcopy(row))
    return result


def navigation_question(question, history):
    return question if history is None else json.dumps({"current_user_utterance": question,
        "prior_dialogue": validate_history(history)}, ensure_ascii=False)


def retrieval_query(question, history, tokenizer, limit=512):
    history = validate_history(history)
    def render(rows):
        return question if not rows else "CURRENT QUESTION\n" + question + "\nPRIOR DIALOGUE\n" + "\n".join(
            row["role"] + ": " + row["content"] for row in rows)
    def count(text):
        return len(tokenizer.encode("query: " + text, add_special_tokens=True))
    if count(question) > limit:
        raise ValueError("Current utterance alone exceeds the embedding budget")
    selected = []
    for row in reversed(history):
        candidate = [row, *selected]
        if count(render(candidate)) > limit:
            break
        selected = candidate
    text = render(selected)
    return text, {"query": text, "embedding_tokens": count(text), "included_history_count": len(selected),
        "omitted_history_indices": list(range(len(history) - len(selected))), "query_rewritten": False}


def generate_dialogue(base, question, items, qid, history):
    from .generation import AnswerGenerator
    class DialogueGenerator(AnswerGenerator):
        dialogue_mode = True

        def _messages(self, question, contexts):
            return [{"role": "system", "content": self.prompt}, {"role": "user", "content": json.dumps({
                "current_user_utterance": question, "prior_dialogue": self.history,
                "original_passages": contexts}, ensure_ascii=False)}]

        def _schema(self, contexts):
            schema = super()._schema(contexts)
            if not contexts:
                # An empty enum is not a valid constrained-decoding grammar.
                schema["properties"]["citation_ids"] = {
                    "type": "array", "items": {"type": "string"}, "maxItems": 0}
            schema["properties"]["response_type"] = {"type": "string", "enum": ["answer", "clarify", "abstain"]}
            schema["required"].append("response_type")
            return schema

        def _validate_answer(self, data, schema):
            validate_schema(data, schema)
            kind = data["response_type"]
            if not data["answer"].strip() or data["answerable"] != (kind == "answer"):
                raise InvalidOutputError("Response type and answerable disagree")
            if kind == "answer" and not data["citation_ids"]:
                raise InvalidOutputError("Factual answers require retrieved citations")
            if kind != "answer" and data["citation_ids"]:
                raise InvalidOutputError("Clarifications/refusals must not assert a cited answer")
            if kind != "abstain" and data.get("missing_information"):
                raise InvalidOutputError("Only abstention may request retrieval recovery")
            if len(set(data["citation_ids"])) != len(data["citation_ids"]):
                raise InvalidOutputError("Duplicate citations")
    instance = DialogueGenerator(base.client, base.config, token_counter=base.token_counter, prompt_content=base.prompt)
    instance.history = validate_history(history)
    instance.prompt = base.prompt + (
        "\nDIALOGUE MODE: Respond to the current user using prior conversation to resolve intent. "
        "History is user context, not authoritative source evidence. If a user-specific condition is missing, "
        "ask a concise clarifying question: response_type=clarify, answerable=false, no citations or missing_information. "
        "Clarification is a completed conversational turn, not retrieval failure. Use response_type=answer only for "
        "a source-grounded factual answer. If sources are insufficient use response_type=abstain, answerable=false, "
        "and explain the limitation without unsupported facts. In dialogue mode answer is the actual reply, "
        "including clarification or refusal. Return no reasoning.")
    instance.prompt_sha256 = hashlib.sha256(instance.prompt.encode()).hexdigest()
    return instance.generate(question, items, qid=qid)
