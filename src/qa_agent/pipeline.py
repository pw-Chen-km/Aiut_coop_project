"""Bounded evidence rounds, using router/retriever/generator contracts."""
from __future__ import annotations

from copy import deepcopy
import time

from .contracts import Generator, Retriever, Router
from .recovery import accumulate_evidence, recovery_feedback


class QAAgent:
    def __init__(self, router: Router, retriever: Retriever, generator: Generator, *,
                 top_k: int = 20, max_rounds: int = 2):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(max_rounds) is not int or not 1 <= max_rounds <= 5:
            raise ValueError("max_rounds must be between 1 and 5 (including the initial retrieval)")
        self.router, self.retriever, self.generator = router, retriever, generator
        self.top_k, self.max_rounds = top_k, max_rounds

    def answer(self, question: str, *, qid: str = "interactive", history=None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be non-empty")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("qid must be non-empty")
        started = time.monotonic()
        from .dialogue import navigation_question, retrieval_query, validate_history
        history = validate_history(history) if history is not None else None
        route_question = navigation_question(question, history)
        result = {"schema_version": "qa-agent-answer-v1", "qid": qid, "question": question,
                  "status": "failed", "answer": None, "answerable": None, "citations": [],
                  "stages": {}, "items": [], "context_items": [], "generation_items": [],
                  "rounds": [], "retrieval_scopes": [], "max_rounds": self.max_rounds,
                  "rounds_used": 0, "stop_reason": None, "missing_information": []}
        previous, exposed, feedback, last_abstention = [], [], None, None
        stage = "navigation"
        try:
            search_question, search_audit = question, None
            if history is not None:
                stage = "retrieval_query"
                result["history"] = deepcopy(history)
                search_question, search_audit = retrieval_query(question, history, self.retriever.encoder.tokenizer)
                result["retrieval_query"] = search_audit
            for number in range(1, self.max_rounds + 1):
                round_started = time.monotonic()
                current = {"round": number, "status": "failed", "stages": {},
                           "items": [], "context_items": [], "generation_items": [],
                           "feedback": deepcopy(feedback), "trigger": "initial" if number == 1 else "insufficient_context"}
                result["rounds"].append(current)
                result["rounds_used"] = number
                result.update(status="failed", answer=None, answerable=None, citations=[],
                              items=[], context_items=[], generation_items=[], missing_information=[])
                result["stages"] = current["stages"]  # Compatibility: latest-round view.
                try:
                    stage = "navigation"
                    route = (self.router.route(route_question, qid=qid) if feedback is None else
                             self.router.route(route_question, qid=qid, feedback=feedback))
                    current["stages"][stage] = route
                    if route.get("status") != "ok":
                        result["error"] = {"stage": stage, **route.get("error", {"message": "Navigation failed"})}
                        result["stop_reason"] = "navigation_error"
                        return result
                    scope = route["scope"]
                    if not isinstance(scope, dict) or not scope.get("doc_ids") or not scope.get("section_ids"):
                        raise ValueError("Navigation returned an empty scope")
                    current["scope"] = deepcopy(scope)
                    result["scope"] = deepcopy(scope)
                    result["retrieval_scopes"].append(deepcopy(scope))
                    stage = "retrieval"
                    search_started = time.monotonic()
                    items = self.retriever.search(search_question, scope, k=self.top_k)
                    result["items"] = current["items"] = items
                    current["stages"][stage] = {"status": "ok", "method": "scoped_dense", "count": len(items),
                                                "elapsed_seconds": time.monotonic() - search_started}
                    pooled, fresh = accumulate_evidence(previous, items, exposed)
                    result["generation_items"] = current["generation_items"] = pooled
                    current["new_evidence_keys"] = fresh
                    if number > 1 and not fresh:
                        current.update(status="ok", generation_skipped="no_new_evidence")
                        result.update(status="ok", answerable=False, answer=last_abstention["answer"],
                                      missing_information=last_abstention.get("missing_information", []),
                                      stop_reason="no_new_evidence")
                        return result
                    stage = "generation"
                    generated = (self.generator.generate(question, pooled, qid=qid) if history is None else
                                 self.generator.generate(question, pooled, qid=qid, history=history))
                    current["stages"][stage] = generated
                    result["context_items"] = current["context_items"] = generated.get("context_items", [])
                    if generated.get("status") != "ok":
                        result["error"] = {"stage": stage, **generated.get("error", {"message": "Generation failed"})}
                        result["stop_reason"] = "generation_error"
                        return result
                    if type(generated.get("answerable")) is not bool:
                        raise ValueError("Generator must return a boolean answerable flag")
                    current.update(status="ok", answer=generated["answer"], answerable=generated["answerable"],
                                   citations=deepcopy(generated.get("citations", [])))
                    result.update(status="ok", answer=generated["answer"], answerable=generated["answerable"],
                                  citations=generated.get("citations", []),
                                  missing_information=generated.get("missing_information", []))
                    if history is not None:
                        result["response_type"] = current["response_type"] = generated.get("response_type", "abstain")
                    if generated.get("response_type") == "clarify":
                        result["stop_reason"] = "clarification_requested"
                        return result
                    if generated["answerable"]:
                        result["stop_reason"] = "answered"
                        return result
                    if number == self.max_rounds:
                        result["stop_reason"] = "round_limit"
                        return result
                    stage = "recovery"
                    feedback = recovery_feedback(scope, generated)
                    # All earlier reads matter when revisiting a scope in round 3+.
                    from .recovery import merge_exposed
                    exposed = merge_exposed(exposed, generated.get("context_items", []))
                    feedback['seen_section_ids'] = list(dict.fromkeys(
                        item['section_id'] for item in exposed if item.get('section_id')))
                    previous, last_abstention = pooled, generated
                finally:
                    current["elapsed_seconds"] = time.monotonic() - round_started
        except Exception as exc:
            result.update(status="failed", answer=None, answerable=None, citations=[],
                          error={"stage": stage, "type": type(exc).__name__, "message": str(exc)},
                          stop_reason=f"{stage}_error")
            if result["rounds"]:
                result["rounds"][-1]["status"] = "failed"
        finally:
            if result.get("error") and result["rounds"]:
                result["rounds"][-1]["error"] = deepcopy(result["error"])
            result["elapsed_seconds"] = time.monotonic() - started
        return result
