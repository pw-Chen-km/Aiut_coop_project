"""Per-round source diagnostics and compact, optimizer-only representations.

The compact view is derived from, and fingerprinted against, independent raw
logs. It is never a replacement for execution logs or evidence verification.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from .data import digest
from .oracle import normalize_with_offsets, project_chunk_slice


def _covers(required, spans):
    for need in required:
        intervals = sorted((max(need["start_char"], s["start_char"]), min(need["end_char"], s["end_char"]))
                           for s in spans if s["block_id"] == need["block_id"] and s.get("page_index") == need.get("page_index"))
        end = need["start_char"]
        for a, b in intervals:
            if b <= a:
                continue
            if a > end:
                break
            end = max(end, b)
        if end < need["end_char"]:
            return False
    return bool(required)


def _target_text_supported(target, spans):
    """Claimed offsets alone are insufficient: reproduce the canonical quote."""
    pieces = []
    for need in target["source_spans"]:
        characters = {}
        for span in spans:
            if span["block_id"] != need["block_id"] or span.get("page_index") != need.get("page_index"):
                continue
            text = span.get("text")
            if not isinstance(text, str) or len(text) != span["end_char"] - span["start_char"]:
                continue
            for offset in range(max(need["start_char"], span["start_char"]), min(need["end_char"], span["end_char"])):
                character = text[offset - span["start_char"]]
                if offset in characters and characters[offset] != character:
                    return False
                characters[offset] = character
        if any(i not in characters for i in range(need["start_char"], need["end_char"])):
            return False
        pieces.append("".join(characters[i] for i in range(need["start_char"], need["end_char"])))
    expected = target.get("canonical_text")
    if expected is None:
        excerpts = target.get("chunk_excerpts", [])
        expected = excerpts[0]["text"] if excerpts else None
    if expected is None:
        return False
    return normalize_with_offsets(" ".join(pieces))[0] == normalize_with_offsets(expected)[0]


def evidence_coverage(oracle_query, *, spans_by_document=None, section_scope=None):
    """Route-aware full coverage, considering declared alternative targets."""
    satisfied, known = set(), set()
    for evidence in oracle_query.get("evidence", []):
        key = evidence.get("equivalence_group") or evidence["evidence_id"]
        targets = evidence.get("targets", [])
        if targets:
            known.add(key)
        for target in targets:
            if spans_by_document is not None:
                returned = spans_by_document.get(target["doc_id"], [])
                match = _covers(target["source_spans"], returned) and _target_text_supported(target, returned)
            else:
                match = bool(target.get("section_ids")) and set(target["section_ids"]) <= \
                        set((section_scope or {}).get(target["doc_id"], []))
            if match:
                satisfied.add(key)
                break
    routes = oracle_query.get("routes", [])
    scored = []
    for route in routes:
        required = set(route.get("required_keys", []))
        if not required:
            continue
        scored.append({"set_id": route.get("set_id"), "required_keys": sorted(required),
                       "covered_keys": sorted(required & satisfied), "unknown_keys": sorted(required - known),
                       "recall": len(required & satisfied) / len(required), "complete": required <= satisfied})
    if not scored:
        return {"complete": False, "recall": None, "unknown_keys": [], "status": "unknown_gold_route"}
    best = max(scored, key=lambda route: (route["complete"], route["recall"], -len(route["unknown_keys"])))
    best["status"] = "known" if not best["unknown_keys"] else "partial_oracle"
    return best


def _expanded_scope(scope, sections):
    result = defaultdict(set)
    allowed_docs = set(scope.get("doc_ids", []))
    selected = set(scope.get("section_ids", []))
    for sid, section in sections.items():
        if section.get("doc_id") not in allowed_docs:
            continue
        current, seen = sid, set()
        while current:
            if current in seen:
                raise ValueError("Section tree contains a cycle")
            seen.add(current)
            if current in selected:
                result[section["doc_id"]].add(sid)
                break
            if current not in sections:
                raise ValueError("Section tree references an unknown parent")
            current = sections[current].get("parent_id")
    return result


def _item_spans(items, errors):
    spans = defaultdict(list)
    for item in items:
        try:
            projected = project_chunk_slice(item, 0, len(item.get("text", "")))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"chunk_id": item.get("chunk_id"), "reason": str(exc)})
            continue
        spans[item.get("doc_id")].extend(projected)
    return spans


def _exposed_spans(contexts, lookup, errors):
    result = defaultdict(list)
    for context in contexts:
        key = (context.get("doc_id"), context.get("chunk_id"))
        original = lookup.get(key)
        if original is None:
            errors.append({"chunk_id": key[1], "reason": "exposed_chunk_missing_from_observed_retrieval"})
            continue
        start = context.get("excerpt_start_char", 0)
        end = context.get("excerpt_end_char", start + len(context.get("text", "")))
        if type(start) is not int or type(end) is not int or context.get("text") != original["text"][start:end]:
            errors.append({"chunk_id": key[1], "reason": "exposed_text_disagrees_with_observed_chunk_slice"})
            continue
        try:
            result[key[0]].extend(project_chunk_slice(original, start, end))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"chunk_id": key[1], "reason": str(exc)})
    return result


def _answer_correct(judgment):
    if not isinstance(judgment, dict):
        return None
    for key in ("answer_correct", "correct", "is_correct"):
        if type(judgment.get(key)) is bool:
            return judgment[key]
    # Provisional semantic judge labels are consumed as prototype diagnostics,
    # never converted into the human-reviewed claim-label schema.
    if judgment.get("verdict") in {"correct", "incorrect"}:
        return judgment["verdict"] == "correct"
    return None


def diagnose_trajectory(trace, oracle_query, sections, judgment=None):
    """Distinguish route, retrieval, exposure and answer failures per round.

    Previous scopes/evidence are preserved for the finite-recovery pooled view.
    Unknown oracle evidence prevents a confident stage-blame assignment.
    """
    if trace.get("qid") != oracle_query.get("qid"):
        raise ValueError("Trajectory and oracle qid disagree")
    registry = sections if isinstance(sections, dict) else {s["section_id"]: s for s in sections}
    rounds = trace.get("rounds") or [trace]
    output, observed, cumulative_scope, cumulative_documents = [], {}, defaultdict(set), defaultdict(set)
    cumulative_retrieved, cumulative_exposed = defaultdict(list), defaultdict(list)
    for index, current in enumerate(rounds, 1):
        errors = []
        for item in [*current.get("items", []), *current.get("generation_items", [])]:
            key = (item.get("doc_id"), item.get("chunk_id"))
            if key in observed and observed[key].get("text") != item.get("text"):
                raise ValueError("Raw trajectory contains conflicting texts for one source chunk")
            observed[key] = item
        scope = current.get("scope", {})
        expanded = _expanded_scope(scope, registry)
        document_scope = {doc: {sid for sid, section in registry.items() if section.get("doc_id") == doc}
                          for doc in scope.get("doc_ids", [])}
        for doc, ids in document_scope.items():
            cumulative_documents[doc].update(ids)
        for doc, ids in expanded.items():
            cumulative_scope[doc].update(ids)
        contexts = current.get("context_items", [])
        if not contexts:
            contexts = current.get("stages", {}).get("generation", {}).get("context_items", [])
        retrieved = _item_spans(current.get("items", []), errors)
        pooled = _item_spans(current.get("generation_items", current.get("items", [])), errors)
        exposed = _exposed_spans(contexts, observed, errors)
        for doc, spans in retrieved.items():
            cumulative_retrieved[doc].extend(spans)
        for doc, spans in exposed.items():
            cumulative_exposed[doc].extend(spans)
        entry = {"round": current.get("round", index), "status": current.get("status", trace.get("status")),
                 "scope": deepcopy(scope), "scope_coverage": evidence_coverage(oracle_query, section_scope=expanded),
                 "document_coverage": evidence_coverage(oracle_query, section_scope=document_scope),
                 "cumulative_document_coverage": evidence_coverage(oracle_query, section_scope=cumulative_documents),
                 "cumulative_scope_coverage": evidence_coverage(oracle_query, section_scope=cumulative_scope),
                 "retrieved_coverage": evidence_coverage(oracle_query, spans_by_document=retrieved),
                 "cumulative_retrieved_coverage": evidence_coverage(oracle_query, spans_by_document=cumulative_retrieved),
                 "pooled_coverage": evidence_coverage(oracle_query, spans_by_document=pooled),
                 "exposed_coverage": evidence_coverage(oracle_query, spans_by_document=exposed),
                 "cumulative_exposed_coverage": evidence_coverage(oracle_query, spans_by_document=cumulative_exposed),
                 "answerable": current.get("answerable"), "provenance_errors": errors}
        if current.get("generation_skipped"):
            entry["generation_skipped"] = current["generation_skipped"]
        output.append(entry)
    # A no-new-evidence round does not erase the preceding actual generation.
    generated = [r for r in output if not r.get("generation_skipped")]
    final = generated[-1] if generated else output[-1]
    correct = _answer_correct(judgment)
    if trace.get("status") != "ok":
        outcome = "execution_failure"
    elif not oracle_query.get("complete_route_available"):
        outcome = "oracle_unresolved"
    elif not final["cumulative_document_coverage"]["complete"]:
        outcome = "document_selection"
    elif not final["cumulative_scope_coverage"]["complete"]:
        outcome = "section_selection"
    elif not final["pooled_coverage"]["complete"]:
        outcome = "retrieval_miss"
    elif not final["exposed_coverage"]["complete"]:
        outcome = "exposure_miss"
    elif correct is None:
        outcome = "judgment_pending"
    else:
        outcome = "success" if correct else "answer_failure"
    return {"schema_version": "qa-skillopt-trajectory-diagnostic-v1", "qid": trace["qid"],
            "raw_trace_sha256": digest(trace), "oracle_status": oracle_query.get("status"),
            "rounds": output, "failure_stage": outcome, "answer_correct": correct,
            "judgment": deepcopy(judgment), "stop_reason": trace.get("stop_reason"),
            "limitations": "Full exact registered source support only; an unresolved oracle is not evidence of routing failure. Prototype judge labels are not SME review."}


def _observed_descriptions(current, max_chars):
    snippets, remaining = [], max_chars
    navigation = current.get("stages", {}).get("navigation", {})
    for event in navigation.get("trace", []):
        for message in event.get("messages", []):
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str):
                continue
            # Preserve prose summaries as well as headers/inventory. The exact
            # observed Markdown is data, never a regenerated current-file view.
            markers = [content.find(marker) for marker in ("OVERALL NAVIGATION", "DOCUMENT NAVIGATION") if marker in content]
            text = content[min(markers):] if markers else content
            if remaining <= 0:
                break
            shown = text[:remaining]
            snippets.append({"stage": event.get("stage"), "observed_description_text": shown,
                             "truncated": len(shown) < len(text), "observed_prompt_sha256": digest(content),
                             "offered_choices": {key: deepcopy(value.get("items", {}).get("enum", []))
                                                 for key, value in event.get("schema", {}).get("properties", {}).items()},
                             "selected_choices": deepcopy(event.get("response", {}).get("data", {}))})
            remaining -= len(shown)
    return snippets


def compact_trajectory(trace, diagnostics=None, *, phase="optimization", max_evidence_chars=12000,
                       max_description_chars=6000):
    """Return a bounded optimizer view containing actual observed source text."""
    if phase not in {"optimization", "validation"}:
        raise ValueError("Test trajectories cannot enter optimizer feedback")
    if any(type(v) is not int or v < 1 for v in (max_evidence_chars, max_description_chars)):
        raise ValueError("Compact-view budgets must be positive integers")
    remaining, description_remaining, rounds, descriptions = max_evidence_chars, max_description_chars, [], []
    for current in trace.get("rounds") or [trace]:
        observed_descriptions = _observed_descriptions(current, description_remaining)
        descriptions.extend(observed_descriptions)
        description_remaining -= sum(len(d["observed_description_text"]) for d in observed_descriptions)
        actual = []
        for context in current.get("context_items", []):
            text = context.get("text", "")
            shown = text[:max(0, remaining)]
            if shown:
                actual.append({**{k: deepcopy(context.get(k)) for k in ("citation_id", "doc_id", "section_id", "chunk_id", "pages", "excerpt_start_char", "excerpt_end_char")},
                               "text": shown, "compact_truncated": len(shown) < len(text),
                               "observed_excerpt_sha256": digest(text)})
                remaining -= len(shown)
        rounds.append({"round": current.get("round", len(rounds) + 1), "scope": deepcopy(current.get("scope", {})),
                       "retrieved_chunk_ids": [i.get("chunk_id") for i in current.get("items", [])],
                       "actual_exposed_evidence": actual, "answerable": current.get("answerable"),
                       "feedback": deepcopy(current.get("feedback")),
                       "missing_information": deepcopy(current.get("missing_information", current.get("stages", {}).get("generation", {}).get("missing_information", [])))})
    return {"schema_version": "qa-skillopt-compact-trajectory-v1", "optimizer_only": True,
            "phase": phase, "qid": trace.get("qid"), "question": trace.get("question"),
            "answer": trace.get("answer"), "answerable": trace.get("answerable"),
            "stop_reason": trace.get("stop_reason"), "raw_trace_sha256": digest(trace),
            "observed_skill_descriptions": descriptions, "rounds": rounds,
            "diagnostics": deepcopy(diagnostics),
            "note": "Derived bounded view only; retain the separate immutable raw trace for replay and auditing."}
