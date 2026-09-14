"""Bounded evidence recovery bookkeeping, without models, storage, or gold labels."""
from __future__ import annotations

from copy import deepcopy


DEFAULT_GAP = "Find the missing source information needed to answer the original question completely."


def merge_exposed(previous, current):
    """Keep every actual context exposure, including distinct clipped spans."""
    import json
    seen, out = set(), []
    for item in [*previous, *current]:
        identity = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if identity not in seen:
            out.append(deepcopy(item))
            seen.add(identity)
    return out


def recovery_feedback(scope: dict, generated: dict) -> dict:
    gaps = generated.get("missing_information", [])
    if (not isinstance(gaps, list) or len(gaps) > 3
            or any(not isinstance(s, str) or not s.strip() or len(s) > 240 for s in gaps)):
        raise ValueError("Recovery gaps must be at most three short information needs")
    seen = list(dict.fromkeys(item["section_id"] for item in generated.get("context_items", [])
                             if item.get("section_id")))
    return {"previous_scope": deepcopy(scope), "seen_section_ids": seen,
            "missing_information": gaps or [DEFAULT_GAP]}


def accumulate_evidence(previous: list[dict], current: list[dict], exposed: list[dict]):
    """New/unread passages first, then retained originals; never turn summaries into evidence.

    Scores/ranks remain those of each original retrieval. This is an answer input
    ordering, not a newly computed dense ranking. Return its explicit chunk keys
    so no-progress detection does not depend on an LLM promise.
    """
    def key(item):
        return item["doc_id"], item["chunk_id"]

    seen = {key(item) for item in exposed}
    fresh, retained, originals = [], [], {}
    for item in [*current, *previous]:
        identity = key(item)
        if identity in originals and originals[identity] != item["text"]:
            raise ValueError("The original text changed between retrieval rounds")
        if identity in originals:
            continue
        originals[identity] = item["text"]
        (fresh if identity not in seen else retained).append(deepcopy(item))
    return fresh + retained, [list(key(item)) for item in fresh]
