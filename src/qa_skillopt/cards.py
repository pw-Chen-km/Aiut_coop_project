"""Deterministic, field-restricted Markdown navigation cards.

The runtime consumes the rendered Markdown directly.  SkillOPT may change only
the six routing fields below; identity, hierarchy, titles and formatting are a
compiler-owned scaffold.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .oracle import load_corpus


SCHEMA_VERSION = "qursor-navigation-cards-v2"
EDITABLE_FIELDS = (
    "use_when",
    "not_when",
    "question_intents",
    "aliases",
    "confusable_with",
    "distinguishing_signals",
)
STATIC_FIELDS = ("card_id", "title", "parent_card_id")
LIST_LIMITS = {
    "use_when": (4, 180),
    "not_when": (4, 180),
    "question_intents": (4, 180),
    "aliases": (8, 80),
    "confusable_with": (4, 256),
    "distinguishing_signals": (4, 180),
}


class CardFormatError(ValueError):
    """Raised when a candidate changes the compiler-owned card scaffold."""


def _list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [" ".join(value.split())] if value.strip() else []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CardFormatError("navigation metadata fields must be strings or string lists")
    return [" ".join(item.split()) for item in value if item.strip()]


def _editable(meta: dict, *, section_seed=False) -> dict:
    use_when = _list(meta.get("use_when") or meta.get("navigation_summary"))
    question_intents = _list(meta.get("question_intents") or meta.get("navigation_intents"))
    # The current compiler metadata often stores the exact same routing cue in
    # both fields.  Keep the information once so two selected document cards
    # still fit the frozen 4B context budget; the empty field remains editable.
    if question_intents == use_when:
        question_intents = []
    return {
        "use_when": use_when,
        # Section-level exclusions are numerous and made legal two-document
        # routing exceed the frozen 4B context budget.  They remain in immutable
        # metadata.json and can be added selectively by SkillOPT when a real
        # confusion is observed.  Overall document exclusions stay in the seed.
        "not_when": [] if section_seed else _list(meta.get("not_when") or meta.get("not_covered")),
        "question_intents": question_intents,
        "aliases": _list(meta.get("aliases")),
        "confusable_with": _list(meta.get("confusable_with")),
        "distinguishing_signals": _list(meta.get("distinguishing_signals")),
    }


def _header(target: str) -> list[str]:
    return [
        "# QURSOR navigation cards",
        "",
        f"Schema: `{SCHEMA_VERSION}`",
        f"Target: `{target}`",
        "",
        "Routing only; not answer evidence. Edit only the six routing fields.",
        "",
    ]


def render_card_skill(target: str, cards: list[dict]) -> str:
    """Render compact canonical Markdown without duplicating empty fields.

    Each card has an immutable three-value tuple followed by a sparse object of
    the six editable fields.  The parser restores omitted fields as empty lists.
    This is materially smaller than a full JSON object per section and is the
    actual text read by the 4B router.
    """
    lines = _header(target)
    for card in cards:
        fixed = [deepcopy(card.get(key)) for key in STATIC_FIELDS]
        sparse = {key: deepcopy(card.get(key, [])) for key in EDITABLE_FIELDS if card.get(key, [])}
        lines.append("- " + json.dumps(fixed, ensure_ascii=False, separators=(",", ":")) + " "
                     + json.dumps(sparse, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines).rstrip() + "\n"


def parse_card_skill(text: str) -> dict:
    """Parse and canonicalize a navigation-card Markdown document."""
    if not isinstance(text, str) or not text.strip():
        raise CardFormatError("empty_navigation_cards")
    lines = text.splitlines()
    if len(lines) < 8 or lines[0] != "# QURSOR navigation cards":
        raise CardFormatError("not_navigation_card_markdown")
    if lines[2] != f"Schema: `{SCHEMA_VERSION}`" or not lines[3].startswith("Target: `") or not lines[3].endswith("`"):
        raise CardFormatError("changed_navigation_card_header")
    target = lines[3][9:-1]
    expected_header = _header(target)
    if lines[:len(expected_header)] != expected_header:
        raise CardFormatError("changed_navigation_card_header")
    cards, cursor = [], len(expected_header)
    decoder = json.JSONDecoder()
    while cursor < len(lines):
        if not lines[cursor].startswith("- "):
            raise CardFormatError("changed_navigation_card_scaffold")
        try:
            payload = lines[cursor][2:]
            fixed, offset = decoder.raw_decode(payload)
            while offset < len(payload) and payload[offset].isspace():
                offset += 1
            sparse, end = decoder.raw_decode(payload, offset)
        except (TypeError, ValueError) as exc:
            raise CardFormatError("invalid_navigation_card_json") from exc
        if payload[end:].strip() or not isinstance(fixed, list) or len(fixed) != len(STATIC_FIELDS):
            raise CardFormatError("changed_navigation_card_fixed_fields")
        if (not isinstance(sparse, dict) or set(sparse) - set(EDITABLE_FIELDS)
                or any(not value for value in sparse.values())):
            raise CardFormatError("changed_navigation_card_fields")
        card = dict(zip(STATIC_FIELDS, fixed))
        card.update({key: deepcopy(sparse.get(key, [])) for key in EDITABLE_FIELDS})
        cards.append(card)
        cursor += 1
    if not cards:
        raise CardFormatError("empty_navigation_card_inventory")
    parsed = {"schema_version": SCHEMA_VERSION, "target": target, "cards": cards}
    if render_card_skill(target, cards) != text:
        raise CardFormatError("noncanonical_navigation_card_markdown")
    return parsed


def validate_card_values(parsed: dict) -> list[str]:
    errors, cards = [], parsed.get("cards", [])
    ids = [card.get("card_id") for card in cards]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        errors.append("invalid_or_duplicate_card_id")
    allowed = set(ids)
    parents = {card.get("card_id"): card.get("parent_card_id") for card in cards}
    for card in cards:
        for field in EDITABLE_FIELDS:
            value = card.get(field)
            count_limit, char_limit = LIST_LIMITS[field]
            if (not isinstance(value, list) or len(value) > count_limit
                    or any(not isinstance(item, str) or not item.strip() or item != " ".join(item.split())
                           or len(item) > char_limit for item in value)
                    or len(set(item.casefold() for item in value if isinstance(item, str))) != len(value)):
                errors.append("invalid_" + field)
        confusing = card.get("confusable_with", [])
        if any(value not in allowed or value == card.get("card_id")
               or parents.get(value) != card.get("parent_card_id") for value in confusing):
            errors.append("invalid_confusable_with_reference")
    return sorted(set(errors))


def compare_card_skills(before: str, after: str, *, expected_target: str) -> dict:
    """Return semantic field edits or fail if immutable scaffold changed."""
    old, new = parse_card_skill(before), parse_card_skill(after)
    errors = []
    if old["target"] != expected_target or new["target"] != expected_target:
        errors.append("changed_navigation_card_target")
    errors.extend(validate_card_values(old))
    errors.extend(validate_card_values(new))
    old_ids = [card["card_id"] for card in old["cards"]]
    new_ids = [card["card_id"] for card in new["cards"]]
    if old_ids != new_ids:
        errors.append("changed_navigation_card_inventory_or_order")
    edits = []
    for left, right in zip(old["cards"], new["cards"]):
        if any(left.get(key) != right.get(key) for key in STATIC_FIELDS):
            errors.append("changed_navigation_card_identity_or_hierarchy")
            continue
        for field in EDITABLE_FIELDS:
            if left[field] != right[field]:
                edits.append({"card_id": left["card_id"], "field": field,
                              "before": deepcopy(left[field]), "after": deepcopy(right[field])})
    if len(edits) > 3:
        errors.append("navigation_card_edit_budget_exceeded")
    return {"valid": not errors, "errors": sorted(set(errors)), "edits": edits,
            "before": old, "after": new}


def build_navigation_card_snapshot(bundle_dir) -> dict[str, str]:
    """Compile immutable corpus metadata into the baseline card snapshot."""
    root = Path(bundle_dir).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    paths = json.loads((root / "skill_paths.json").read_text(encoding="utf-8"))
    corpus = load_corpus(root)
    aliases = paths.get("section_aliases", {})
    route_for = {canonical: alias for alias, canonical in aliases.items()}
    documents = {doc["doc_id"]: doc for doc in corpus["documents"]}
    sections = {section["section_id"]: section for section in corpus["sections"]}

    overall_cards = []
    for doc in corpus["documents"]:
        did = doc["doc_id"]
        overall_cards.append({
            "card_id": did,
            "title": " ".join(str(doc.get("title", did)).split()),
            "parent_card_id": None,
            **_editable(metadata["documents"][did]),
        })
    snapshot = {"overall": render_card_skill("overall", overall_cards)}

    for did in documents:
        cards = []
        for section in corpus["sections"]:
            if section["doc_id"] != did:
                continue
            sid = section["section_id"]
            cards.append({
                # Use the exact short ID offered to the 4B response schema as
                # the card identity.  Keep the immutable canonical section ID
                # separately so gold routing feedback can still be matched.
                "card_id": route_for.get(sid, sid),
                "title": " ".join(str(section.get("title", sid)).split()),
                "parent_card_id": route_for.get(section.get("parent_id"), section.get("parent_id")),
                **_editable(metadata["sections"][sid], section_seed=True),
            })
        snapshot["document:" + did] = render_card_skill("document:" + did, cards)
    return snapshot
