"""Fail-closed navigation-card validation, separate from optimizer and runtime.

Structural checks never manufacture grounding. Acceptance requires an injected
corpus-only verifier and exact, independently checked original-source references.
The leak checks are conservative heuristics, not a proof against memorization.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3

from qa_agent.bundle import local_asset, validate_serving_bundle


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_navigation_snapshot(bundle_dir) -> dict[str, str]:
    root = Path(bundle_dir).resolve()
    validate_serving_bundle(root)
    paths = json.loads((root / "skill_paths.json").read_text(encoding="utf-8"))
    names = {"overall": paths["overall_doc_skill"],
             **{"document:" + did: path for did, path in paths["documents"].items()}}
    return {key: local_asset(root, path).read_text(encoding="utf-8") for key, path in names.items()}


def _registries(bundle_dir):
    root = Path(bundle_dir).resolve()
    with sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True) as conn:
        docs = [json.loads(p) for (p,) in conn.execute("SELECT payload FROM documents ORDER BY ordinal")]
        sections = [json.loads(p) for (p,) in conn.execute("SELECT payload FROM sections ORDER BY ordinal")]
    paths = json.loads((root / "skill_paths.json").read_text(encoding="utf-8"))
    return docs, sections, paths


def _section_tree(text: str) -> tuple[dict, list[str]]:
    """Parse explicit Section entries; formatting may change, identity/ancestry may not."""
    tree, order, stack = {}, [], []
    for line in text.splitlines():
        match = re.match(r"^(\s*)([-+*]|\d+[.)]|#{1,6})\s+(?:\*\*)?Section\s+"
                         r"(`?[^\s:`*]+`?)(?:\*\*)?\s*:\s*(\S.*)$", line)
        if not match:
            continue
        indent, bullet, alias, description = match.groups()
        alias = alias.strip("`")
        depth = len(bullet) if bullet.startswith("#") else len(indent.expandtabs(4))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if alias in tree:
            raise ValueError("duplicate_section_entry")
        tree[alias] = stack[-1][1] if stack else None
        order.append(alias)
        stack.append((depth, alias))
    return tree, order


def validate_structure(before: str, after: str, *, target_key: str, bundle_dir) -> list[str]:
    """Freeze registry IDs, inventory membership, ancestry and asset links, not descriptions."""
    if not isinstance(after, str) or not after.strip():
        return ["empty_navigation_md"]
    docs, sections, paths = _registries(bundle_dir)
    valid_keys = {"overall", *("document:" + doc["doc_id"] for doc in docs)}
    if target_key not in valid_keys:
        return ["unknown_navigation_target"]
    errors = []
    # Registry references may be repeated as cross-links, never invented. The
    # actual entries below still occur exactly once, with the same parent tree.
    known_sids = {s["section_id"] for s in sections}
    known_aliases = set(paths.get("section_aliases", known_sids))
    if (set(re.findall(r"\bsec_[a-zA-Z0-9]+\b", after)) - known_sids
            or set(re.findall(r"\bs\d{4,}\b", after)) - known_aliases):
        errors.append("changed_section_identifiers")
    links = lambda text: Counter(re.findall(r"`([^`\n]+\.md)`", text))
    if links(before) != links(after):
        errors.append("changed_navigation_asset_links")
    if target_key == "overall":
        # Document entries remain headings. Headings may
        # use a different level/style and description, but not another document ID.
        def entries(text):
            result = []
            for line in text.splitlines():
                if re.match(r"^\s*#{1,6}\s", line):
                    result.extend(doc["doc_id"] for doc in docs if doc["doc_id"] in line)
            return result
        expected = [doc["doc_id"] for doc in docs]
        original, current = entries(before), entries(after)
        if Counter(original) != Counter(expected) or Counter(current) != Counter(expected):
            errors.append("changed_document_inventory")
    else:
        did = target_key.removeprefix("document:")
        if did not in after:
            errors.append("changed_document_identifier")
        aliases = paths.get("section_aliases", {s["section_id"]: s["section_id"] for s in sections})
        canonical = {s["section_id"]: s for s in sections if s["doc_id"] == did}
        reverse = {sid: alias for alias, sid in aliases.items()}
        expected = {reverse[sid]: reverse.get(s.get("parent_id")) for sid, s in canonical.items()}
        try:
            old_tree, old_order = _section_tree(before)
            new_tree, new_order = _section_tree(after)
            if old_tree != expected or new_tree != expected:
                errors.append("changed_section_inventory_or_parent_tree")
        except ValueError as exc:
            errors.append(str(exc))
    return sorted(set(errors))


def _leak_errors(text: str, forbidden_questions, *, before="") -> list[str]:
    errors = []
    if re.search(r"\bqursor:p\d+:(?:dev|test|train):q\d+\b|\bq\d{4,}\b|\bqid\s*[:=]", text, re.I):
        errors.append("question_identifier_leak")
    if re.search(r'"(?:question|answer|canonical|gold_answer)"\s*:|^\s*(?:Q|A|Question|Answer)\s*[:：]', text, re.I | re.M):
        errors.append("serialized_qa_or_answer_lookup")
    if re.search(r"\b(?:if|when)\b[^\n]{0,80}\b(?:query|question)\b[^\n]{0,80}"
                 r"\b(?:contains|equals|matches|is exactly|starts with)\b|"
                 r"\b(?:exact[- ]match|keyword[- ]lookup|question[- ]to[- ]answer|qid[- ]lookup)\b", text, re.I):
        errors.append("query_specific_lookup_rule")
    if re.search(r"ignore (?:all |the )?(?:previous|prior|system) instructions|"
                 r"(?:system|assistant|user)\s*<\||<\|(?:im_start|im_end|system|assistant)", text, re.I):
        errors.append("instruction_override_payload")
    normalized = " ".join(text.casefold().split())
    old_normalized = " ".join(before.casefold().split())
    for question in forbidden_questions:
        if not isinstance(question, str):
            raise ValueError("forbidden_questions must contain only question strings")
        q = " ".join(question.casefold().split())
        if len(q) >= 12 and q in normalized and q not in old_normalized:
            errors.append("verbatim_dataset_question")
            break
    return errors


def validate_source_refs(bundle_dir, refs, *, target_key) -> list[str]:
    """Check exact corpus quotes, never fuzzy gold alignment or skill metadata."""
    if not isinstance(refs, list) or not refs:
        return ["missing_grounding_source_refs"]
    errors = []
    root = Path(bundle_dir).resolve()
    with sqlite3.connect((root / "corpus.sqlite").as_uri() + "?mode=ro", uri=True) as conn:
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get("quote"), str) or not ref["quote"].strip():
                errors.append("invalid_grounding_source_ref")
                continue
            kind, identifier = ("blocks", ref.get("block_id")) if ref.get("block_id") else ("chunks", ref.get("chunk_id"))
            if not isinstance(identifier, str) or not isinstance(ref.get("doc_id"), str):
                errors.append("invalid_grounding_source_ref")
                continue
            row = conn.execute(f"SELECT payload FROM {kind} WHERE record_id=?", (identifier,)).fetchone()
            source = json.loads(row[0]) if row else None
            if (not source or source.get("doc_id") != ref["doc_id"]
                    or ref["quote"] not in source.get("text", "")
                    or (ref.get("section_id") is not None and ref["section_id"] != source.get("section_id"))):
                errors.append("grounding_quote_not_in_original_source")
    return sorted(set(errors))


def format_only_grounding(grounding) -> bool:
    return (isinstance(grounding, dict) and grounding.get("valid") is True
            and grounding.get("change_types") == ["format"]
            and all(grounding.get(key) is True for key in
                    ("meaning_preserved", "hierarchy_consistent", "no_lookup_rules")))


def validate_navigation_budget(bundle_dir, snapshot, *, token_counter, max_input_tokens,
                               question_token_reserve=768) -> dict:
    """Exercise actual router envelopes for every permitted document pair/order.

    No model is called. Reserve extra question and recovery-hint tokens; online
    checks still enforce the exact real request budget, without truncation.
    """
    from qa_agent.navigation import SkillRouter
    if not callable(token_counter) or type(max_input_tokens) is not int or max_input_tokens <= 0:
        raise ValueError("A real token counter and positive navigation budget are required")
    if type(question_token_reserve) is not int or question_token_reserve < 1:
        raise ValueError("question_token_reserve must be positive")
    docs, sections, _ = _registries(bundle_dir)
    maximum = 0
    envelopes = 0
    class Probe:
        def complete(self, messages, schema, stage, **_):
            nonlocal maximum, envelopes
            raw = json.dumps({"messages": messages, "schema": schema}, ensure_ascii=False)
            count = token_counter(raw)
            if type(count) is not int or count < 0:
                raise ValueError("Token counter must return a nonnegative integer")
            # A 768-token question plus 720 tokens of missing-information text
            # in the recovery path is reserved beyond the probe's actual text.
            count += question_token_reserve + (720 if self.recovery else 0)
            maximum, envelopes = max(maximum, count), envelopes + 1
            key = "doc_ids" if stage == "select_documents" else "section_ids"
            values = list(self.selected) if key == "doc_ids" else [schema["properties"][key]["items"]["enum"][0]]
            return {"data": {key: values}}
    client = Probe()
    router = SkillRouter(docs, sections, client, bundle_dir,
                         {"max_input_tokens": max_input_tokens}, token_counter=token_counter,
                         navigation_md_overrides=snapshot)
    dids = [doc["doc_id"] for doc in docs]
    pairs = [(did,) for did in dids] + list(itertools.permutations(dids, 2))
    for pair in pairs:
        client.selected = pair
        chosen = [s["section_id"] for s in sections if s["doc_id"] in pair][:4]
        for recovery in (False, True):
            client.recovery = recovery
            feedback = ({"previous_scope": {"doc_ids": list(pair), "section_ids": chosen},
                         "seen_section_ids": [s["section_id"] for s in sections][:40],
                         "missing_information": ["missing configuration", "missing precondition", "missing procedure"]}
                        if recovery else None)
            result = router.route("Navigation input budget probe", feedback=feedback)
            if result["status"] != "ok":
                return {"valid": False, "max_tokens": maximum, "max_input_tokens": max_input_tokens,
                        "envelopes": envelopes, "reason": "navigation_envelope_rejected"}
    return {"valid": maximum <= max_input_tokens, "max_tokens": maximum,
            "max_input_tokens": max_input_tokens, "envelopes": envelopes,
            "question_token_reserve": question_token_reserve, "recovery_gap_token_reserve": 720}


def validate_candidate(before, after, *, target_key, bundle_dir, snapshot, patch, report,
                       token_counter, max_input_tokens, forbidden_questions=(), grounding_verifier=None):
    """Validate an already-applied upstream SkillOPT candidate; never reapply it."""
    from doc2skill.config import sha256
    from .cards import (build_navigation_card_snapshot, compare_card_skills,
                        parse_card_skill, validate_card_values)
    errors = []
    card_mode = isinstance(before, str) and before.startswith("# QURSOR navigation cards\n")
    baseline = build_navigation_card_snapshot(bundle_dir) if card_mode else load_navigation_snapshot(bundle_dir)
    if (not isinstance(snapshot, dict) or set(snapshot) - baseline.keys()
            or any(not isinstance(v, str) or not v.strip() for v in snapshot.values())):
        return {"valid": False, "errors": ["invalid_navigation_snapshot"], "sidecar": {}}
    current = {**baseline, **snapshot}
    if target_key not in current or before != current.get(target_key):
        errors.append("candidate_before_does_not_match_snapshot")
    edits = patch.get("edits") if isinstance(patch, dict) else None
    allowed = {"append": {"applied_append", "applied_append_before_protected_region"},
               "insert_after": {"applied_insert_after"}, "replace": {"applied_replace"}, "delete": {"applied_delete"}}
    if not isinstance(edits, list) or not 1 <= len(edits) <= 3:
        errors.append("candidate_requires_one_to_three_atomic_edits")
    elif not isinstance(report, list) or len(report) != len(edits):
        errors.append("incomplete_patch_application_report")
    else:
        for index, (edit, record) in enumerate(zip(edits, report), 1):
            if (not isinstance(edit, dict) or not isinstance(record, dict)
                    or edit.get("op") not in allowed or record.get("op") != edit.get("op")
                    or record.get("index") != index or record.get("status") not in allowed[edit["op"]]):
                errors.append("patch_partial_fallback_or_failed_application")
    if not isinstance(before, str) or not isinstance(after, str):
        errors.append("candidate_md_must_be_text")
        return {"valid": False, "errors": sorted(set(errors)), "sidecar": {}}
    if before == after:
        errors.append("no_candidate_change")
    semantic_edits = []
    if card_mode:
        try:
            comparison = compare_card_skills(before, after, expected_target=target_key)
            errors += comparison["errors"]
            semantic_edits = comparison["edits"]
            if not 1 <= len(semantic_edits) <= 3:
                errors.append("candidate_requires_one_to_three_card_field_edits")
        except ValueError as exc:
            errors.append(str(exc))
    else:
        errors += validate_structure(before, after, target_key=target_key, bundle_dir=bundle_dir)
    errors += _leak_errors(after, forbidden_questions, before=before)
    candidate = {**current, target_key: after}
    budget = None
    grounding = {"valid": False, "status": "not_run"}
    if not errors:
        for key, content in candidate.items():
            if card_mode:
                try:
                    parsed = parse_card_skill(content)
                    if parsed["target"] != key:
                        errors.append("changed_navigation_card_target")
                    errors += validate_card_values(parsed)
                    inventory = compare_card_skills(baseline[key], content, expected_target=key)
                    errors += [error for error in inventory["errors"]
                               if error != "navigation_card_edit_budget_exceeded"]
                except ValueError as exc:
                    errors.append(str(exc))
            else:
                errors += validate_structure(baseline[key], content, target_key=key, bundle_dir=bundle_dir)
    if not errors:
        budget = validate_navigation_budget(bundle_dir, candidate, token_counter=token_counter,
                                            max_input_tokens=max_input_tokens)
        if not budget["valid"]:
            errors.append("complete_navigation_input_budget_exceeded")
    if not errors:
        if not callable(grounding_verifier):
            errors.append("grounding_verifier_required")
        else:
            try:
                grounding = grounding_verifier(target_key, before, after, semantic_edits if card_mode else None)
                if not isinstance(grounding, dict) or grounding.get("valid") is not True:
                    errors.append("grounding_rejected")
                    if not isinstance(grounding, dict):
                        grounding = {"valid": False, "reason": "invalid_grounding_response"}
                    # Preserve the private audit reason/refs.  The upstream
                    # rejection path receives only a generic validation error.
                    grounding = {key: grounding[key] for key in (
                        "source_refs", "change_types", "meaning_preserved", "hierarchy_consistent",
                        "no_lookup_rules", "reason", "invalid_refs", "provenance", "verifier") if key in grounding}
                    grounding.update(valid=False, status="rejected")
                else:
                    refs = grounding.get("source_refs")
                    if not (format_only_grounding(grounding) and refs == []):
                        errors += validate_source_refs(bundle_dir, refs, target_key=target_key)
                    # Keep attributable support, not arbitrary judge payloads or reasoning.
                    grounding = {key: grounding[key] for key in (
                        "source_refs", "change_types", "meaning_preserved", "hierarchy_consistent",
                        "no_lookup_rules", "reason", "invalid_refs", "provenance", "verifier") if key in grounding}
                    grounding.update(valid=not errors, status="checked")
            except Exception as exc:
                errors.append("grounding_verifier_failed")
                grounding = {"valid": False, "status": "failed", "error_type": type(exc).__name__}
    sidecar = {"schema_version": "navigation-candidate-validation-v1", "valid": not errors,
               "target_key": target_key, "before_sha256": text_sha256(before), "after_sha256": text_sha256(after),
               "base_bundle_manifest_sha256": sha256(Path(bundle_dir) / "bundle_manifest.json"),
               "patch_sha256": text_sha256(json.dumps(patch, sort_keys=True, ensure_ascii=False)),
               "application_report_sha256": text_sha256(json.dumps(report, sort_keys=True, ensure_ascii=False)),
               "atomic_edits": len(semantic_edits) if card_mode else len(edits) if isinstance(edits, list) else 0,
               "card_field_edits": semantic_edits if card_mode else None,
               "budget": budget, "grounding": grounding, "errors": sorted(set(errors))}
    return {"valid": not errors, "errors": sidecar["errors"], "sidecar": sidecar}
