"""Gold-derived routing labels and leakage-free SkillOPT feedback."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import itertools

from .trajectory import diagnose_trajectory


MAX_ROUTE_COMBINATIONS = 4096


def _evidence_by_key(query: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for evidence in query.get("evidence", []):
        result[evidence.get("equivalence_group") or evidence["evidence_id"]].append(evidence)
    return result


def _minimal_sets(values) -> list[list[str]]:
    unique = {frozenset(value) for value in values if value}
    minimal = [value for value in unique if not any(other < value for other in unique)]
    return [sorted(value) for value in sorted(minimal, key=lambda value: (len(value), sorted(value)))]


def _product(options):
    count = 1
    for values in options:
        count *= len(values)
        if count > MAX_ROUTE_COMBINATIONS:
            raise ValueError("Gold routing alternatives exceed the bounded navigation-oracle budget")
    return itertools.product(*options)


def build_navigation_query(query: dict) -> dict:
    """Project a source oracle into document/section IDs without source text."""
    by_key = _evidence_by_key(query)
    document_scopes, section_scopes = [], []
    for route in query.get("routes", []):
        required = route.get("required_keys", [])
        if not required:
            continue
        doc_options, section_options, docs_known, sections_known = [], [], True, True
        for key in required:
            records = by_key.get(key, [])
            docs, targets = set(), set()
            for evidence in records:
                gold_doc = evidence.get("gold", {}).get("doc_id")
                if gold_doc:
                    docs.add(gold_doc)
                for target in evidence.get("targets", []):
                    if target.get("doc_id"):
                        docs.add(target["doc_id"])
                    for sid in target.get("section_ids", []):
                        if target.get("doc_id") and sid:
                            targets.add((target["doc_id"], sid))
            if not docs:
                docs_known = False
            else:
                doc_options.append(sorted(docs))
            if not targets:
                sections_known = False
            else:
                section_options.append(sorted(targets))
        if docs_known:
            for choice in _product(doc_options):
                document_scopes.append(set(choice))
        if sections_known:
            for choice in _product(section_options):
                grouped: dict[str, set[str]] = defaultdict(set)
                for did, sid in choice:
                    grouped[did].add(sid)
                section_scopes.append({did: sorted(ids) for did, ids in grouped.items()})

    document_sets = _minimal_sets(document_scopes)
    forced_documents = set.intersection(*(set(value) for value in document_sets)) if document_sets else set()
    section_sets_by_document = {}
    for did in sorted(forced_documents):
        scopes = [scope[did] for scope in section_scopes if did in scope]
        if scopes and len(scopes) == len(section_scopes):
            section_sets_by_document[did] = _minimal_sets(scopes)
    return {
        "schema_version": "qursor-navigation-oracle-query-v1",
        "qid": query.get("qid"),
        "document_status": "known" if document_sets else "unknown",
        "required_document_id_sets": document_sets,
        "forced_document_ids": sorted(forced_documents),
        "section_status": {did: "known" for did in section_sets_by_document},
        "required_section_id_sets_by_document": section_sets_by_document,
        "evidence_status": "known" if query.get("complete_route_available") else "unknown",
    }


def build_navigation_oracle(queries: dict[str, dict]) -> dict:
    projected = {qid: build_navigation_query(query) for qid, query in sorted(queries.items())}
    return {
        "schema_version": "qursor-navigation-oracle-v1",
        "optimizer_only": True,
        "contains_source_text": False,
        "queries": projected,
        "summary": {
            "query_count": len(projected),
            "document_known": sum(q["document_status"] == "known" for q in projected.values()),
            "evidence_known": sum(q["evidence_status"] == "known" for q in projected.values()),
        },
    }


def _ancestors(section_id: str, registry: dict[str, dict]) -> set[str]:
    result, current = set(), section_id
    while current:
        if current in result or current not in registry:
            raise ValueError("Invalid section hierarchy in routing metric")
        result.add(current)
        current = registry[current].get("parent_id")
    return result


def _route_score(selected: set[str], required_sets: list[list[str]], *, registry=None) -> dict:
    scored = []
    for required_list in required_sets:
        required = set(required_list)
        if registry is None:
            covered = required & selected
            relevant_selected = selected & required
        else:
            covered = {sid for sid in required if _ancestors(sid, registry) & selected}
            relevant_selected = {sid for sid in selected
                                 if any(sid in _ancestors(required_sid, registry) for required_sid in required)}
        recall = len(covered) / len(required) if required else 0.0
        precision = len(relevant_selected) / len(selected) if selected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scored.append({"required": sorted(required), "covered": sorted(covered),
                       "missing": sorted(required - covered), "extra": sorted(selected - relevant_selected),
                       "complete": bool(required) and required <= covered,
                       "precision": precision, "recall": recall, "f1": f1})
    if not scored:
        return {"required": [], "covered": [], "missing": [], "extra": sorted(selected),
                "complete": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return max(scored, key=lambda row: (row["complete"], row["f1"], row["recall"], row["precision"]))


def navigation_feedback(trace: dict, source_oracle_query: dict, navigation_oracle_query: dict,
                        sections, active_target: str, *, section_aliases=None) -> dict:
    """Score only the stage controlled by ``active_target``."""
    registry = sections if isinstance(sections, dict) else {s["section_id"]: s for s in sections}
    rounds = trace.get("rounds") or [trace]
    selected_docs_by_round = [list(dict.fromkeys(r.get("scope", {}).get("doc_ids", []))) for r in rounds]
    selected_sections_by_round = [list(dict.fromkeys(r.get("scope", {}).get("section_ids", []))) for r in rounds]
    cumulative_docs: set[str] = set()
    cumulative_sections: set[str] = set()
    first = None

    if active_target == "overall":
        required_sets = navigation_oracle_query.get("required_document_id_sets", [])
        for index, values in enumerate(selected_docs_by_round):
            cumulative_docs.update(values)
            score = _route_score(cumulative_docs, required_sets)
            if index == 0:
                first = deepcopy(score)
        final = _route_score(cumulative_docs, required_sets)
        scorable = navigation_oracle_query.get("document_status") == "known"
        view = {
            "schema_version": "qursor-navigation-feedback-overall-v1",
            "user_utterance": trace.get("question"),
            "selected_document_ids_by_round": selected_docs_by_round,
            "required_document_id_sets": deepcopy(required_sets),
            "missing_document_ids": final["missing"],
            "extra_document_ids": final["extra"],
            "document_route_complete": final["complete"],
        }
    else:
        did = active_target.removeprefix("document:")
        required_sets = navigation_oracle_query.get("required_section_id_sets_by_document", {}).get(did, [])
        aliases = section_aliases or {sid: sid for sid in registry}
        if (not isinstance(aliases, dict) or set(aliases.values()) != set(registry)
                or len(set(aliases.values())) != len(aliases)):
            raise ValueError("Section routing aliases must be a complete bijection")
        alias_for = {canonical: alias for alias, canonical in aliases.items()}
        active_sections_by_round = [
            [sid for sid in values if registry.get(sid, {}).get("doc_id") == did]
            for values in selected_sections_by_round
        ]
        for index, (docs, values) in enumerate(zip(selected_docs_by_round, active_sections_by_round)):
            cumulative_docs.update(docs)
            if did in cumulative_docs:
                cumulative_sections.update(values)
            score = _route_score(cumulative_sections, required_sets, registry=registry)
            if index == 0:
                first = deepcopy(score)
        final = _route_score(cumulative_sections, required_sets, registry=registry)
        scorable = bool(required_sets) and did in cumulative_docs
        view = {
            "schema_version": "qursor-navigation-feedback-document-v1",
            "user_utterance": trace.get("question"),
            "active_document_id": did,
            "selected_section_ids_by_round": [[alias_for[sid] for sid in values]
                                                for values in active_sections_by_round],
            "required_section_id_sets": [[alias_for[sid] for sid in values] for values in required_sets],
            "missing_section_ids": [alias_for[sid] for sid in final["missing"]],
            "extra_section_ids": [alias_for[sid] for sid in final["extra"]],
            "section_route_complete": final["complete"],
        }

    evidence = {"complete": False, "recall": None, "status": "unknown"}
    if source_oracle_query.get("complete_route_available"):
        diagnostic = diagnose_trajectory(trace, source_oracle_query, registry, judgment=None)
        usable = [row for row in diagnostic.get("rounds", []) if not row.get("generation_skipped")]
        if usable:
            evidence = deepcopy(usable[-1].get("pooled_coverage", evidence))
    route_f1 = final["f1"] if scorable else 1.0
    evidence_recall = evidence.get("recall")
    # Unknown evidence is neutral rather than a fabricated failure.
    soft = (route_f1 + (route_f1 if evidence_recall is None else float(evidence_recall))) / 2
    return {
        "scorable": scorable,
        "hard": int(final["complete"]) if scorable else 1,
        "soft": soft if scorable else 1.0,
        "route": final,
        "first_round_route": first or final,
        "evidence": evidence,
        "optimizer_view": view,
        "excluded_reason": None if scorable else (
            "required_document_not_selected_by_fixed_overall" if active_target != "overall" and required_sets
            else "unknown_target_routing_oracle"),
    }


def aggregate_navigation(results: list[dict]) -> dict:
    scorable = [row for row in results if row.get("navigation_scorable")]
    evidence = [row for row in scorable if row.get("evidence_recall") is not None]
    if not scorable:
        return {"scorable_count": 0, "evidence_count": 0, "route_complete": None,
                "route_precision": None, "route_recall": None, "route_f1": None,
                "first_round_route_complete": None, "evidence_recall": None,
                "evidence_complete": None, "primary_score": None}
    mean = lambda key: sum(float(row[key]) for row in scorable) / len(scorable)
    result = {
        "scorable_count": len(scorable),
        "evidence_count": len(evidence),
        "route_complete": mean("route_complete"),
        "route_precision": mean("route_precision"),
        "route_recall": mean("route_recall"),
        "route_f1": mean("route_f1"),
        "first_round_route_complete": mean("first_round_route_complete"),
        "evidence_recall": (sum(float(row["evidence_recall"]) for row in evidence) / len(evidence)) if evidence else None,
        "evidence_complete": (sum(int(row["evidence_complete"]) for row in evidence) / len(evidence)) if evidence else None,
    }
    # SkillOPT requires one finite soft score per trajectory.  When evidence is
    # unknown, reuse that row's routing F1 as a neutral value: it neither creates
    # a retrieval failure nor lets an unknown row change the denominator.  This
    # makes the external primary score exactly the mean upstream mixed score.
    evidence_components = [float(row["route_f1"] if row.get("evidence_recall") is None
                                 else row["evidence_recall"]) for row in scorable]
    result["gate_evidence_component"] = sum(evidence_components) / len(evidence_components)
    result["primary_score"] = (.50 * result["route_complete"] + .25 * result["route_f1"]
                               + .25 * result["gate_evidence_component"])
    result["upstream_mixed_score"] = sum(.5 * row["hard"] + .5 * float(row["soft"])
                                         for row in scorable) / len(scorable)
    return result
