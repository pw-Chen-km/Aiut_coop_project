"""Immutable-data preparation and explicitly reviewed, question-only split groups.

Similarity retrieves review candidates; it NEVER adjudicates near duplicates.
Gold evidence can overlap between phases under the chosen prototype protocol.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata


PHASES = ("optimization", "validation", "test")
TYPES = ("simple", "complex", "multihop")
SMOKE_QIDS = ("qursor:p3:dev:q000001", "qursor:p3:dev:q000101", "qursor:p3:dev:q000151")
QUARANTINE_QID = "qursor:p3:dev:q000101"
DEFAULT_QUOTAS = {"optimization": {"simple": 18, "complex": 12, "multihop": 10},
                  "validation": {"simple": 18, "complex": 12, "multihop": 10},
                  "test": {"simple": 54, "complex": 36, "multihop": 30}}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_question(text):
    """Keep numbers, negation and terminology; never mask scenario conditions."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _records(qa):
    rows = sorted(deepcopy(list(qa)), key=lambda row: str(row.get("qid", "")))
    ids = [row.get("qid") for row in rows]
    if any(not isinstance(qid, str) or not qid.strip() for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("QA qids must be unique nonempty strings")
    if any(not isinstance(row.get("question"), str) or not row["question"].strip() for row in rows):
        raise ValueError("QA questions must be nonempty strings")
    return rows


def _pair_id(left, right):
    return "dup_" + digest(sorted((left, right)))[:20]


def duplicate_candidates(qa_records, *, threshold=0.35):
    """Return stable candidates, including broad family hints, with NO labels."""
    if not 0 < threshold <= 1:
        raise ValueError("Candidate threshold must be in (0, 1]")
    rows = _records(qa_records)
    normalized = {q["qid"]: normalize_question(q["question"]) for q in rows}
    grams = {qid: {text[i:i + 5] for i in range(max(1, len(text) - 4))}
             for qid, text in normalized.items()}
    output = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            a, b = left["qid"], right["qid"]
            score = 2 * len(grams[a] & grams[b]) / (len(grams[a]) + len(grams[b]))
            reasons = []
            if normalized[a] == normalized[b]:
                reasons.append("normalized_exact_question")
            if left.get("question_family_id") and left.get("question_family_id") == right.get("question_family_id"):
                reasons.append("family_hint_only")
            if score >= threshold:
                reasons.append("lexical_similarity_candidate_only")
            if reasons:
                output.append({"candidate_id": _pair_id(a, b), "left_qid": a, "right_qid": b,
                               "left_question_sha256": digest(left["question"]),
                               "right_question_sha256": digest(right["question"]),
                               "reasons": reasons, "char5gram_dice": score,
                               "review_status": "pending", "duplicate": None})
    return output


def _features(row):
    features = ["type:" + str(row.get("question_type")),
                "visual:" + str(bool(row.get("visual_diagnostic"))).lower(),
                "scope:" + str(row.get("scope", "unknown")),
                "difficulty:" + str(row.get("difficulty", "unknown")),
                "persona:" + str(row.get("scenario", {}).get("persona", "unknown"))]
    docs = {str(e["doc_id"]) for e in row.get("evidence", []) if e.get("doc_id")}
    features += ["document:" + doc for doc in sorted(docs)]
    features.append("document_count:" + str(len(docs)))
    return features


def _assign(groups, by_id, quotas, seed, pinned, *, exact_types=True):
    """Exact integer quotas; seeded soft-balance ordering, never split a group."""
    buckets = TYPES if exact_types else ("total",)
    capacity = {phase: Counter(quotas[phase] if exact_types else {"total": sum(quotas[phase].values())}) for phase in PHASES}
    totals = Counter(f for row in by_id.values() for f in _features(row))
    used_features = {phase: Counter() for phase in PHASES}
    allocation = {}
    target_sizes = {p: sum(quotas[p].values()) for p in PHASES}

    def counts(group):
        return Counter(by_id[qid]["question_type"] if exact_types else "total" for qid in group)

    def fits(group, phase):
        return all(n <= capacity[phase][t] for t, n in counts(group).items())

    def put(group, phase, sign):
        for qid in group:
            capacity[phase][by_id[qid]["question_type"] if exact_types else "total"] -= sign
            used_features[phase].update({f: sign for f in _features(by_id[qid])})
            if sign == 1:
                allocation[qid] = phase
            else:
                allocation.pop(qid)

    rest = []
    for group in groups:
        if set(group) & pinned:
            if not fits(group, "optimization"):
                raise ValueError("Reviewed duplicate group containing smoke queries exceeds optimization quotas")
            put(group, "optimization", 1)
        else:
            rest.append(group)
    rest.sort(key=lambda g: (-len(g), digest([seed, g])))
    # Singletons can always fill each remaining type capacity. They are handled
    # separately, avoiding exponential search over the majority of the dataset.
    multiple = [g for g in rest if len(g) > 1]
    singles = [g for g in rest if len(g) == 1]
    single_types = Counter(by_id[g[0]]["question_type"] if exact_types else "total" for g in singles)
    failed = set()

    def cost(group, phase):
        delta = Counter(f for qid in group for f in _features(by_id[qid]))
        ratio = target_sizes[phase] / len(by_id)
        soft = sum(((used_features[phase][f] + n - totals[f] * ratio) ** 2
                    - (used_features[phase][f] - totals[f] * ratio) ** 2)
                   / max(1, totals[f]) for f, n in delta.items())
        return soft, digest([seed, phase, group])

    def search(index):
        state = (index, tuple(capacity[p][t] for p in PHASES for t in buckets))
        if state in failed:
            return False
        if index == len(multiple):
            return all(sum(capacity[p][t] for p in PHASES) == single_types[t] for t in buckets)
        group = multiple[index]
        for phase in sorted(PHASES, key=lambda p: cost(group, p)):
            if fits(group, phase):
                put(group, phase, 1)
                if search(index + 1):
                    return True
                put(group, phase, -1)
        failed.add(state)
        return False

    if not search(0):
        raise ValueError("Reviewed duplicate groups make the requested exact phase/type quotas infeasible")
    for group in singles:
        available = [p for p in PHASES if fits(group, p)]
        if not available:
            raise AssertionError("Singleton quota allocation failed")
        put(group, min(available, key=lambda p: cost(group, p)), 1)
    if any(capacity[p][t] for p in PHASES for t in buckets):
        raise AssertionError("Split quotas were not filled exactly")
    return dict(sorted(allocation.items()))


def build_split_manifest(qa_records, duplicate_decisions=(), *, seed=42,
                         source_sha256=None, quotas=None):
    """Fail closed on pending duplicate review; originals and IDs are unchanged.

    Each reviewed decision requires candidate_id, left/right_qid, duplicate
    (boolean), reviewer, review_kind, reason, and both question SHA256 values.
    Agent judgments are explicitly distinct from human/SME review.
    """
    rows = _records(qa_records)
    by_id = {q["qid"]: q for q in rows}
    quotas = deepcopy(DEFAULT_QUOTAS if quotas is None else quotas)
    if set(quotas) != set(PHASES) or any(set(quotas[p]) != set(TYPES) for p in PHASES):
        raise ValueError("Quotas require optimization/validation/test and all three question types")
    if any(type(n) is not int or n < 0 for p in PHASES for n in quotas[p].values()):
        raise ValueError("Quotas must be nonnegative integers")
    if Counter(q.get("question_type") for q in rows) != Counter({t: sum(quotas[p][t] for p in PHASES) for t in TYPES}):
        raise ValueError("Dataset question types/counts do not match the requested quotas")
    candidates = duplicate_candidates(rows)
    decisions = duplicate_decisions.get("decisions", []) if isinstance(duplicate_decisions, dict) else duplicate_decisions
    reviewed = {}
    for decision in decisions:
        a, b = decision.get("left_qid"), decision.get("right_qid")
        if a not in by_id or b not in by_id or a == b:
            raise ValueError("Duplicate decision refers to invalid question IDs")
        identifier = _pair_id(a, b)
        if decision.get("candidate_id") != identifier or identifier in reviewed:
            raise ValueError("Duplicate decision ID is invalid or repeated")
        if (decision.get("review_status") != "reviewed" or type(decision.get("duplicate")) is not bool
                or not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip()
                or decision.get("review_kind") not in {"agent_review", "human_review", "sme_review"}
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError("Every duplicate decision must carry an explicit attributed reviewed judgment")
        if any(decision.get(side + "_question_sha256") != digest(by_id[decision[side + "_qid"]]["question"])
               for side in ("left", "right")):
            raise ValueError("Duplicate review is stale for the current question text")
        reviewed[identifier] = deepcopy(decision)
    pending = [c["candidate_id"] for c in candidates if c["candidate_id"] not in reviewed]
    quarantine = ({QUARANTINE_QID: {"status": "pending_sme_clarification", "exclude_from_objective": True,
                                  "reason": "False-condition successor configuration is unspecified; no unconditional advancement gold is safe."}}
                  if QUARANTINE_QID in by_id else {})
    manifest = {"schema_version": "qa-skillopt-split-v1", "seed": seed,
                "status": "pending_duplicate_review" if pending else "ready",
                "source_sha256": source_sha256, "records_sha256": digest(rows),
                "quotas": quotas, "assignments": {}, "groups": [],
                "duplicate_candidates": candidates, "duplicate_decisions": list(reviewed.values()),
                "unreviewed_candidate_ids": pending, "quarantine": quarantine,
                "prior_exposure_qids": [q for q in SMOKE_QIDS if q in by_id],
                "policy": {"shared_evidence_allowed": True, "group_by_evidence": False,
                           "group_by_family_without_review": False, "original_ids_unchanged": True,
                           "soft_balance": ["visual", "scope", "document", "document_count", "persona", "difficulty"],
                           "duplicate_rubric": "Same material conditions and substantially interchangeable requested answer/action; shared source or broad topic alone is insufficient."}}
    if pending:
        return manifest
    parent = {qid: qid for qid in by_id}

    def find(qid):
        while parent[qid] != qid:
            qid = parent[qid]
        return qid

    for decision in reviewed.values():
        if decision["duplicate"]:
            parent[find(decision["left_qid"])] = find(decision["right_qid"])
    groups = defaultdict(list)
    for qid in by_id:
        groups[find(qid)].append(qid)
    grouped = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    try:
        manifest["assignments"] = _assign(grouped, by_id, quotas, seed, set(SMOKE_QIDS) & set(by_id))
        manifest["type_balance"] = "exact_targets_met"
    except ValueError:
        # Total phase sizes and confirmed duplicate groups are hard. Type
        # targets are soft and must never force a legitimate duplicate apart.
        manifest["assignments"] = _assign(grouped, by_id, quotas, seed, set(SMOKE_QIDS) & set(by_id), exact_types=False)
        manifest["type_balance"] = "soft_targets_deviated_to_preserve_duplicate_groups"
    manifest["groups"] = [{"group_id": "qgroup_" + digest(g)[:16], "qids": g,
                           "phase": manifest["assignments"][g[0]]} for g in grouped]
    manifest["balance"] = {p: {"question_count": sum(v == p for v in manifest["assignments"].values()),
                               "question_types": dict(Counter(by_id[qid]["question_type"] for qid, phase in manifest["assignments"].items() if phase == p)),
                               "type_target_deviation": {t: sum(by_id[qid]["question_type"] == t for qid, phase in manifest["assignments"].items() if phase == p) - quotas[p][t] for t in TYPES},
                               "features": dict(Counter(f for qid, phase in manifest["assignments"].items()
                                                        if phase == p for f in _features(by_id[qid])))} for p in PHASES}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def phase_payload(qa_records, manifest, phase, *, purpose="training", include_quarantined=False):
    """Gold stays with the trusted evaluator/optimizer, never the target runner.

    Test gold is accessible only under the explicit final_evaluation purpose.
    Assigned quarantines are retained in the manifest but excluded by default;
    callers must report both assigned and eligible denominators.
    """
    if manifest.get("status") != "ready":
        raise ValueError("Split manifest is not ready: duplicate review is unresolved")
    if phase not in PHASES or purpose not in {"training", "validation", "final_evaluation"}:
        raise ValueError("Unknown phase or purpose")
    if phase == "test" and purpose != "final_evaluation":
        raise ValueError("Test payload is sealed against training and validation access")
    rows = _records(qa_records)
    if digest(rows) != manifest.get("records_sha256"):
        raise ValueError("QA records disagree with the immutable split input fingerprint")
    if set(manifest.get("assignments", {})) != {q["qid"] for q in rows}:
        raise ValueError("Manifest must assign every original qid exactly once")
    return [row for row in rows if manifest["assignments"][row["qid"]] == phase
            and (include_quarantined or row["qid"] not in manifest.get("quarantine", {}))]


def write_preparation(qa_path, bundle_dir, out_dir, decisions_path=None):
    """Write one new private preparation snapshot; never overwrite inputs/outputs."""
    qa_path, bundle_dir, out = Path(qa_path).resolve(), Path(bundle_dir).resolve(), Path(out_dir).resolve()
    if out.exists() or out == qa_path.parent or out.is_relative_to(qa_path.parent) or out.is_relative_to(bundle_dir):
        raise ValueError("Preparation output must be a new directory separate from dataset and bundle")
    rows = read_jsonl(qa_path)
    decisions = json.loads(Path(decisions_path).read_text()) if decisions_path else []
    manifest = build_split_manifest(rows, decisions, source_sha256=hashlib.sha256(qa_path.read_bytes()).hexdigest())
    manifest["source_path"] = str(qa_path)
    manifest["bundle_dir"] = str(bundle_dir)
    identity_path = bundle_dir / "bundle_manifest.json"
    if not identity_path.exists():
        identity_path = bundle_dir / "manifest.json"
    bundle_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    preparation_id = "prep_" + digest([manifest["records_sha256"], manifest["duplicate_decisions"], bundle_sha, manifest["seed"]])[:24]
    manifest["preparation_id"] = preparation_id
    manifest["bundle_manifest_sha256"] = bundle_sha
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest(manifest)
    out.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    for name, value in (("split_manifest.json", manifest), ("duplicate_candidates.json", manifest["duplicate_candidates"]),
                        ("quarantine.json", manifest["quarantine"])):
        write_json(name, value)
    if manifest["status"] == "ready":
        from .oracle import build_oracle, load_corpus
        corpus = load_corpus(bundle_dir)
        legacy = read_jsonl(qa_path.parent / "documents.jsonl")
        atoms_path = qa_path.parent / "source_atoms.jsonl"
        atoms = read_jsonl(atoms_path) if atoms_path.exists() else []
        for phase in PHASES:
            payload = phase_payload(rows, manifest, phase, purpose="final_evaluation" if phase == "test" else "training")
            output = out / (phase + ".jsonl")
            with output.open("x", encoding="utf-8") as stream:
                for row in payload:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            qids = [q["qid"] for q in payload]
            phase_manifest = {"schema_version": "qa-skillopt-phase-v1", "preparation_id": preparation_id,
                "phase": phase, "status": "ready", "assigned_count": sum(p == phase for p in manifest["assignments"].values()),
                "eligible_count": len(payload), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "qids": qids,
                "records_sha256": digest(payload), "bundle_manifest_sha256": bundle_sha,
                "quarantined_qids": [qid for qid in manifest["quarantine"] if manifest["assignments"][qid] == phase],
                "source_dataset_sha256": manifest["source_sha256"], "split_manifest_sha256": manifest["manifest_sha256"]}
            oracle = build_oracle(payload, corpus, legacy_documents=legacy, source_atoms=atoms)
            oracle["preparation_id"] = preparation_id
            oracle["phase"] = phase
            write_json("oracle_" + phase + ".json", oracle)
            phase_manifest["oracle_file_sha256"] = hashlib.sha256((out / ("oracle_" + phase + ".json")).read_bytes()).hexdigest()
            write_json(phase + "_manifest.json", phase_manifest)
    return manifest


def write_navigation_preparation(preparation_dir, bundle_dir, out_dir, *, min_target_samples=10):
    """Upgrade a sealed split with source-free navigation-oracle sidecars.

    Phase payloads and the held-out files are copied byte-for-byte.  This is a
    dataset-preparation operation, not part of training; the training loader
    continues to open only optimization and validation files.
    """
    from .cards import build_navigation_card_snapshot
    from .navigation_feedback import build_navigation_oracle

    source, bundle, out = Path(preparation_dir).resolve(), Path(bundle_dir).resolve(), Path(out_dir).resolve()
    if not source.is_dir() or out.exists() or out == source or out.is_relative_to(source):
        raise ValueError("Navigation preparation requires an existing source and a separate new output")
    if type(min_target_samples) is not int or min_target_samples < 1:
        raise ValueError("min_target_samples must be positive")
    out.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, out / path.name)

    phase_views, counts = {}, {}
    for phase in ("optimization", "validation"):
        source_oracle = json.loads((source / f"oracle_{phase}.json").read_text(encoding="utf-8"))
        view = build_navigation_oracle(source_oracle["queries"])
        view.update(phase=phase, preparation_id=source_oracle.get("preparation_id"),
                    source_oracle_sha256=hashlib.sha256((source / f"oracle_{phase}.json").read_bytes()).hexdigest())
        target = out / f"navigation_oracle_{phase}.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(view, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        phase_views[phase] = view
        current = Counter({"overall": sum(q["document_status"] == "known" for q in view["queries"].values())})
        for query in view["queries"].values():
            for did, status in query.get("section_status", {}).items():
                if status == "known":
                    current["document:" + did] += 1
        counts[phase] = dict(current)

    all_targets = sorted(set(counts["optimization"]) | set(counts["validation"]),
                         key=lambda value: (value != "overall", value))
    eligible = [target for target in all_targets
                if counts["optimization"].get(target, 0) >= min_target_samples
                and counts["validation"].get(target, 0) >= min_target_samples]
    snapshot = build_navigation_card_snapshot(bundle)
    base_manifest = bundle / "bundle_manifest.json"
    manifest = {
        "schema_version": "qursor-navigation-preparation-v1",
        "source_preparation_dir": str(source),
        "preparation_id": phase_views["optimization"].get("preparation_id"),
        "bundle_manifest_sha256": hashlib.sha256(base_manifest.read_bytes()).hexdigest(),
        "source_phase_sha256": {phase: hashlib.sha256((source / f"{phase}.jsonl").read_bytes()).hexdigest()
                                for phase in ("optimization", "validation", "test")},
        "navigation_oracle_sha256": {phase: hashlib.sha256((out / f"navigation_oracle_{phase}.json").read_bytes()).hexdigest()
                                     for phase in ("optimization", "validation")},
        "card_snapshot_sha256": {key: hashlib.sha256(text.encode()).hexdigest() for key, text in snapshot.items()},
        "min_target_samples": min_target_samples,
        "target_label_counts": counts,
        "eligible_targets": eligible,
        "skipped_targets": [{"target": target, "reason": "insufficient_route_labels",
                             "optimization": counts["optimization"].get(target, 0),
                             "validation": counts["validation"].get(target, 0)}
                            for target in all_targets if target not in eligible],
        "heldout_test_oracle_built": False,
    }
    with (out / "navigation_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest
