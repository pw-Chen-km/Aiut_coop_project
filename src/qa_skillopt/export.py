"""Export jointly accepted navigation snapshots without touching source assets."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from doc2skill.config import sha256, write_json
from qa_agent.bundle import local_asset, validate_serving_bundle
from .validation import (format_only_grounding, load_navigation_snapshot,
                         text_sha256, validate_source_refs, validate_structure)


def export_snapshot(bundle_dir, output_dir, navigation_md_overrides, *, sidecar=None) -> dict:
    """Publish a full new serving bundle after an externally recorded joint gate.

    ``candidate_validations`` is the chronological sequence of *accepted* edits,
    not rejected proposals. Its hashes must chain from this source bundle to the
    exported texts. The sidecar is local audit evidence, not a cryptographic
    attestation of an independent judge or an improvement claim.
    """
    source, target = Path(bundle_dir).resolve(), Path(output_dir).resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Snapshot output must not overlap the source bundle")
    if target.exists():
        raise FileExistsError("Snapshot output exists; choose a new version")
    manifest = validate_serving_bundle(source)
    original_manifest_hash = sha256(source / "bundle_manifest.json")
    card_mode = isinstance(sidecar, dict) and sidecar.get("navigation_mode") == "cards-v1"
    if card_mode:
        from .cards import build_navigation_card_snapshot, compare_card_skills
        baseline = build_navigation_card_snapshot(source)
        if sidecar.get("card_baseline_sha256") != {key: text_sha256(value) for key, value in baseline.items()}:
            raise ValueError("Export card baseline differs from the sealed preparation")
    else:
        baseline = load_navigation_snapshot(source)
    if (not isinstance(navigation_md_overrides, dict)
            or set(navigation_md_overrides) - baseline.keys()
            or any(not isinstance(value, str) or not value.strip() for value in navigation_md_overrides.values())):
        raise ValueError("Snapshot accepts only overall/document navigation MD contents")
    texts = {**baseline, **navigation_md_overrides}
    for key, text in texts.items():
        errors = (compare_card_skills(baseline[key], text, expected_target=key)["errors"]
                  if card_mode else validate_structure(baseline[key], text, target_key=key, bundle_dir=source))
        if card_mode:
            errors = [error for error in errors if error != "navigation_card_edit_budget_exceeded"]
        if errors:
            raise ValueError("Invalid snapshot structure: " + ", ".join(errors))
    if (not isinstance(sidecar, dict) or sidecar.get("joint_accepted") is not True
            or sidecar.get("base_bundle_manifest_sha256") != original_manifest_hash
            or not isinstance(sidecar.get("candidate_validations"), list)):
        raise ValueError("Export requires a joint-accepted sidecar tied to the exact source bundle")
    current = {key: text_sha256(value) for key, value in baseline.items()}
    for record in sidecar["candidate_validations"]:
        if not isinstance(record, dict):
            raise ValueError("Invalid candidate validation in export sidecar")
        key = record.get("target_key")
        grounding = record.get("grounding")
        if (record.get("valid") is not True or record.get("errors") != []
                or key not in current or record.get("before_sha256") != current[key]
                or record.get("base_bundle_manifest_sha256") != original_manifest_hash
                or type(record.get("atomic_edits")) is not int or not 1 <= record["atomic_edits"] <= 3
                or not isinstance(record.get("budget"), dict) or record["budget"].get("valid") is not True
                or not isinstance(grounding, dict) or grounding.get("valid") is not True
                or not isinstance(record.get("after_sha256"), str) or len(record["after_sha256"]) != 64):
            raise ValueError("Candidate validations must form a grounded, budget-checked single-target hash chain")
        if not (format_only_grounding(grounding) and grounding.get("source_refs") == []):
            errors = validate_source_refs(source, grounding.get("source_refs"), target_key=key)
            if errors:
                raise ValueError("Invalid exported grounding: " + ", ".join(errors))
        current[key] = record["after_sha256"]
    if current != {key: text_sha256(value) for key, value in texts.items()}:
        raise ValueError("Exported navigation texts are not the jointly accepted candidate snapshot")

    optimized = sorted(key for key in texts if texts[key] != baseline[key])
    on_disk_baseline = load_navigation_snapshot(source)
    changed = sorted(key for key in texts if texts[key] != on_disk_baseline[key])
    payload = copy.deepcopy(sidecar)
    payload.update(schema_version="qa-skillopt-snapshot-v1", private_data=True,
                   changed_targets=changed, optimized_targets=optimized,
                   card_migration_targets=sorted(set(changed) - set(optimized)) if card_mode else [],
                   status=("accepted_navigation_snapshot" if optimized else
                           "navigation_card_baseline" if card_mode else "unchanged_baseline"),
                   improvement_claimed=False,
                   navigation_md_sha256={key: text_sha256(text) for key, text in sorted(texts.items())})
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    sidecar_relative = "skillopt/acceptance-" + text_sha256(encoded)[:24] + ".json"
    paths = json.loads((source / "skill_paths.json").read_text(encoding="utf-8"))
    names = {"overall": paths["overall_doc_skill"],
             **{"document:" + did: path for did, path in paths["documents"].items()}}
    # Exclusive creation: a partial failure is not a valid bundle, since the
    # ready manifest is written last. Existing outputs are never replaced.
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    for relative, digest in manifest["artifacts"].items():
        original = local_asset(source, relative)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        if sha256(destination) != digest:
            raise ValueError("Source changed during export; incomplete output must not be served")
    for key in changed:
        (target / names[key]).write_text(texts[key], encoding="utf-8")
    write_json(target / sidecar_relative, payload)
    output_manifest = copy.deepcopy(manifest)
    assets = set(manifest["artifacts"]) | {sidecar_relative}
    output_manifest["artifacts"] = {path: sha256(target / path) for path in sorted(assets)}
    output_manifest["skillopt_snapshot"] = {
        "base_bundle_manifest_sha256": original_manifest_hash, "acceptance_sidecar": sidecar_relative,
        "changed_targets": changed, "status": payload["status"], "joint_accepted": True}
    # Frozen source DB/vector/metadata/policy hashes must be byte-identical.
    mutable = {names[key] for key in changed}
    for relative, digest in manifest["artifacts"].items():
        if relative not in mutable and output_manifest["artifacts"][relative] != digest:
            raise ValueError("Export changed an immutable corpus or runtime asset")
    if sha256(source / "bundle_manifest.json") != original_manifest_hash:
        raise ValueError("Source manifest changed during export")
    write_json(target / "bundle_manifest.json", output_manifest)
    return validate_serving_bundle(target)
