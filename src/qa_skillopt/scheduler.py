"""Alternate MD targets; each real upstream run lives in a fresh process."""
from __future__ import annotations

import functools
import difflib
import json
from pathlib import Path
import subprocess
import sys

from doc2skill.config import sha256
from .adapter import load_training_data, make_adapter
from .io import digest_text, new_directory, read_json, write_new


def make_transport(config, *, confirmed=False):
    from .transport import OllamaTransport
    off = config["offline"]
    return OllamaTransport(base_url=off["base_url"], model=off["model"], model_digest=off["model_digest"],
                           timeout=off["timeout_seconds"], usage_window_confirmed=confirmed)


def target_worker(job_path):
    """Worker knows only phase-local train/validation data, never a test payload."""
    from .config import load_config
    from .adapter import run_qa_batch
    from .judge import AnswerJudge, GroundingVerifier, JudgmentUnavailable
    from .navigation_feedback import aggregate_navigation
    from .oracle import load_corpus
    from .transport import TransportError
    from .upstream import load_upstream, run_training
    from .validation import validate_candidate
    from qa_agent.factory import create_session, load_online_config
    from doc2skill.runtime import validate_runtime_identity

    job = read_json(job_path)
    if job.get("usage_window_confirmed") is not True:
        raise ValueError("Generation requires a confirmed AMD usage window")
    config = load_config(job["config_path"])
    if sha256(Path(job["config_path"])) != job.get("config_sha256") or sha256(Path(config["online_config"])) != job.get("online_config_sha256"):
        raise ValueError("Experiment config changed after scheduling")
    online = load_online_config(config["online_config"])
    if online.get("loop", {}).get("max_rounds", 2) != 2:
        raise ValueError("SkillOPT freezes the two-round QA flow")
    all_train, all_valid, phases = load_training_data(config["preparation_dir"])
    prep = Path(config["preparation_dir"])
    source_oracles, navigation_oracles = {}, {}
    nav_manifest = read_json(prep / "navigation_manifest.json")
    for phase, rows in (("optimization", all_train), ("validation", all_valid)):
        oracle_path = prep / f"oracle_{phase}.json"
        nav_path = prep / f"navigation_oracle_{phase}.json"
        source_oracles[phase] = read_json(oracle_path)
        navigation_oracles[phase] = read_json(nav_path)
        if sha256(oracle_path) != phases[phase].get("oracle_file_sha256"):
            raise ValueError(phase + " source oracle fingerprint mismatch")
        if sha256(nav_path) != nav_manifest.get("navigation_oracle_sha256", {}).get(phase):
            raise ValueError(phase + " navigation oracle fingerprint mismatch")
        expected = {q["qid"] for q in rows}
        if set(source_oracles[phase]["queries"]) != expected or set(navigation_oracles[phase]["queries"]) != expected:
            raise ValueError(phase + " oracle must contain exactly the eligible phase questions")
    corpus = load_corpus(online["bundle_dir"])
    section_aliases = read_json(Path(online["bundle_dir"]) / "skill_paths.json").get("section_aliases", {})
    from .data import digest
    corpus_digest = digest({kind: corpus.get(kind, []) for kind in ("documents", "blocks", "sections", "chunks")})
    if any(oracle.get("corpus_sha256") != corpus_digest for oracle in source_oracles.values()):
        raise ValueError("Oracle refers to a different corpus snapshot")
    base_hash = sha256(Path(online["bundle_dir"]) / "bundle_manifest.json")
    if any(p.get("bundle_manifest_sha256") != base_hash for p in phases.values()):
        raise ValueError("Prepared data refers to a different serving bundle")
    target, snapshot = job["active_target"], job["snapshot"]
    if target not in nav_manifest.get("eligible_targets", []):
        raise ValueError("Worker was scheduled for an ineligible navigation target")
    if target == "overall":
        eligible = lambda q, phase: navigation_oracles[phase]["queries"][q["qid"]].get("document_status") == "known"
    else:
        did = target.removeprefix("document:")
        eligible = lambda q, phase: bool(navigation_oracles[phase]["queries"][q["qid"]]
                                         .get("required_section_id_sets_by_document", {}).get(did))
    train = [q for q in all_train if eligible(q, "optimization")]
    valid = [q for q in all_valid if eligible(q, "validation")]
    minimum = config["optimization"].get("min_target_samples", 10)
    if len(train) < minimum or len(valid) < minimum:
        raise ValueError("Target no longer meets the sealed routing-label minimum")
    counter = validate_runtime_identity(online["runtime"])
    transport = make_transport(config, confirmed=True)
    preflight = transport.preflight()
    destination = Path(job["output_dir"])
    destination.mkdir(parents=True, exist_ok=False)
    transport.audit_dir = destination / "offline_calls"
    write_new(destination / "preflight.json", preflight)
    initial = destination / "initial_skill.md"
    with initial.open("x", encoding="utf-8") as stream:
        stream.write(snapshot[target])
    load_upstream(config["upstream_source"])
    session_factory = functools.partial(create_session, online)
    adapter = make_adapter(train_rows=train, validation_rows=valid, active_target=target,
                           snapshot=snapshot, session_factory=session_factory,
                           oracle={phase: value["queries"] for phase, value in source_oracles.items()},
                           navigation_oracles={phase: value["queries"] for phase, value in navigation_oracles.items()},
                           sections=corpus["sections"], section_aliases=section_aliases)
    grounder = GroundingVerifier(transport, corpus, snapshot)
    checks = []

    def check(before, after, patch, report):
        current_snapshot = {**snapshot, target: before}
        if before != snapshot[target] and not any(c["valid"] and c["sidecar"]["after_sha256"] == digest_text(before) for c in checks):
            raise RuntimeError("Upstream proposed a candidate from an unchecked skill version")
        grounder.snapshot = current_snapshot
        result = validate_candidate(before, after, target_key=target, bundle_dir=online["bundle_dir"],
                                    snapshot=current_snapshot, patch=patch, report=report, token_counter=counter,
                                    max_input_tokens=online["runtime"].get("max_input_tokens", 7000),
                                    forbidden_questions=[q["question"] for q in train], grounding_verifier=grounder)
        checks.append(result)
        write_new(destination / f"candidate_check_{len(checks):03d}.json", result)
        with (destination / f"candidate_diff_{len(checks):03d}.patch").open("x", encoding="utf-8") as stream:
            stream.writelines(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                                  fromfile="before/" + target, tofile="candidate/" + target))
        if "grounding_verifier_failed" in result["errors"]:
            raise JudgmentUnavailable("Grounding service/schema unavailable; step stopped")
        return result

    result = run_training(source_root=config["upstream_source"], adapter=adapter, initial_skill=initial,
                          output_dir=destination / "upstream", train_size=len(train), candidate_validator=check,
                          batch_size=config["optimization"].get("batch_size"),
                          config_overrides={"minibatch_size": config["optimization"].get("reflection_minibatch", 4)},
                          edit_budget=3, transport=transport, usage_window_confirmed=True)
    result["upstream_navigation_accepted"] = result["accepted"]
    selected_hash = digest_text(result["selected_skill"])
    accepted_checks = []
    current_hash = digest_text(snapshot[target])
    for event in result["history"]:
        if event.get("action") not in {"accept", "accept_new_best"}:
            continue
        candidate_file = destination / "upstream/steps" / f"step_{event['step']:04d}" / "candidate_skill.md"
        candidate_hash = sha256(candidate_file)
        matches = [c["sidecar"] for c in checks if c["valid"] and c["sidecar"]["before_sha256"] == current_hash
                   and c["sidecar"]["after_sha256"] == candidate_hash]
        if len(matches) != 1:
            raise RuntimeError("Accepted history is not a unique fully checked candidate chain")
        accepted_checks.append(matches[0])
        current_hash = candidate_hash
    if current_hash != selected_hash:
        raise RuntimeError("Selected skill differs from the gated candidate chain")
    navigation_comparison = {"status": "not_run_no_upstream_candidate"}
    answer_guard = {"status": "not_run_no_upstream_candidate"}
    if result["accepted"]:
        baseline_nav = run_qa_batch(valid, overrides=snapshot, session_factory=session_factory,
                                    out_dir=destination / "navigation_gate_baseline", phase="validation",
                                    oracle=source_oracles["validation"]["queries"], sections=corpus["sections"],
                                    navigation_oracle=navigation_oracles["validation"]["queries"], active_target=target,
                                    section_aliases=section_aliases)
        candidate_snapshot = {**snapshot, target: result["selected_skill"]}
        candidate_nav = run_qa_batch(valid, overrides=candidate_snapshot, session_factory=session_factory,
                                     out_dir=destination / "navigation_gate_candidate", phase="validation",
                                     oracle=source_oracles["validation"]["queries"], sections=corpus["sections"],
                                     navigation_oracle=navigation_oracles["validation"]["queries"], active_target=target,
                                     section_aliases=section_aliases)
        base_metrics, cand_metrics = aggregate_navigation(baseline_nav), aggregate_navigation(candidate_nav)
        stable = (base_metrics["scorable_count"] == cand_metrics["scorable_count"]
                  and base_metrics["evidence_count"] == cand_metrics["evidence_count"]
                  and base_metrics["scorable_count"] >= minimum)
        component_guard = (stable and cand_metrics["primary_score"] > base_metrics["primary_score"]
                           and cand_metrics["route_complete"] >= base_metrics["route_complete"]
                           and (base_metrics["evidence_complete"] is None
                                or cand_metrics["evidence_complete"] >= base_metrics["evidence_complete"]))
        navigation_comparison = {"status": "passed" if component_guard else "rejected",
                                 "baseline": base_metrics, "candidate": cand_metrics,
                                 "stable_denominator": stable,
                                 "requirements": {"primary_strictly_improved": bool(stable and cand_metrics["primary_score"] > base_metrics["primary_score"]),
                                                  "route_complete_non_regression": bool(stable and cand_metrics["route_complete"] >= base_metrics["route_complete"]),
                                                  "evidence_complete_non_regression": bool(stable and (base_metrics["evidence_complete"] is None or cand_metrics["evidence_complete"] >= base_metrics["evidence_complete"]))}}
        if component_guard:
            judge = AnswerJudge(transport)
            try:
                baseline_answers = run_qa_batch(all_valid, overrides=snapshot, session_factory=session_factory,
                                                judge=judge, out_dir=destination / "answer_guard_baseline", phase="validation")
                candidate_answers = run_qa_batch(all_valid, overrides=candidate_snapshot, session_factory=session_factory,
                                                 judge=judge, out_dir=destination / "answer_guard_candidate", phase="validation")
                baseline_accuracy = sum(row["hard"] for row in baseline_answers) / len(baseline_answers)
                candidate_accuracy = sum(row["hard"] for row in candidate_answers) / len(candidate_answers)
                passed = candidate_accuracy >= baseline_accuracy
                base_by_id = {row["qid"]: row["hard"] for row in baseline_answers}
                cand_by_id = {row["qid"]: row["hard"] for row in candidate_answers}
                answer_guard = {"status": "passed" if passed else "rejected",
                                "baseline_accuracy": baseline_accuracy, "candidate_accuracy": candidate_accuracy,
                                "improved_qids": sorted(qid for qid in base_by_id if base_by_id[qid] < cand_by_id[qid]),
                                "regressed_qids": sorted(qid for qid in base_by_id if base_by_id[qid] > cand_by_id[qid]),
                                "eligible_count": len(all_valid)}
            except (JudgmentUnavailable, TransportError) as exc:
                passed = False
                answer_guard = {"status": "unavailable", "reason": str(exc), "eligible_count": len(all_valid)}
            if not passed:
                result["accepted"] = False
        else:
            result["accepted"] = False
    if not result["accepted"]:
        result["selected_skill"] = snapshot[target]

    result.update(active_target=target,
                  candidate_validations=accepted_checks if result["accepted"] else [],
                  navigation_candidate_validations=accepted_checks,
                  navigation_comparison=navigation_comparison, answer_guard=answer_guard,
                  batch_size=config["optimization"].get("batch_size") or len(train),
                  reflection_minibatch=config["optimization"].get("reflection_minibatch", 4),
                  phase_denominators={p: {k: m[k] for k in ("assigned_count", "eligible_count")}
                                      for p, m in phases.items()},
                  target_denominators={"optimization": len(train), "validation": len(valid),
                                       "answer_guard_validation": len(all_valid)})
    write_new(destination / "result.json", result)
    return result


def _subprocess_worker(job_path, output_path):
    # No shell or state shared with another target's upstream module globals.
    log_path = output_path.parent / (output_path.name + ".log")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "qa_skillopt.cli", "_target", "--job", str(job_path)],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise RuntimeError(f"SkillOPT target process failed ({status}); inspect {log_path}")
    return read_json(output_path / "result.json")


def train_schedule(config_path, config, out_dir, *, confirmed=False, worker=_subprocess_worker):
    from .cards import build_navigation_card_snapshot
    from qa_agent.factory import load_online_config

    if confirmed is not True:
        raise ValueError("No generation: confirm the AMD usage window before train")
    online = load_online_config(config["online_config"])
    _, _, phases = load_training_data(config["preparation_dir"])
    output = new_directory(out_dir, protected=(online["bundle_dir"], config["preparation_dir"], config["upstream_source"]))
    snapshot = build_navigation_card_snapshot(online["bundle_dir"])
    nav_manifest = read_json(Path(config["preparation_dir"]) / "navigation_manifest.json")
    if nav_manifest.get("bundle_manifest_sha256") != sha256(Path(online["bundle_dir"]) / "bundle_manifest.json"):
        raise ValueError("Navigation preparation and serving bundle disagree")
    if nav_manifest.get("min_target_samples") != config["optimization"].get("min_target_samples", 10):
        raise ValueError("Configured target threshold differs from sealed navigation preparation")
    expected_cards = nav_manifest.get("card_snapshot_sha256", {})
    if expected_cards != {key: digest_text(text) for key, text in snapshot.items()}:
        raise ValueError("Navigation card baseline changed after preparation")
    order = list(nav_manifest.get("eligible_targets", []))
    if not order or order[0] != "overall" or any(target not in snapshot for target in order):
        raise ValueError("Invalid eligible target order in navigation preparation")
    base_hash = sha256(Path(online["bundle_dir"]) / "bundle_manifest.json")
    sidecar = {"schema_version": "qa-skillopt-selection-v2", "joint_accepted": True,
               "base_bundle_manifest_sha256": base_hash, "candidate_validations": [],
               "navigation_mode": "cards-v1", "card_baseline_sha256": expected_cards,
               "eligible_targets": order, "skipped_targets": nav_manifest.get("skipped_targets", []),
               "selection_basis": "navigation_primary_with_answer_non_regression",
               "evaluation_label": "prototype_navigation_oracle_same_model_answer_guard",
               "optimization_settings": config["optimization"],
               "preparation_id": phases["optimization"]["preparation_id"],
               "source_dataset_sha256": phases["optimization"].get("source_dataset_sha256"),
               "config_sha256": sha256(Path(config_path)),
               "online_config_sha256": sha256(Path(config["online_config"])),
               "phase_denominators": {p: {k: m[k] for k in ("assigned_count", "eligible_count")} for p, m in phases.items()},
               "history": [], "test_evaluated": False}
    state = {"snapshot": snapshot, "sidecar": sidecar, "last_accepted_version": 0}
    write_new(output / "accepted_000.json", state)
    step, version = 0, 0
    try:
        for cycle in range(1, config["optimization"]["max_cycles"] + 1):
            accepted_in_cycle = 0
            for target in order:
                step += 1
                target_dir = output / f"step_{step:03d}"
                job = {"config_path": str(Path(config_path).resolve()), "active_target": target,
                       "snapshot": snapshot, "output_dir": str(target_dir), "usage_window_confirmed": True,
                       "config_sha256": sidecar["config_sha256"], "online_config_sha256": sidecar["online_config_sha256"]}
                job_path = output / f"job_{step:03d}.json"
                write_new(job_path, job)
                result = worker(job_path, target_dir)
                sidecar["history"].append({"cycle": cycle, "target": target, "accepted": result["accepted"],
                                          "best_validation_score": result["best_validation_score"], "step": step,
                                          "navigation_comparison": result.get("navigation_comparison"),
                                          "answer_guard": result.get("answer_guard"),
                                          "target_denominators": result.get("target_denominators")})
                if result["accepted"]:
                    snapshot = {**snapshot, target: result["selected_skill"]}
                    sidecar["candidate_validations"].extend(result["candidate_validations"])
                    sidecar["selection_basis"] = "navigation_primary_with_answer_non_regression"
                    accepted_in_cycle += 1
                    version += 1
                    state = {"snapshot": snapshot, "sidecar": sidecar, "last_accepted_version": version}
                    write_new(output / f"accepted_{version:03d}.json", state)
            if not accepted_in_cycle:
                break
    except BaseException as exc:
        write_new(output / "interrupted.json", {"last_accepted_version": version, "error_type": type(exc).__name__,
                                               "reason": str(exc), "test_evaluated": False})
        raise
    state = {"snapshot": snapshot, "sidecar": sidecar, "last_accepted_version": version}
    write_new(output / "selected.json", state)
    return {"status": "completed", "accepted_updates": version, "steps": step,
            "selected": str(output / "selected.json"), "test_evaluated": False}
