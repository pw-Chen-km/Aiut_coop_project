"""Private prepare / doctor / train / export / held-out evaluate commands."""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys

from .io import new_directory, read_json, write_new


def _parser():
    parser = argparse.ArgumentParser(prog="qa-skillopt")
    sub = parser.add_subparsers(dest="command", required=True)
    joint_prep = sub.add_parser('prepare-joint', help='Rebind an existing split to a NEW adaptive bundle')
    joint_prep.add_argument('--preparation-dir', required=True)
    joint_prep.add_argument('--bundle-dir', required=True)
    joint_prep.add_argument('--out-dir', required=True)
    native_prep = sub.add_parser('prepare-multidoc2dial-joint', help='Explicit official-train assignments and native GT for SkillOPT v2')
    native_prep.add_argument('--root', required=True)
    native_prep.add_argument('--bundle-dir', required=True)
    native_prep.add_argument('--assignments', required=True)
    native_prep.add_argument('--out-dir', required=True)
    native_prep.add_argument('--view', choices=['opening-answer', 'all'], default='opening-answer')
    joint_train = sub.add_parser('train-joint', help='Explicit cross-layer SkillOPT v2 run')
    joint_train.add_argument('--config', required=True)
    joint_train.add_argument('--out-dir', required=True)
    joint_train.add_argument('--usage-window-confirmed', action='store_true')
    joint_export = sub.add_parser('export-joint', help='Export complete joint MD snapshot to a new directory')
    joint_export.add_argument('--bundle-dir', required=True)
    joint_export.add_argument('--selection', required=True)
    joint_export.add_argument('--out-dir', required=True)
    prep = sub.add_parser("prepare", help="Review-gated 40/40/120 split and exact oracle")
    prep.add_argument("--qa", required=True)
    prep.add_argument("--bundle-dir", required=True)
    prep.add_argument("--decisions", help="Attributed, question-hash-bound duplicate decisions JSON")
    prep.add_argument("--out-dir", required=True)
    navprep = sub.add_parser("prepare-navigation", help="Add navigation-only oracle/card manifests to a sealed split")
    navprep.add_argument("--preparation-dir", required=True)
    navprep.add_argument("--bundle-dir", required=True)
    navprep.add_argument("--out-dir", required=True)
    navprep.add_argument("--min-target-samples", type=int, default=10)
    for name in ("doctor", "train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        if name == "doctor":
            cmd.add_argument("--check-endpoint", action="store_true", help="Execution host only: read-only AMD API checks")
            cmd.add_argument("--check-online", action="store_true", help="Local QA assets/tokenizer/budget checks, no generation")
        else:
            cmd.add_argument("--out-dir", required=True)
            cmd.add_argument("--usage-window-confirmed", action="store_true",
                             help="Operator has explicitly confirmed availability of the shared AMD service")
        if name == "evaluate":
            cmd.add_argument("--frozen-bundle", required=True)
    export = sub.add_parser("export", help="Export a complete accepted state to a NEW serving bundle")
    export.add_argument("--bundle-dir", required=True)
    export.add_argument("--selection", required=True, help="selected.json or last accepted_NNN.json after interruption")
    export.add_argument("--out-dir", required=True)
    target = sub.add_parser("_target", help=argparse.SUPPRESS)
    target.add_argument("--job", required=True)
    return parser


def doctor(config, *, check_endpoint=False, check_online=False):
    from .adapter import load_training_data
    from .upstream import verify_source
    from .scheduler import make_transport
    from qa_agent.bundle import validate_serving_bundle
    from qa_agent.factory import load_online_config

    online = load_online_config(config["online_config"])
    bundle = validate_serving_bundle(online["bundle_dir"])
    train, valid, phases = load_training_data(config["preparation_dir"])
    nav_manifest = read_json(Path(config["preparation_dir"]) / "navigation_manifest.json")
    result = {"status": "preflight_complete", "upstream": verify_source(config["upstream_source"]),
              "bundle_counts": bundle["counts"], "optimization_eligible": len(train), "validation_eligible": len(valid),
              "phase_denominators": {p: {k: m[k] for k in ("assigned_count", "eligible_count")} for p, m in phases.items()},
              "navigation_targets": nav_manifest.get("eligible_targets", []),
              "skipped_navigation_targets": nav_manifest.get("skipped_targets", []),
              "generation_requested": False, "availability_confirmed": False, "test_read": False,
              "note": "Loaded does not mean idle. Confirm the shared AMD usage window before generation."}
    if check_endpoint:
        result["offline_model"] = make_transport(config).preflight()
    if check_online:
        from qa_agent.factory import create_session
        from .cards import build_navigation_card_snapshot
        from .validation import validate_navigation_budget
        with create_session(online) as session:
            router = session.agent.router
            card_snapshot = build_navigation_card_snapshot(online["bundle_dir"])
            result["online_manifest"] = session.manifest
            result["navigation_budget"] = validate_navigation_budget(online["bundle_dir"],
                card_snapshot, token_counter=router.token_counter,
                max_input_tokens=router.config["max_input_tokens"])
            if not result["navigation_budget"]["valid"]:
                raise ValueError("Initial navigation snapshot exceeds the fixed context budget")
            from urllib.request import ProxyHandler, build_opener
            # create_session has already enforced a loopback-only target client.
            with build_opener(ProxyHandler({})).open(online["runtime"]["endpoint"].rstrip("/") + "/models", timeout=10) as response:
                models = json.load(response)
            available = [r.get("id") for r in models.get("data", []) if isinstance(r, dict)]
            if online["runtime"]["model"] not in available:
                raise ValueError("Configured QA target is not served by its existing endpoint")
            result["online_model_connection"] = {"status": "available", "generation_requested": False}
    return result


def evaluate(config, frozen_bundle, out_dir, *, confirmed=False):
    from .adapter import load_phase, run_qa_batch
    from .judge import AnswerJudge
    from .scheduler import make_transport
    from qa_agent.factory import create_session, load_online_config
    from qa_agent.bundle import validate_serving_bundle
    from doc2skill.config import sha256

    if confirmed is not True:
        raise ValueError("Held-out generation also requires confirmed AMD availability")
    bundle = Path(frozen_bundle).resolve()
    manifest = validate_serving_bundle(bundle)
    if not manifest.get("skillopt_snapshot", {}).get("joint_accepted"):
        raise ValueError("Test evaluation requires a previously exported complete selected snapshot")
    online = load_online_config(config["online_config"])
    online["bundle_dir"] = str(bundle)
    rows, phase = load_phase(config["preparation_dir"], "test", final_evaluation=True)
    selection = read_json(bundle / manifest["skillopt_snapshot"]["acceptance_sidecar"])
    if not phase.get("preparation_id") or phase["preparation_id"] != selection.get("preparation_id"):
        raise ValueError("Held-out split does not belong to the selected experiment")
    if phase.get("bundle_manifest_sha256") != manifest["skillopt_snapshot"].get("base_bundle_manifest_sha256"):
        raise ValueError("Held-out split and frozen snapshot originate from different corpora")
    output = new_directory(out_dir, protected=(bundle, config["preparation_dir"]))
    frozen_hash = sha256(bundle / "bundle_manifest.json")
    write_new(output / "frozen_version.json", {"bundle_manifest_sha256": frozen_hash, "test_manifest": phase,
              "selection_performed": False, "optimizer_feedback_allowed": False, "prototype": True})
    results = run_qa_batch(rows, overrides=None, session_factory=functools.partial(create_session, online),
                          judge=AnswerJudge(make_transport(config, confirmed=True)), out_dir=output / "evaluation", phase="test")
    if sha256(bundle / "bundle_manifest.json") != frozen_hash:
        raise RuntimeError("Frozen bundle changed during evaluation")
    summary = {"status": "completed", "assigned_count": phase["assigned_count"], "eligible_count": len(rows),
               "answer_accuracy": sum(r["hard"] for r in results) / len(rows),
               "prototype_same_model_judge": True, "selection_performed": False}
    write_new(output / "summary.json", summary)
    return summary


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == 'prepare-joint':
            from .joint_run import prepare_joint
            result = prepare_joint(args.preparation_dir, args.bundle_dir, args.out_dir)
        elif args.command == 'prepare-multidoc2dial-joint':
            from .multidoc_preparation import prepare_multidoc_joint
            result = prepare_multidoc_joint(args.root, args.bundle_dir, args.assignments, args.out_dir, view=args.view)
        elif args.command == 'train-joint':
            from .joint_run import load_joint_config, train_joint
            result = train_joint(load_joint_config(args.config), args.out_dir, confirmed=args.usage_window_confirmed)
        elif args.command == 'export-joint':
            from .joint_run import export_joint
            result = export_joint(args.bundle_dir, args.selection, args.out_dir)
        elif args.command == "prepare":
            from .data import write_preparation
            value = write_preparation(args.qa, args.bundle_dir, args.out_dir, args.decisions)
            result = {"status": value["status"], "out_dir": args.out_dir,
                      "unreviewed_pairs": len(value.get("unreviewed_candidate_ids", [])),
                      "balance": value.get("balance")}
        elif args.command == "prepare-navigation":
            from .data import write_navigation_preparation
            value = write_navigation_preparation(args.preparation_dir, args.bundle_dir, args.out_dir,
                                                  min_target_samples=args.min_target_samples)
            result = {"status": "ready", "out_dir": args.out_dir,
                      "eligible_targets": value["eligible_targets"],
                      "skipped_targets": value["skipped_targets"]}
        elif args.command == "export":
            from .export import export_snapshot
            state = read_json(args.selection)
            result = export_snapshot(args.bundle_dir, args.out_dir, state["snapshot"], sidecar=state["sidecar"])
        elif args.command == "_target":
            from .scheduler import target_worker
            value = target_worker(args.job)
            result = {"status": "completed", "accepted": value["accepted"]}
        else:
            from .config import load_config
            config = load_config(args.config)
            if args.command == "doctor":
                result = doctor(config, check_endpoint=args.check_endpoint, check_online=args.check_online)
            elif args.command == "train":
                from .scheduler import train_schedule
                result = train_schedule(args.config, config, args.out_dir, confirmed=args.usage_window_confirmed)
            else:
                result = evaluate(config, args.frozen_bundle, args.out_dir, confirmed=args.usage_window_confirmed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "pending_duplicate_review" else 3
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
