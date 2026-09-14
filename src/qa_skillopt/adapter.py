"""The real SkillOPT environment adapter around complete, gold-blind QA runs."""
from __future__ import annotations

from copy import deepcopy
import json
import random
import time
from pathlib import Path

from doc2skill.config import sha256
from .io import digest_text, read_json, read_jsonl, write_new


NAVIGATION_CONSTRAINTS = """QURSOR-specific constraints take priority over generic
editing suggestions. The supplied Markdown is ONLY a set of navigation cards: it
decides where to search and is never an answer manual. Learn reusable routing cues
from the user's wording and selected-vs-required IDs. Overall feedback concerns
document selection only. Document feedback concerns section selection only.
Do not infer product answers, procedures, parameter values or operational rules.
Do not add qids, full-question lookup rules, answers, evidence, citations or source
quotes. Only the following existing JSON-card fields may change: use_when,
not_when, question_intents, aliases, confusable_with, distinguishing_signals.
Never change card IDs, routing IDs, titles, hierarchy, order, paths, Markdown
scaffold, policy, retrieval or model settings. confusable_with may name only an
existing same-level card ID. Keep useful descriptions from routing successes and
propose no edit when routing feedback does not support one. At most 3 field edits.
"""


class PhaseBatch(list):
    def __init__(self, rows, phase):
        super().__init__(deepcopy(rows))
        self.phase = phase


def load_phase(directory, phase, *, final_evaluation=False):
    if phase not in {"optimization", "validation"} and not (phase == "test" and final_evaluation):
        raise ValueError("Held-out phase access denied")
    root = Path(directory).resolve()
    # Never enumerate directories or open split_manifest, which contains review
    # packets from the full dataset. Training opens exactly its two phase files.
    path = root / (phase + ".jsonl")
    manifest = read_json(root / (phase + "_manifest.json"))
    if manifest.get("phase") != phase or manifest.get("sha256") != sha256(path):
        raise ValueError("Phase payload fingerprint mismatch")
    rows = read_jsonl(path)
    if (len(rows) != manifest.get("eligible_count") or not rows
            or sorted(q["qid"] for q in rows) != sorted(manifest.get("qids", []))
            or len({q["qid"] for q in rows}) != len(rows)):
        raise ValueError("Phase eligibility/denominator mismatch")
    return rows, manifest


def load_training_data(directory):
    train, tm = load_phase(directory, "optimization")
    valid, vm = load_phase(directory, "validation")
    if set(q["qid"] for q in train) & set(q["qid"] for q in valid):
        raise ValueError("Optimization and validation IDs overlap")
    if tm.get("preparation_id") != vm.get("preparation_id") or not tm.get("preparation_id"):
        raise ValueError("Phase files must originate from the same sealed preparation")
    return train, valid, {"optimization": tm, "validation": vm}


def run_qa_batch(rows, *, overrides, session_factory, judge=None, out_dir, phase,
                 oracle=None, sections=(), navigation_oracle=None, active_target=None,
                 section_aliases=None):
    """Run QA while keeping answer judging and navigation reward separate.

    ``active_target`` plus ``navigation_oracle`` selects navigation-only mode.
    The target model still runs the complete bounded QA loop, but the offline
    answer judge is not called and its answer never enters SkillOPT feedback.
    """
    from .trajectory import compact_trajectory, diagnose_trajectory
    from .navigation_feedback import aggregate_navigation, navigation_feedback

    output, results = Path(out_dir), []
    output.mkdir(parents=True, exist_ok=True)
    with session_factory(navigation_md_overrides=overrides) as session:
        write_new(output / "qa_runtime_manifest.json", session.manifest)
        for qa in rows:
            started = time.monotonic()
            print(f"[qa] phase={phase} qid={qa['qid']} started", flush=True)
            record_dir = output / "predictions" / digest_text(qa["qid"])
            record_dir.mkdir(parents=True, exist_ok=False)
            trajectory = session.agent.answer(qa["question"], qid=qa["qid"])
            write_new(record_dir / "raw_trajectory.json", trajectory)
            navigation_mode = active_target is not None and navigation_oracle is not None
            if navigation_mode:
                source_query = (oracle or {}).get(qa["qid"], {})
                route_query = navigation_oracle.get(qa["qid"], {})
                feedback = navigation_feedback(trajectory, source_query, route_query, sections, active_target,
                                               section_aliases=section_aliases)
                row = {"id": record_dir.name, "qid": qa["qid"], "hard": feedback["hard"],
                       "soft": float(feedback["soft"]), "task_type": qa["question_type"],
                       "n_turns": trajectory.get("rounds_used", 0), "phase": phase,
                       "navigation_scorable": feedback["scorable"],
                       "excluded_reason": feedback["excluded_reason"],
                       "route_complete": int(feedback["route"]["complete"]),
                       "route_precision": feedback["route"]["precision"],
                       "route_recall": feedback["route"]["recall"],
                       "route_f1": feedback["route"]["f1"],
                       "first_round_route_complete": int(feedback["first_round_route"]["complete"]),
                       "evidence_recall": feedback["evidence"].get("recall"),
                       "evidence_complete": bool(feedback["evidence"].get("complete")),
                       "evidence_status": feedback["evidence"].get("status", "unknown")}
                write_new(record_dir / "navigation_metrics.json", {
                    key: deepcopy(value) for key, value in feedback.items() if key != "optimizer_view"})
                if phase == "optimization" and feedback["scorable"]:
                    view = feedback["optimizer_view"]
                    # This is the only per-example payload read by upstream reflection.
                    write_new(record_dir / "navigation_trajectory.json", view)
                    write_new(record_dir / "conversation.json", [
                        {"role": "user", "content": json.dumps(view, ensure_ascii=False)}])
                    row["task_description"] = qa["question"]
            else:
                if judge is None:
                    raise ValueError("Answer evaluation requires an explicit judge")
                try:
                    judgment = judge.score(qa, trajectory)
                    write_new(record_dir / "model_judgment.json", judgment)
                except Exception as exc:
                    write_new(record_dir / "unscored.json", {"status": "comparison_stopped",
                              "error_type": type(exc).__name__, "reason": str(exc), "qid": qa["qid"]})
                    raise
                row = {"id": record_dir.name, "qid": qa["qid"], "hard": judgment["hard"],
                       "soft": float(judgment["hard"]), "task_type": qa["question_type"],
                       "n_turns": trajectory.get("rounds_used", 0), "phase": phase}
                # Retain the old diagnostic only for non-SkillOPT compatibility.
                # Navigation training never enters this branch.
                if phase == "optimization":
                    query_oracle = (oracle or {}).get(qa["qid"], {})
                    diagnostic = diagnose_trajectory(trajectory, query_oracle, sections, judgment=judgment)
                    compact = compact_trajectory(trajectory, diagnostic)
                    write_new(record_dir / "compact_trajectory.json", compact)
                    row.update(diagnosis=diagnostic,
                               observed_doc_ids=sorted({d for r in trajectory.get("rounds", [])
                                                        for d in r.get("scope", {}).get("doc_ids", [])}))
            results.append(row)
            print(f"[qa] phase={phase} qid={qa['qid']} hard={row['hard']} rounds={row['n_turns']} seconds={time.monotonic()-started:.2f}", flush=True)
    payload = {"phase": phase, "eligible_count": len(rows),
               "hard_accuracy": sum(r["hard"] for r in results) / len(rows), "results": results}
    if active_target is not None and navigation_oracle is not None:
        payload["active_target"] = active_target
        payload["navigation"] = aggregate_navigation(results)
    write_new(output / "qa_results.json", payload)
    return results


def make_adapter(*, train_rows, validation_rows, active_target, snapshot, session_factory,
                 judge=None, oracle=None, sections=(), navigation_oracles=None,
                 section_aliases=None):
    """Import optional upstream only inside its version-checked worker process."""
    from skillopt.envs.base import EnvAdapter

    class QURSORAdapter(EnvAdapter):
        def setup(self, cfg):
            super().setup(cfg)
            self.analyst_workers = 1
            self.failure_only = False
            self.minibatch_size = cfg.get("minibatch_size", 4)
            self.edit_budget = 3
            self._train_order = sorted(train_rows, key=lambda q: q["qid"])
            random.Random(cfg.get("seed", 42)).shuffle(self._train_order)
            self._train_cursor = 0
            if cfg.get("eval_test") is not False or cfg.get("use_slow_update") or cfg.get("use_meta_skill"):
                raise ValueError("Unsafe SkillOPT side updates or test evaluation")

        def build_train_env(self, batch_size, seed, **kwargs):
            if type(batch_size) is not int or not 1 <= batch_size <= len(train_rows):
                raise ValueError("Batch exceeds the full eligible optimization set")
            order = getattr(self, "_train_order", train_rows)
            cursor = getattr(self, "_train_cursor", 0)
            batch = order[cursor:cursor + batch_size]
            if not batch:
                raise ValueError("Optimization epoch exhausted; no repeated/padded questions")
            self._train_cursor = cursor + len(batch)
            return PhaseBatch(batch, "optimization")

        def build_eval_env(self, env_num, split, seed, **kwargs):
            if split not in {"val", "validation", "valid_seen"}:
                raise ValueError("Test/unknown phase denied inside the trainer")
            return PhaseBatch(validation_rows, "validation")

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            if not isinstance(env_manager, PhaseBatch) or env_manager.phase not in {"optimization", "validation"}:
                raise ValueError("Unapproved environment")
            edited = {**snapshot, active_target: skill_content}
            route_oracle = None if navigation_oracles is None else navigation_oracles[env_manager.phase]
            return run_qa_batch(env_manager, overrides=edited, session_factory=session_factory,
                                judge=judge, out_dir=out_dir, phase=env_manager.phase,
                                oracle=(oracle or {}).get(env_manager.phase, oracle or {}),
                                navigation_oracle=route_oracle, active_target=active_target,
                                sections=sections, section_aliases=section_aliases)

        def get_task_types(self):
            return sorted({q["question_type"] for q in train_rows + validation_rows})

        def reflect(self, results, skill_content, out_dir, **kwargs):
            if any(r.get("phase") != "optimization" for r in results):
                raise ValueError("Only optimization trajectories can enter reflection")
            if navigation_oracles is not None:
                applicable = [row for row in results if row.get("navigation_scorable")]
            else:
                applicable = []
                for row in results:
                    exposed = active_target == "overall" or active_target.removeprefix("document:") in row.get("observed_doc_ids", [])
                    diagnostic = row.get("diagnosis", {})
                    nav_failure = diagnostic.get("failure_stage") in {"document_selection", "section_selection"}
                    if exposed and (row["hard"] == 1 or nav_failure):
                        applicable.append(row)
            if not applicable:
                write_new(Path(out_dir) / "no_applicable_navigation_cases.json", {"active_target": active_target})
                return []
            write_new(Path(out_dir) / "reflection_selection.json", {
                "active_target": active_target, "batch_qids": [r["qid"] for r in results if "qid" in r],
                "eligible_ids": [r["id"] for r in applicable], "reflection_minibatch": getattr(self, "minibatch_size", 4),
                "success_count": sum(r["hard"] == 1 for r in applicable),
                "failure_count": sum(r["hard"] == 0 for r in applicable),
                "success_definition": "target_route_complete",
                "failure_definition": "target_route_incomplete"})
            return super().reflect(applicable, skill_content, out_dir, **kwargs)

        def get_error_minibatch_prompt(self):
            return NAVIGATION_CONSTRAINTS + "\n" + (super().get_error_minibatch_prompt() or "")

        def get_success_minibatch_prompt(self):
            return NAVIGATION_CONSTRAINTS + "\n" + (super().get_success_minibatch_prompt() or "")

    return QURSORAdapter()
