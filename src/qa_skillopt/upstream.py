"""Pinned Microsoft SkillOpt runner, with explicit boundary checks only.

The optimization algorithm is ReflACTTrainer from the verified upstream tree.
We neither reproduce its loop nor substitute a local optimizer implementation.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterator


PINNED_VERSION = "0.2.0"
PINNED_COMMIT = "e4ea6a6771e797ef820cdd8bfea64c57e0481065"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "private/vendor/SkillOpt-v0.2.0"
_HASHES = {
    "skillopt/engine/trainer.py": "deae957533ae6b50ffc241bf91df4707230f5a09540c5a996875e92018eb9951",
    "skillopt/optimizer/skill.py": "1cb3ecd9da7ab5ea89ad86780f3e4536fcf5cda84455ea6ae8fcc9c86cc73d3f",
    "skillopt/model/qwen_backend.py": "d0e049fbe7f0674aa224c8546cff0bd520856a75ad0223c848e89925c197cc13",
    "skillopt/model/__init__.py": "b9753eb60761b99926724de0454bc58b3ff5b9e38a0c96295fcecc177740662f",
    "skillopt/evaluation/gate.py": "3dfa23e45bb24c4126c4b14f668c3df4315483caddd0767f72c64a6f30554e46",
}
_loaded_root: Path | None = None
CandidateValidator = Callable[[str, str, dict, list[dict]], bool | None | dict]


class UpstreamContractError(RuntimeError):
    pass


def verify_source(source_root: Path | str) -> dict:
    root = Path(source_root).resolve(strict=True)
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        if git("rev-parse", "HEAD") != PINNED_COMMIT:
            raise UpstreamContractError("SkillOpt checkout is not the exact pinned commit")
        if git("status", "--porcelain", "--untracked-files=all", "--", "skillopt"):
            raise UpstreamContractError("SkillOpt source tree has local changes or untracked source files")
    except subprocess.CalledProcessError as exc:
        raise UpstreamContractError("A verified pinned SkillOpt git checkout is required") from exc
    for relative, expected in _HASHES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise UpstreamContractError(f"Upstream compatibility hash mismatch: {relative}")
    return {"version": PINNED_VERSION, "commit": PINNED_COMMIT, "source_root": str(root),
            "compatibility_hashes": dict(_HASHES)}


def load_upstream(source_root: Path | str = DEFAULT_SOURCE_ROOT) -> Any:
    global _loaded_root
    root = Path(source_root).resolve(strict=True)
    verify_source(root)
    for name, module in tuple(sys.modules.items()):
        if name == "skillopt" or name.startswith("skillopt."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise UpstreamContractError("Another SkillOpt installation is already imported; use a fresh process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package = importlib.import_module("skillopt")
    if package.__version__ != PINNED_VERSION:
        raise UpstreamContractError("Imported SkillOpt version does not match the pin")
    trainer = importlib.import_module("skillopt.engine.trainer")
    if tuple(inspect.signature(trainer.apply_patch_with_report).parameters) != ("skill", "patch"):
        raise UpstreamContractError("Upstream patch application signature changed")
    _loaded_root = root
    return trainer


def verify_loaded_upstream() -> dict:
    if _loaded_root is None:
        raise UpstreamContractError("load_upstream must verify the source before installing any shim")
    return verify_source(_loaded_root)


def _patch_precheck(before: str, patch: dict, edit_budget: int | None) -> None:
    if not isinstance(patch, dict) or not isinstance(patch.get("edits"), list):
        raise UpstreamContractError("Patch must contain an edits array")
    edits = patch["edits"]
    if edit_budget is not None and len(edits) > edit_budget:
        raise UpstreamContractError("Ranked patch exceeds the total edit budget")
    # Targets may be introduced by earlier edits: check sequentially using the real applier.
    from skillopt.optimizer.skill import apply_edit
    current = before
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("op") not in {"append", "insert_after", "replace", "delete"}:
            raise UpstreamContractError("Unknown or malformed edit")
        if not isinstance(edit.get("content", ""), str) or not isinstance(edit.get("target", ""), str):
            raise UpstreamContractError("Edit content and target must be strings")
        if edit["op"] != "append":
            target = edit.get("target", "")
            if not target or current.count(target) != 1:
                raise UpstreamContractError("Edit target must match exactly once; fallback is forbidden")
        current = apply_edit(current, edit)


@contextmanager
def guarded_patch_application(trainer: Any, validator: CandidateValidator, reports: list[dict],
                              edit_budget: int = 3) -> Iterator[None]:
    original = trainer.apply_patch_with_report
    def checked(before: str, patch: dict) -> tuple[str, list[dict]]:
        application_reports: list[dict] = []
        candidate = before
        record: dict = {"valid": False, "edit_count": len(patch.get("edits", [])) if isinstance(patch, dict) else 0}
        try:
            _patch_precheck(before, patch, edit_budget)
            candidate, application_reports = original(before, patch)
            if any(not str(row.get("status", "")).startswith("applied") or
                   "fallback" in str(row.get("status", "")) for row in application_reports):
                raise UpstreamContractError("Patch application was partial, skipped, or used fallback")
            decision = validator(before, candidate, patch, application_reports)
            record["validator_report"] = decision
            valid = decision is None or decision is True or isinstance(decision, dict) and decision.get("valid") is True
            if not valid:
                raise UpstreamContractError("Candidate validator rejected the final patch")
            record["valid"] = True
        except Exception as exc:
            from .transport import TransportError
            from .judge import JudgmentUnavailable
            if isinstance(exc, (TransportError, JudgmentUnavailable)):
                raise
            record["reason"] = str(exc)
            record["error_type"] = type(exc).__name__
        record["before_hash"] = hashlib.sha256(before.encode()).hexdigest()
        record["candidate_hash"] = hashlib.sha256(candidate.encode()).hexdigest()
        reports.append(record)
        if not record["valid"]:
            return before, [{"status": "skipped_candidate_guard", "reason": record["reason"]}]
        return candidate, application_reports
    trainer.apply_patch_with_report = checked
    try:
        yield
    finally:
        trainer.apply_patch_with_report = original


def run_training(*, source_root: Path | str = DEFAULT_SOURCE_ROOT, adapter: Any,
                 initial_skill: Path | str, output_dir: Path | str, train_size: int,
                 batch_size: int | None = None,
                 candidate_validator: CandidateValidator, edit_budget: int = 3,
                 transport: Any = None, config_overrides: dict | None = None,
                 usage_window_confirmed: bool = False, joint_gate=None) -> dict:
    """Run one genuine upstream epoch; no test material is accepted.

    Adapters must hold only train/validation data, return no dataloader, and reject
    any eval split other than valid_seen/val. No resume or in-process re-training.
    """
    if usage_window_confirmed is not True:
        raise UpstreamContractError("Training requires an explicitly confirmed usage window")
    if isinstance(train_size, bool) or not isinstance(train_size, int) or train_size <= 0:
        raise ValueError("train_size must be positive")
    batch_size = train_size if batch_size is None else batch_size
    if type(batch_size) is not int or not 1 <= batch_size <= train_size:
        raise ValueError("batch_size must be positive and no larger than the eligible pool")
    if not (joint_gate is not None and edit_budget is None) and (isinstance(edit_budget, bool) or not isinstance(edit_budget, int) or not 1 <= edit_budget <= 3):
        raise ValueError("The total edit budget must be between 1 and 3")
    if not callable(candidate_validator):
        raise ValueError("A candidate validator is required")
    if adapter.get_dataloader() is not None:
        raise UpstreamContractError("V1 requires a train/validation-only adapter without SplitDataLoader")
    initial_path = Path(initial_skill).resolve(strict=True)
    initial_content = initial_path.read_text(encoding="utf-8")
    destination = Path(output_dir).absolute()
    if destination.exists() and any(destination.iterdir()):
        raise UpstreamContractError("A fresh empty output directory is required; resume is forbidden")
    trainer_module = load_upstream(source_root)
    from .transport import OLLAMA_BASE_URL, OLLAMA_MODEL
    cfg = {
        "env": "qa_skillopt", "skill_init": str(initial_path), "out_root": str(destination),
        "model_backend": "qwen_chat", "optimizer_backend": "qwen_chat", "target_backend": "qwen_chat",
        "optimizer_model": OLLAMA_MODEL, "target_model": OLLAMA_MODEL,
        "qwen_chat_base_url": OLLAMA_BASE_URL, "qwen_chat_temperature": 0,
        "qwen_chat_enable_thinking": False, "qwen_chat_max_tokens": 8192,
        "qwen_chat_timeout_seconds": 300, "reasoning_effort": "",
        "train_size": train_size, "batch_size": batch_size, "num_epochs": 1, "accumulation": 1,
        "seed": 42, "merge_batch_size": 8, "minibatch_size": 4, "analyst_workers": 1, "workers": 1,
        "max_analyst_rounds": 1, "failure_only": False,
        # A scheduler integer is required upstream; the v2 compatibility layer
        # removes clipping at reflection AND rank boundaries, not by a huge cap.
        "edit_budget": edit_budget or 1, "min_edit_budget": edit_budget or 1, "lr_scheduler": "constant",
        "lr_control_mode": "fixed", "skill_update_mode": "patch",
        # hard=target route complete; soft=(route F1 + evidence recall)/2.
        # Upstream mixed therefore implements 0.50 complete + 0.25 F1 + 0.25 evidence.
        "use_gate": True, "gate_metric": "mixed", "gate_mixed_weight": 0.5,
        "sel_env_num": 0, "test_env_num": 0, "eval_test": False,
        "use_slow_update": False, "use_meta_skill": False, "use_skill_aware_reflection": False,
        "slow_update_gate_with_selection": True,
    }
    # Deliberately small allowlist: forbidden flags cannot be snuck through YAML.
    allowed = {"seed", "minibatch_size", "merge_batch_size", "failure_only",
               "qwen_chat_max_tokens", "qwen_chat_timeout_seconds"}
    overrides = dict(config_overrides or {})
    if set(overrides) - allowed:
        raise UpstreamContractError(f"Unsupported training overrides: {sorted(set(overrides) - allowed)}")
    for key in allowed - {"failure_only", "seed"}:
        if key in overrides and (isinstance(overrides[key], bool) or not isinstance(overrides[key], (int, float))
                                 or overrides[key] <= 0):
            raise ValueError(f"{key} must be positive")
    cfg.update(overrides)
    if transport is None:
        raise UpstreamContractError("An explicit native Ollama transport is required; raw backend fallback is forbidden")
    reports: list[dict] = []
    original_build_eval = adapter.build_eval_env
    original_setup = adapter.setup
    original_rollout = adapter.rollout
    def checked_eval(*args: Any, **kwargs: Any) -> Any:
        split = kwargs.get("split", args[1] if len(args) > 1 else None)
        if split not in {"valid_seen", "val"}:
            raise UpstreamContractError("Test and unknown splits are forbidden during training")
        env = original_build_eval(*args, **kwargs)
        if hasattr(env, "__len__") and len(env) == 0:
            raise UpstreamContractError("Validation set must be nonempty")
        return env
    def checked_setup(config: dict) -> None:
        original_setup(config)
        if adapter.get_dataloader() is not None:
            raise UpstreamContractError("Adapter setup introduced a forbidden dataloader")
    def checked_rollout(env_manager: Any, skill_content: str, out_dir: str, **kwargs: Any) -> list[dict]:
        results = original_rollout(env_manager, skill_content, out_dir, **kwargs)
        if (not isinstance(results, list) or not results or not hasattr(env_manager, "__len__")
                or len(results) != len(env_manager)):
            raise UpstreamContractError("Rollout changed the fixed evaluation denominator")
        identifiers: list[str] = []
        for row in results:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                    or type(row.get("hard")) is not int or row["hard"] not in {0, 1}
                    or isinstance(row.get("soft"), bool) or not isinstance(row.get("soft"), (int, float))
                    or not math.isfinite(row["soft"]) or not 0 <= row["soft"] <= 1):
                raise UpstreamContractError("Rollout must provide finite navigation scores and unique IDs")
            identifiers.append(row["id"])
        if len(set(identifiers)) != len(identifiers):
            raise UpstreamContractError("Rollout contains duplicate IDs")
        if all(isinstance(item, dict) and "qid" in item for item in env_manager):
            if [row.get("qid") for row in results] != [item["qid"] for item in env_manager]:
                raise UpstreamContractError("Rollout question IDs do not match the fixed requested batch")
        return results
    adapter.build_eval_env = checked_eval
    adapter.setup = checked_setup
    adapter.rollout = checked_rollout
    try:
        from .joint_upstream import joint_compatibility
        with guarded_patch_application(trainer_module, candidate_validator, reports, edit_budget), \
                transport.install_upstream_backend(), \
                (joint_compatibility(trainer_module, joint_gate) if joint_gate is not None else nullcontext()):
            summary = trainer_module.ReflACTTrainer(cfg, adapter).train()
    finally:
        adapter.build_eval_env = original_build_eval
        adapter.setup = original_setup
        adapter.rollout = original_rollout
    selected_path = destination / "best_skill.md"
    selected = selected_path.read_text(encoding="utf-8")
    history_path = destination / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    baseline_result_path = destination / "selection_eval_baseline/qa_results.json"
    if baseline_result_path.is_file():
        baseline_payload = json.loads(baseline_result_path.read_text(encoding="utf-8"))
        baseline_hard = float(baseline_payload["hard_accuracy"])
        baseline_soft = sum(float(row["soft"]) for row in baseline_payload["results"]) / len(baseline_payload["results"])
    else:
        # Generic/fake upstream adapters need not write QURSOR's audit file.
        baseline_hard = float(summary["baseline_selection_hard"])
        baseline_soft = baseline_hard
    baseline = .5 * baseline_hard + .5 * baseline_soft
    # SkillOPT v0.2.0 names this field best_selection_hard even when gate_metric
    # is mixed; it actually contains the metric-space best score.
    best = float(summary["best_selection_hard"])
    accepted = (joint_gate.selected_is_accepted(initial_content, selected) if joint_gate is not None
                else best > baseline and selected != initial_content)
    if not accepted and selected != initial_content:
        raise UpstreamContractError("Upstream changed the selected skill without strict validation improvement")
    if initial_path.read_text(encoding="utf-8") != initial_content:
        raise UpstreamContractError("The initial skill file was modified during optimization")
    summary["baseline_selection_soft"] = baseline_soft
    summary["baseline_gate_score"] = baseline
    summary["best_gate_score"] = best
    if joint_gate is not None:
        summary['joint_compatibility'] = {'effective_edit_limit': None,
            'reflection_truncation': False, 'rank_clipping': False,
            'gate': 'quality_or_equal_quality_pareto_cost',
            'note': 'Upstream scheduler logs an inert integer placeholder; it does not limit v2 edits.'}
    return {"summary": summary, "selected_skill": selected, "selected_skill_path": str(selected_path),
            "best_validation_score": best, "accepted": accepted, "history": history,
            "guard_reports": reports, "upstream": verify_loaded_upstream()}
