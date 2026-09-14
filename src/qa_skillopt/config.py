"""Configuration without secrets, held-out paths or automatic endpoint fallbacks."""
from __future__ import annotations

import json
from pathlib import Path
import tomllib


MODEL = "qwen3.6:35b-a3b-bf16"
MODEL_DIGEST = "94061ddd23a7de9c902f7bc468455aff03564d26e929b7e2e9ce62c5e6a3a492"


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream) if path.suffix == ".toml" else json.load(stream)
    if set(cfg) - {"online_config", "preparation_dir", "upstream_source", "offline", "optimization"}:
        raise ValueError("Unknown SkillOPT config fields; no test paths or credentials allowed")
    for name in ("online_config", "preparation_dir", "upstream_source"):
        if not isinstance(cfg.get(name), str) or not cfg[name].strip():
            raise ValueError(name + " is required")
        target = Path(cfg[name]).expanduser()
        cfg[name] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    offline = cfg.setdefault("offline", {})
    if set(offline) - {"base_url", "model", "model_digest", "timeout_seconds"}:
        raise ValueError("Unsupported offline option; service/model settings are immutable")
    offline.setdefault("base_url", "http://127.0.0.1:11437")
    offline.setdefault("model", MODEL)
    offline.setdefault("model_digest", MODEL_DIGEST)
    offline.setdefault("timeout_seconds", 600)
    if offline["base_url"] != "http://127.0.0.1:11437" or offline["model"] != MODEL or offline["model_digest"] != MODEL_DIGEST:
        raise ValueError("This prototype uses only the approved loopback model and digest")
    opt = cfg.setdefault("optimization", {})
    if set(opt) - {"max_cycles", "edit_budget", "batch_size", "reflection_minibatch", "min_target_samples"}:
        raise ValueError("Unknown optimization option; gate and leakage controls are fixed")
    opt.setdefault("max_cycles", 2)
    opt.setdefault("edit_budget", 3)
    opt.setdefault("batch_size", None)
    opt.setdefault("reflection_minibatch", 4)
    opt.setdefault("min_target_samples", 10)
    if opt["batch_size"] is not None and (type(opt["batch_size"]) is not int or opt["batch_size"] < 1):
        raise ValueError("batch_size must be a positive integer")
    if type(opt["reflection_minibatch"]) is not int or opt["reflection_minibatch"] < 1:
        raise ValueError("reflection_minibatch must be a positive integer")
    if type(opt["min_target_samples"]) is not int or opt["min_target_samples"] < 1:
        raise ValueError("min_target_samples must be a positive integer")
    if type(opt["max_cycles"]) is not int or not 1 <= opt["max_cycles"] <= 2 or opt["edit_budget"] != 3:
        raise ValueError("Maximum two cycles, exactly a three-edit upper bound")
    return cfg
