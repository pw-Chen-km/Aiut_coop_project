"""The only online wiring module that knows concrete storage/model adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from doc2skill.config import fingerprint, load_config, sha256
from .bundle import validate_serving_bundle
from .pipeline import QAAgent
from .retrieval import ScopedDenseRetriever

ANSWER_OPTIONS = {"reading_tokens", "max_input_tokens", "max_output_tokens", "max_answer_chars",
                  "timeout_seconds", "prompt_path"}


def create_comparison_session(config):
    from .comparison_runtime import create_comparison_session as create
    return create(config)


def _loop_options(config):
    options = config.get("loop", {})
    if not isinstance(options, dict) or set(options) - {"max_rounds"}:
        raise ValueError("loop accepts only max_rounds")
    rounds = options.get("max_rounds", 2)
    if type(rounds) is not int or not 1 <= rounds <= 5:
        raise ValueError("loop.max_rounds must be between 1 and 5, including the initial retrieval")
    return rounds


def load_online_config(path) -> dict:
    """Resolve deployment paths relative to the config file, never the shell cwd."""
    config = load_config(path)
    base = Path(path).resolve().parent
    if not config.get("bundle_dir"):
        raise ValueError("bundle_dir is required; use a portable export-bundle output")
    config.setdefault("answer", {})
    config.setdefault("retrieval", {})
    _loop_options(config)
    if set(config["answer"]) - ANSWER_OPTIONS:
        raise ValueError("Unknown answer options; endpoint/model belong only in runtime")
    for mapping, key in ((config, "bundle_dir"), (config["embedding"], "cache_dir"),
                         (config["embedding"], "local_path"),
                         (config["runtime"], "gguf_path"), (config["runtime"], "tokenizer_path"),
                         (config["runtime"], "policy_path"), (config["answer"], "prompt_path")):
        if mapping.get(key):
            value = Path(mapping[key]).expanduser()
            mapping[key] = str((base / value).resolve() if not value.is_absolute() else value.resolve())
    return config


@dataclass
class RuntimeSession:
    agent: QAAgent
    store: object
    manifest: dict

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_session(config: dict, *, navigation_md_overrides=None) -> RuntimeSession:
    max_rounds = _loop_options(config)
    from doc2skill.storage import Store
    from doc2skill.embedding import E5Encoder
    from doc2skill.runtime import validate_execution_mode, validate_runtime_identity
    from .navigation import SkillRouter
    from doc2skill.llm import ChatClient
    from .generation import AnswerGenerator

    root = Path(config["bundle_dir"])
    bundle = validate_serving_bundle(root)
    if bundle.get('schema_version') == 'qa-serving-bundle-v2' and 'max_rounds' not in config.get('loop', {}):
        max_rounds = 5
    runtime = {"max_input_chars": 100000, "max_input_tokens": 7000, "context_tokens": 8192,
               "max_output_tokens": 512, "execution_mode": "cpu", "cpu_only": True, "gpu_layers": 0,
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **config["runtime"]}
    if runtime.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("Online QA requires non-thinking output; set enable_thinking=false")
    execution_mode = validate_execution_mode(runtime)
    counter = validate_runtime_identity(runtime)
    runtime["token_counter"] = counter
    client = ChatClient(runtime, local_only=True)
    if set(config.get("answer", {})) - ANSWER_OPTIONS:
        raise ValueError("Unknown answer option")
    answer_config = {**runtime, "reading_tokens": 3000, "max_output_tokens": 768,
                     "max_answer_chars": 6000, **config.get("answer", {})}
    for options in (runtime, answer_config):
        if options["max_input_tokens"] < 1 or options["max_input_tokens"] + options["max_output_tokens"] + 256 > options["context_tokens"]:
            raise ValueError("Input budget plus output and 256-token reserve exceeds runtime context")
    answer_client = ChatClient(answer_config, local_only=True)
    store = Store(root / "corpus.sqlite")
    try:
        if store.metadata.get("embedding") != bundle["embedding"]:
            raise ValueError("Database and serving embedding identities disagree")
        ec = {**config["embedding"], "model": bundle["embedding"]["model"],
              "revision": bundle["embedding"]["revision"]}
        encoder = E5Encoder(ec, local_only=True)
        policy = Path(runtime["policy_path"]).read_text(encoding="utf-8") if runtime.get("policy_path") else None
        answer_prompt = Path(answer_config["prompt_path"]).read_text(encoding="utf-8") if answer_config.get("prompt_path") else None
        import json
        from .hierarchical_navigation import HierarchicalSkillRouter
        paths = json.loads((root / "skill_paths.json").read_text())
        from .adaptive_navigation import AdaptiveSkillRouter
        router_type = (AdaptiveSkillRouter if paths.get('adaptive_tree') else
                       HierarchicalSkillRouter if paths.get("hierarchy") else SkillRouter)
        router = router_type(store.records("documents"), store.records("sections"), client, root,
                             runtime, token_counter=counter, skill_content=policy,
                             navigation_md_overrides=navigation_md_overrides)
        retriever = ScopedDenseRetriever(store, encoder)
        generator = AnswerGenerator(answer_client, answer_config, token_counter=counter, prompt_content=answer_prompt)
        agent = QAAgent(router, retriever, generator, top_k=config.get("retrieval", {}).get("top_k", 20),
                        max_rounds=max_rounds)
        # This is declared runtime identity, not proof of the external server hardware.
        identity = {k: runtime[k] for k in ("model", "endpoint", "gguf_sha256", "llama_cpp_revision",
                                          "chat_template_sha256", "execution_mode", "cpu_only", "gpu_layers")}
        identity["hardware_attested"] = False
        identity["evaluation_label"] = ("gpu_test_not_cpu_benchmark" if execution_mode == "gpu_test"
                                        else "cpu_run_hardware_unverified")
        manifest = {"schema_version": "qa-agent-run-v1", "bundle_manifest_sha256": sha256(root / "bundle_manifest.json"),
                    "runtime": identity, "embedding": encoder.provenance, "config_sha256": fingerprint(config),
                    "settings": {
                        "loop": {"max_rounds": agent.max_rounds, "includes_initial_round": True,
                                 "trigger": "answerable_false", "no_progress_stop": True,
                                 "evidence_pool": "new_or_unread_originals_first_then_previous",
                                 "reading_tokens_budget": "per_round", "transport_error_retry": False},
                        "navigation": {k: router.config[k] for k in ("max_documents", "max_sections", "max_input_tokens",
                            "max_active_branches", "max_scopes", "max_navigation_calls") if k in router.config},
                        "retrieval": {"top_k": agent.top_k},
                        "answer": {k: getattr(generator, k) for k in ("reading_tokens", "max_input_tokens", "max_output_tokens", "max_answer_chars")},
                        "context_tokens": runtime["context_tokens"], "temperature": runtime["temperature"],
                        "navigation_output_tokens": runtime["max_output_tokens"],
                        "timeout_seconds": client.timeout, "answer_timeout_seconds": answer_client.timeout,
                        "chat_template_kwargs": runtime["chat_template_kwargs"]},
                    "answer_prompt_sha256": generator.prompt_sha256,
                    "navigation_policy_override_sha256": fingerprint(policy) if policy is not None else None,
                    "answer_prompt_override_sha256": fingerprint(answer_prompt) if answer_prompt is not None else None,
                    "flow": ["navigation", "scoped_dense", "answer"], "global_expansion": False,
                    "recovery": {"enabled": agent.max_rounds > 1, "maximum_rechecks": agent.max_rounds - 1,
                                 "mode": "skill_guided_renavigation", "previous_answers_as_evidence": False}}
        if router.navigation_md_overrides:
            manifest["navigation_md_override_sha256"] = router.navigation_override_hashes()
        return RuntimeSession(agent, store, manifest)
    except Exception:
        store.close()
        raise


def model_identity(gguf_path, tokenizer_path) -> dict:
    """Read local files to fill the pinned runtime config; no downloads or inference."""
    from transformers import AutoTokenizer
    import hashlib
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("Expected a tokenizer with one explicit chat template")
    return {"gguf_sha256": sha256(gguf_path),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
