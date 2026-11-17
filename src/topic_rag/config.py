from pathlib import Path
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    for key in ("dataset_dir", "taxonomy", "labels", "index_dir", "results_dir",
                "e5_model_dir", "e5_tokenizer", "jev_prompt", "answer_prompt"):
        if key not in config:
            raise ValueError(f"missing config field: {key}")
        candidate = Path(config[key]).expanduser()
        config[key] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    if config.get("candidate_k") != 20 or config.get("final_k") != 5:
        raise ValueError("phase-one protocol fixes candidate_k=20 and answer final_k=5")
    if config.get("rrf_k") != 60 or config.get("soft_weights") != [1, 1, 1]:
        raise ValueError("phase-one protocol uses rrf_k=60 and three equal soft weights")
    if set(config.get("variants", [])) != {"baseline", "hard_topic", "soft_topic"}:
        raise ValueError("phase-one requires baseline, hard_topic, soft_topic")
    config["config_path"] = str(path)
    return config

