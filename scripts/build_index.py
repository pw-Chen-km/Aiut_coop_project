import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from topic_rag.config import load_config
from topic_rag.indexing import build_index
from topic_rag.validation import validate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local E5 vectors; does not run any QA question")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/emanual.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_dataset(config)
    print(json.dumps(build_index(Path(config["dataset_dir"]) / "chunks.jsonl",
                                 config["e5_model_dir"], Path(config["index_dir"])), indent=2))

