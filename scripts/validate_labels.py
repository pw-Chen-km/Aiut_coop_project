import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from topic_rag.config import load_config
from topic_rag.validation import validate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data, source spans and labels without inference")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/emanual.yaml")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(load_config(args.config)), ensure_ascii=False, indent=2))

