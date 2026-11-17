import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from topic_rag.config import load_config
from topic_rag.experiment import plan, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Default is offline review; --execute starts real inference")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/emanual.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.execute:
        print(run(config, skip_answer=args.skip_answer))
    else:
        print(json.dumps(plan(config), ensure_ascii=False, indent=2))

