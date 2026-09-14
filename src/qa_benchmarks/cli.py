from __future__ import annotations

import argparse
import json
from pathlib import Path

from .multidoc2dial import export_evaluation, prepare, validate_preparation


def main(argv=None):
    p = argparse.ArgumentParser(prog="qa-benchmarks")
    sub = p.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("prepare")
    cmd.add_argument("--dataset", choices=["multidoc2dial"], default="multidoc2dial")
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--archive", type=Path)
    cmd = sub.add_parser("validate")
    cmd.add_argument("--root", required=True, type=Path)
    cmd = sub.add_parser("export-eval")
    cmd.add_argument("--root", required=True, type=Path)
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--split", choices=["train", "validation"], default="validation")
    cmd.add_argument("--comparison", action="store_true")
    cmd = sub.add_parser('export-gt', help='Map official evidence to a specified corpus; no model calls')
    cmd.add_argument('--root', required=True, type=Path)
    cmd.add_argument('--corpus-dir', required=True, type=Path)
    cmd.add_argument('--output', required=True, type=Path)
    cmd.add_argument('--split', choices=['train', 'validation'], default='train')
    cmd.add_argument('--view', choices=['all', 'opening-answer'], default='opening-answer')
    cmd = sub.add_parser('score-evidence', help='Score existing QA logs against a fixed GT export')
    cmd.add_argument('--gt-dir', required=True, type=Path)
    cmd.add_argument('--corpus-dir', required=True, type=Path)
    cmd.add_argument('--trajectory', required=True, type=Path)
    cmd.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv)
    if a.command == "prepare":
        result = prepare(a.output, a.archive)
    elif a.command == "validate":
        result = validate_preparation(a.root)
    elif a.command == 'export-gt':
        from .multidoc_gt import export_gt
        result = export_gt(a.root, a.corpus_dir, a.output, split=a.split, view=a.view)
    elif a.command == 'score-evidence':
        from .multidoc_gt import score_traces
        result = score_traces(a.gt_dir, a.corpus_dir, a.trajectory, a.output)
    else:
        result = export_evaluation(a.root, a.output, a.split, a.comparison)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, ensure_ascii=False, indent=2))
    return 0
