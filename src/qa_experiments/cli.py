"""CLI: private preparation, doctor, native index build, smoke/freeze/run, report."""
import argparse
import json

from . import METHODS

def main():
    failed = False
    parser = argparse.ArgumentParser(prog="qa-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("project", "output", "online-config", "prepared"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--arag-max-loops", type=int, default=15,
                         help="Sealed in preparation identity; use a new output for another budget")
    for name in ("doctor", "build-index", "freeze", "run", "summarize"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", required=True)
        if name in ("run", "build-index"):
            sub.add_argument("--method", choices=(*METHODS, "all"), required=True)
        if name in ("run", "summarize"):
            sub.add_argument("--phase", choices=("smoke", "test"), default="test")
        if name == "run":
            sub.add_argument("--arag-max-loops", type=int, default=None)
        if name == "freeze":
            sub.add_argument("--methods", nargs="+", choices=METHODS)
    judge = commands.add_parser("judge")
    judge.add_argument("--root", required=True, help="Complete frozen inference experiment")
    judge.add_argument("--prepared", required=True, help="Sealed test dataset")
    judge.add_argument("--output", required=True, help="New or exactly resumable private output directory")
    judge.add_argument("--usage-window-confirmed", action="store_true")
    judge_summary = commands.add_parser("judge-summary")
    judge_summary.add_argument("--root", required=True)
    judge_summary.add_argument("--prepared", required=True)
    judge_summary.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        result = prepare(args.project, args.output, args.online_config, args.prepared,
                         arag_max_loops=args.arag_max_loops)
    elif args.command in ("judge", "judge-summary"):
        from qa_experiment_judge.evaluator import run_judge, summarize_judgments
        if args.command == "judge":
            result = run_judge(args.root, args.prepared, args.output,
                               usage_window_confirmed=args.usage_window_confirmed)
        else:
            result = summarize_judgments(args.root, args.prepared, args.output)
    else:
        from . import runner
        if args.command == "doctor":
            result = runner.doctor(args.root)
        elif args.command == "build-index":
            methods = ("arag", "linear") if args.method == "all" else (args.method,)
            result = {}
            for method in methods:
                try:
                    result[method] = runner.build_index(args.root, method)
                except Exception as exc:
                    failed = True
                    result[method] = {"status": "blocked", "error": str(exc)}
        elif args.command == "freeze":
            result = runner.freeze(args.root, methods=args.methods)
        elif args.command == "run":
            if args.method == "all" and args.phase == "test":
                from pathlib import Path
                methods = runner.read(Path(args.root) / "frozen.json")["active_methods"]
            else:
                methods = METHODS if args.method == "all" else (args.method,)
            result = {}
            for method in methods:
                try:
                    extra = {"arag_max_loops": args.arag_max_loops} if method == "arag" else {}
                    result[method] = runner.run_method(args.root, method, args.phase, **extra)
                except Exception as exc:
                    failed = True
                    result[method] = {"status": "blocked" if args.phase == "smoke" else "interrupted", "error": str(exc)}
        else:
            result = runner.summarize(args.root, args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
