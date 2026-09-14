"""Doc2Skill compilation, validation, retrieval and source-only evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config, read_jsonl, write_json, write_jsonl


def _parser():
    parser = argparse.ArgumentParser(prog="doc2skill")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", default="configs/doc2skill.toml")
    build.add_argument("--out-dir")
    build.add_argument("--resume", action="store_true")
    build.add_argument("--stop-after", choices=["parse", "index", "complete"], default="complete")
    adaptive = sub.add_parser('build-adaptive', help='Build a new corpus-only v2 navigation tree')
    adaptive.add_argument('--config', required=True)
    comparison_prepare = sub.add_parser('prepare-comparison', help='Freeze one corrected corpus for A/B/C')
    comparison_prepare.add_argument('--config', required=True)
    comparison_prepare.add_argument('--resume', action='store_true')
    comparison_build = sub.add_parser('build-comparison', help='Build one controlled navigation strategy')
    comparison_build.add_argument('--config', required=True)
    comparison_build.add_argument('--prepared-dir', required=True)
    comparison_build.add_argument('--strategy', required=True, choices=['source', 'llm', 'corpus2skill'])
    comparison_build.add_argument('--output-dir', required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--artifact-dir", required=True)
    prepare = sub.add_parser("prepare-structure", help="Export full corpus-only context and rule hints for offline LLM review")
    prepare.add_argument("--artifact-dir", required=True)
    prepare.add_argument("--output", required=True)
    refine = sub.add_parser("refine", help="Replay reviewed structure and navigation into a new version")
    refine.add_argument("--config", default="configs/doc2skill.toml")
    refine.add_argument("--artifact-dir", required=True)
    refine.add_argument("--review", required=True)
    refine.add_argument("--metadata-review")
    refine.add_argument("--out-dir", required=True)
    refine.add_argument("--stop-after", choices=["structure", "complete"], default="complete")
    for command in ("query", "evaluate"):
        item = sub.add_parser(command)
        item.add_argument("--config", default="configs/doc2skill.toml")
        item.add_argument("--artifact-dir", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--method", choices=["bm25", "dense", "navigation", "navigation-expand", "all"] if command == "evaluate" else ["bm25", "dense", "navigation", "navigation-expand"], default="navigation-expand")
        item.add_argument("--policy", help="Optional experimental Markdown policy override; corpus remains fixed")
        if command == "query":
            item.add_argument("question")
        else:
            item.add_argument("--qa", required=True)
            item.add_argument("--limit", type=int, default=0)
    return parser


def make_retriever(root, config, method, policy=None):
    from .validation import validate_bundle
    integrity = validate_bundle(root)
    if not integrity["valid"]:
        raise ValueError("Artifact integrity validation failed: " + "; ".join(integrity["errors"][:3]))
    if not (Path(root) / "manifest.json").exists():
        raise ValueError("A finalized artifact manifest is required before querying")
    if method.startswith("navigation") and integrity["build_status"] != "complete":
        raise ValueError("Navigation requires completed, genuine LLM metadata and MD skills")
    if method.startswith('navigation') and json.loads((Path(root) / 'skill_paths.json').read_text()).get('adaptive_tree'):
        raise ValueError('Adaptive v2 navigation uses qa-agent ask/batch; the legacy navigation-expand evaluator has different loop semantics')
    from .storage import Store
    store = Store(Path(root) / "corpus.sqlite")
    if method == "bm25":
        from evaluation.retrievers import BM25Retriever, Document
        chunks = store.records("chunks")
        by_id = {c["chunk_id"]: c for c in chunks}
        retriever = BM25Retriever([Document(c["chunk_id"], c["text"], c["doc_id"], c) for c in chunks])
        def run(question, qid):
            started = time.monotonic()
            found = retriever.search(question, k=int(config["runtime"].get("top_k", 20)))
            items = [dict(by_id[r.chunk_id], rank=r.rank, score=r.score) for r in found]
            return {"qid": qid, "status": "ok", "items": items, "elapsed_seconds": time.monotonic() - started}
        return store, run
    from .embedding import E5Encoder
    status = json.loads((Path(root) / "build_status.json").read_text())
    econfig = {**config["embedding"], "model": status["embedding"]["model"], "revision": status["embedding"]["revision"]}
    encoder = E5Encoder(econfig, local_only=True)
    if method == "dense":
        def run(question, qid):
            started = time.monotonic()
            items = store.search(encoder.encode_queries([question])[0], k=int(config["runtime"].get("top_k", 20)))
            return {"qid": qid, "status": "ok", "items": items, "elapsed_seconds": time.monotonic() - started}
        return store, run
    from .llm import ChatClient
    from .runtime import Navigator, validate_runtime_identity
    counter = validate_runtime_identity(config["runtime"])
    client = ChatClient({**config["runtime"], "token_counter": counter, "max_input_chars": 100000}, local_only=True)
    navigator = Navigator(store, encoder, client, root, config["runtime"], token_counter=counter,
                          skill_content=Path(policy).read_text(encoding="utf-8") if policy else None)
    return store, lambda question, qid: navigator.query(question, qid=qid, allow_expansion=method == "navigation-expand")


def export_run(result, alignment=None, *, field="items"):
    from .alignment import apply_alignment
    items = result.get(field, []) if result.get("status", "ok") == "ok" else []
    if alignment is not None:
        items = apply_alignment(items, alignment)
    return {"qid": result["qid"], "status": result.get("status", "ok"),
            "items": [dict(item, rank=i + 1) for i, item in enumerate(items)],
            "elapsed_seconds": result.get("elapsed_seconds", 0)}


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command in ('prepare-comparison', 'build-comparison'):
        from .comparison_pipeline import prepare_comparison, build_comparison
        config = load_config(args.config)
        try:
            result = (prepare_comparison(config, resume=args.resume) if args.command == 'prepare-comparison'
                      else build_comparison(config, args.prepared_dir, args.strategy, args.output_dir))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc)}}, ensure_ascii=False), file=sys.stderr)
            return 1
    try:
        if args.command == "prepare-structure":
            from .normalization import prepare_structure
            from .pipeline import load_corpus
            from .validation import validate_bundle
            if Path(args.output).exists():
                raise ValueError("Review packet output already exists")
            integrity = validate_bundle(args.artifact_dir)
            if not integrity["valid"]:
                raise ValueError("Input corpus failed integrity checks")
            if Path(args.output).resolve().is_relative_to(Path(args.artifact_dir).resolve()):
                raise ValueError("Write review packets outside the immutable input bundle")
            packet = prepare_structure(load_corpus(args.artifact_dir))
            write_json(args.output, packet)
            print(json.dumps({"output": args.output, "input_fingerprint": packet["input_fingerprint"]}))
            return 0
        if args.command == "validate":
            from .validation import validate_bundle
            report = validate_bundle(args.artifact_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 2
        config = load_config(args.config)
        if args.command == 'build-adaptive':
            from .adaptive_pipeline import build_adaptive
            result = build_adaptive(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'] == 'complete' else 3
        if args.command == "refine":
            from .refinement import refine_corpus
            status = refine_corpus(config, args.artifact_dir, args.review, args.out_dir,
                                   metadata_review=args.metadata_review, stop_after=args.stop_after)
            print(json.dumps({k: status[k] for k in ("status", "completed_stages", "source_hashes_unchanged", "input_bundle_unchanged")}, indent=2))
            return 0
        if args.command == "build":
            from .pipeline import build_corpus
            status = build_corpus(config, out_dir=args.out_dir, stop_after=args.stop_after, resume=args.resume,
                                  progress=lambda s: print(s, file=sys.stderr, flush=True))
            print(json.dumps({key: status[key] for key in ("status", "completed_stages", "embedding", "source_hashes_unchanged") if key in status}, ensure_ascii=False, indent=2))
            return 3 if status["status"] == "paused_resource_busy" else 2 if status["status"] == "blocked" else 0
        root = Path(args.artifact_dir)
        if args.command == "query":
            if args.method.startswith("navigation") and json.loads((root / "skill_paths.json").read_text()).get("hierarchy"):
                raise ValueError("Use qa-agent for hierarchical dialogue navigation; the legacy doc2skill query router is two-level only")
            if Path(args.output).exists():
                raise ValueError("Query output already exists; choose a new file")
            store, run = make_retriever(root, config, args.method, args.policy)
            try:
                result = run(args.question, "interactive")
            finally:
                store.close()
            write_json(args.output, result)
            print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))
            return 0 if result["status"] == "ok" else 2
        from .evaluation import evaluate_runs
        from .runtime import bounded_excerpt
        qa = read_jsonl(args.qa)
        if args.limit:
            qa = qa[:args.limit]
        alignment_path = root / "alignment.json"
        if not alignment_path.exists():
            raise ValueError("Benchmark evaluation requires an audited legacy source alignment")
        alignment = json.loads(alignment_path.read_text())
        from .embedding import E5Encoder
        built = json.loads((root / "build_status.json").read_text())
        budget_encoder = E5Encoder({**config["embedding"], **built["embedding"]}, local_only=True, tokenizer_only=True)
        budget_counter = lambda text: len(budget_encoder.tokenizer.encode(text, add_special_tokens=True))
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose a new empty evaluation output directory")
        output.mkdir(parents=True, exist_ok=True)
        methods = ["bm25", "dense", "navigation", "navigation-expand"] if args.method == "all" else [args.method]
        summary = {"status": "prototype_evaluation", "benchmark_frozen": False, "coverage_threshold": 1.0,
                   "runtime_config": config["runtime"], "methods": {}}
        for method in methods:
            traces, store = [], None
            try:
                store, run = make_retriever(root, config, method, args.policy)
                for row in qa:
                    try:
                        # Only question and ID reach the executor, never answers or routes.
                        trace = run(row["question"], row["qid"])
                    except Exception as exc:
                        trace = {"qid": row["qid"], "items": [], "status": "failed", "error_type": type(exc).__name__}
                    traces.append(trace)
            except Exception as exc:
                traces = [{"qid": row["qid"], "items": [], "status": "setup_failed", "error": str(exc)} for row in qa]
            finally:
                if store:
                    store.close()
            write_jsonl(output / f"{method}.traces.jsonl", traces)
            runs = [export_run(trace, alignment) for trace in traces]
            write_jsonl(output / f"{method}.run.jsonl", runs)
            # A subset evaluation must not silently count unexecuted questions as failures.
            subset_path = output / "evaluation_queries.jsonl"
            write_jsonl(subset_path, qa)
            original_split = Path(args.qa).parent / "split_diagnostics.json"
            if original_split.exists():
                write_json(output / "split_diagnostics.json", json.loads(original_split.read_text()))
            report = evaluate_runs(subset_path, runs, alignment_report=alignment["report"])
            report["execution"] = {"total": len(traces), "failed": sum(t["status"] != "ok" for t in traces),
                                   "expanded": sum(t.get("expanded", False) for t in traces)}
            budget_runs = []
            for trace in traces:
                packed = dict(trace, context_items=bounded_excerpt(trace.get("items", []), config["runtime"].get("reading_tokens", 4000), budget_counter))
                budget_runs.append(export_run(packed, alignment, field="context_items"))
            report["fixed_context_budget"] = {"tokens": config["runtime"].get("reading_tokens", 4000),
                                              "tokenizer": built["embedding"],
                                              "meaning": "retrieved context packed for a reader; not a claim the LLM saw it",
                                              "metrics": evaluate_runs(subset_path, budget_runs, alignment_report=alignment["report"])}
            import numpy as np
            durations = [t["elapsed_seconds"] for t in traces if "elapsed_seconds" in t]
            report["execution"].update(p50_seconds=float(np.percentile(durations, 50)) if durations else None,
                                       p95_seconds=float(np.percentile(durations, 95)) if durations else None)
            if method.startswith("navigation"):
                seen_runs = [export_run(t, alignment, field="seen_items") for t in traces]
                report["actually_read_evidence"] = evaluate_runs(subset_path, seen_runs, alignment_report=alignment["report"])
                before_runs = [export_run(t, alignment, field="scoped_items") for t in traces]
                report["before_expansion"] = evaluate_runs(subset_path, before_runs, alignment_report=alignment["report"])
            write_json(output / f"{method}.metrics.json", report)
            setup_blocked = bool(traces) and all(t["status"] == "setup_failed" for t in traces)
            summary["methods"][method] = {"failed": report["execution"]["failed"], "report": f"{method}.metrics.json",
                                          "status": "not_run_setup_blocked" if setup_blocked else "executed",
                                          "comparable": not setup_blocked}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if any(v["failed"] for v in summary["methods"].values()) else 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
