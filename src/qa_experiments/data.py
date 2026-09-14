"""Gold-stripping preparation and immutable inference-only manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from doc2skill.config import sha256, fingerprint, read_jsonl, write_json, write_jsonl
from qa_agent.bundle import validate_serving_bundle
from qa_agent.factory import load_online_config
from doc2skill.storage import Store
from . import METHODS, UPSTREAM


def read(path):
    return json.loads(Path(path).read_text())

def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not any(x in p.parts for x in (".git", "__pycache__", ".cache"))}

def installed_packages(python):
    script = "import importlib.metadata as m,json,platform; print(json.dumps({'python':platform.python_version(),'packages':dict(sorted((d.metadata['Name'].lower(),d.version) for d in m.distributions()))},sort_keys=True))"
    return json.loads(subprocess.check_output([str(python), "-c", script], text=True, timeout=30))

def safe_questions(rows):
    output = []
    for row in rows:
        if not isinstance(row.get("qid"), str) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Invalid question")
        output.append({"qid": row["qid"], "question": row["question"],
                       "question_type": row.get("question_type"),
                       "visual_diagnostic": row.get("visual_diagnostic", False),
                       "persona": row.get("persona")})
    if len({r["qid"] for r in output}) != len(output):
        raise ValueError("Duplicate qids")
    return output

def prepare(project, output, online_config, prepared, *, arag_max_loops=15):
    if type(arag_max_loops) is not int or not 1 <= arag_max_loops <= 15:
        raise ValueError("A-RAG max_loops must be an integer in [1, 15]")
    project, output, prepared = map(lambda p: Path(p).resolve(), (project, output, prepared))
    config = load_online_config(online_config)
    bundle = Path(config["bundle_dir"])
    for protected in (bundle, prepared, project / "data"):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output overlaps immutable inputs")
    if output.exists():
        raise FileExistsError("Choose a new preparation directory")
    bundle_info = validate_serving_bundle(bundle)
    if bundle_info["counts"] != {"documents": 5, "sections": 107, "chunks": 104}:
        raise ValueError("Unexpected corpus snapshot")
    phases = {}
    for phase, expected in (("test", 120), ("optimization", 40)):
        info = read(prepared / (phase + "_manifest.json"))
        source = prepared / (phase + ".jsonl")
        if sha256(source) != info["sha256"]:
            raise ValueError("Dataset phase checksum changed")
        rows = read_jsonl(source)
        # Eligible optimization excludes the pre-existing ambiguous gold record.
        if phase == "test" and (len(rows) != expected or {r["qid"] for r in rows} != set(info["qids"])):
            raise ValueError("Test must exactly match its 120-qid manifest")
        phases[phase] = safe_questions(rows)
    eligible = [r for r in phases["optimization"] if not r["qid"].endswith("q000101")]
    types = sorted({r["question_type"] for r in eligible})
    if len(types) != 3:
        raise ValueError("Expected three question types")
    smoke = [min((r for r in eligible if r["question_type"] == typ), key=lambda r: r["qid"]) for typ in types]
    output.mkdir(parents=True)
    shutil.copytree(bundle, output / "bundle")
    config["bundle_dir"] = str(output / "bundle")
    write_json(output / "online.json", config)
    write_jsonl(output / "test.questions.jsonl", phases["test"])
    write_jsonl(output / "smoke.questions.jsonl", smoke)
    with Store(bundle / "corpus.sqlite") as store:
        originals = store.records("chunks")
        chunks = []
        # Numeric IDs preserve upstream syntax; gaps prevent +/-1 crossing PDFs.
        last_doc, alias = None, -1
        for row in originals:
            alias += 100 if last_doc is not None and row["doc_id"] != last_doc else 1
            last_doc = row["doc_id"]
            chunks.append({"id": str(alias), "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
                           "section_id": row["section_id"], "text": row["text"],
                           "source_spans": row.get("source_spans", [])})
    write_json(output / "chunks.json", chunks)
    write_json(output / "linear.chunks.json", [c["text"] for c in chunks])
    environments = {}
    for method in ("arag", "linear"):
        repo = project / "private/vendor" / method
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM[method]:
            raise ValueError("Wrong upstream revision")
        model_path = project / "models/baselines" / method
        if not (model_path / "config.json").exists() or not list(model_path.glob("*.safetensors")):
            raise ValueError("Incomplete public model download: " + method)
        python = project / "private/experiments/envs" / method / "bin/python"
        environments[method] = {"python": str(python), "installed_packages": installed_packages(python),
            "repo": str(repo), "model": str(project / "models/baselines" / method),
            "source_hashes": tree_hashes(repo), "model_hashes": tree_hashes(project / "models/baselines" / method),
            "environment": read(project / "private/experiments" / (method + "-environment-ready.json"))}
    manifest = {"schema_version": "qa-experiment-preparation-v1", "methods": METHODS,
        "project": str(project), "upstream": UPSTREAM, "environments": environments,
        "runtime_packages": installed_packages(project / ".venv/bin/python"),
        "protected": {str(p): sha256(p) for p in sorted((project / "data").rglob("*")) if p.is_file()},
        "original_bundle": str(bundle), "original_bundle_hashes": tree_hashes(bundle),
        "split_manifest_sha256": sha256(prepared / "test_manifest.json"),
        "artifacts": tree_hashes(output), "judge_requested": False,
        "settings": {"linear": {"max_iterations": 3, "iteration_threshold": 0.4, "passage_ratio": 2,
                     "top_k_sentence": 3, "retrieval_top_k": 5}, "arag": {"max_loops": arag_max_loops},
                     "seed": 42, "embedding_device": "cpu", "workers": 1},
        "evaluation_label": "prototype_shared_gpu_4b_8k_unscored", "skillopt_optimized": False}
    manifest["preparation_id"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return {"directory": str(output), "test_count": 120, "smoke_qids": [r["qid"] for r in smoke],
            "preparation_id": manifest["preparation_id"]}

def verify_preparation(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    identity = dict(manifest)
    claimed = identity.pop("preparation_id")
    if fingerprint(identity) != claimed:
        raise ValueError("Preparation manifest changed")
    for name, expected in manifest["artifacts"].items():
        if sha256(root / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    return manifest

def verify_originals(manifest):
    for name, expected in manifest["protected"].items():
        if sha256(name) != expected:
            raise ValueError("Original dataset changed: " + name)
    if tree_hashes(manifest["original_bundle"]) != manifest["original_bundle_hashes"]:
        raise ValueError("Original serving bundle changed")
