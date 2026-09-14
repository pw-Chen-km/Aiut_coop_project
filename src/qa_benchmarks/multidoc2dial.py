"""Verbatim native MultiDoc2Dial import; no model calls or synthesized QA."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from doc2skill.config import fingerprint, read_jsonl, sha256, write_json, write_jsonl
from doc2skill.pipeline import COLLECTIONS
from doc2skill.structure import stable_id

ARCHIVE_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
ARCHIVE_SHA256 = "f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00"
DOMAIN_COUNTS = {"ssa": 109, "va": 138, "dmv": 149, "studentaid": 92}


def import_documents(doc_data: dict) -> tuple[dict, list[dict]]:
    corpus = {k: [] for k in COLLECTIONS}
    original_spans = []
    for domain, documents in doc_data.items():
        for native_id, doc in documents.items():
            text = doc["doc_text"]
            did = stable_id("doc", "multidoc2dial-v1", domain, native_id)
            digest = hashlib.sha256(text.encode()).hexdigest()
            corpus["documents"].append({"doc_id": did, "native_doc_id": native_id, "domain": domain,
                "title": doc["title"], "source_type": "html", "source_filename": native_id,
                "source_sha256": digest, "document_version": "multidoc2dial-v1.0", "page_count": 0,
                "doc_text": text, "doc_html_raw": doc.get("doc_html_raw", ""),
                "doc_html_ts": doc.get("doc_html_ts", "")})
            spans = list(doc["spans"].values())
            for span in spans:
                a, b = span["start_sp"], span["end_sp"]
                if not 0 <= a < b <= len(text):
                    raise ValueError(f"Invalid native offsets: {native_id}/{span['id_sp']}")
                original_spans.append({"doc_id": did, "native_doc_id": native_id, "domain": domain,
                    **span, "canonical_text": text[a:b], "annotation_text_matches": text[a:b] == span["text_sp"]})
            heads = sorted((s for s in spans if s["tag"] in {"h1", "h2", "h3", "h4", "h5", "h6"}),
                           key=lambda s: (s["start_sp"], s["end_sp"]))
            # A repeated annotation at the same range is one native heading, not two chapters.
            heads = list({(s["start_sp"], s["end_sp"]): s for s in heads}.values())
            root_id = stable_id("sec", did, "root")
            sections = [{"section_id": root_id, "doc_id": did, "title": doc["title"], "parent_id": None,
                         "path": [doc["title"]], "level": 0, "kind": "document_root", "block_ids": [],
                         "page_start": None, "page_end": None, "source_start_char": 0}]
            stack, heading_section = [], {}
            for head in heads:
                level = int(head["tag"][1:])
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = stack[-1][1] if stack else sections[0]
                title = text[head["start_sp"]:head["end_sp"]].strip()
                if not title:
                    continue
                node = {"section_id": stable_id("sec", did, head["start_sp"], head["end_sp"]),
                    "doc_id": did, "title": title, "parent_id": parent["section_id"],
                    "path": parent["path"] + [title], "level": parent["level"] + 1,
                    "kind": "native_heading", "block_ids": [], "page_start": None, "page_end": None,
                    "source_start_char": head["start_sp"], "native_parent_titles": head.get("parent_titles", [])}
                sections.append(node)
                stack.append((level, node))
                heading_section[head["start_sp"]] = node
            # Paragraph boundaries are blocks only. Intersecting annotations never duplicate text.
            boundaries = {0, len(text)}
            for span in spans:
                for field in ("start_sec", "end_sec"):
                    pos = span.get(field)
                    if isinstance(pos, int) and 0 <= pos <= len(text):
                        boundaries.add(pos)
            for h in heads:
                boundaries.update((h["start_sp"], h["end_sp"]))
            ranges = sorted(boundaries)
            active, head_index = sections[0], 0
            for a, b in zip(ranges, ranges[1:]):
                while head_index < len(heads) and heads[head_index]["start_sp"] <= a:
                    active = heading_section.get(heads[head_index]["start_sp"], active)
                    head_index += 1
                is_heading = any(h["start_sp"] <= a and b <= h["end_sp"] for h in heads)
                bid = stable_id("blk", did, digest, a, b)
                block = {"block_id": bid, "doc_id": did, "section_id": active["section_id"],
                    "text": text[a:b], "kind": "heading" if is_heading else "text", "source_type": "html",
                    "page_index": None, "bbox": None, "doc_start_char": a, "doc_end_char": b,
                    "provenance": [{"source_type": "html", "native_doc_id": native_id,
                                    "start_char": a, "end_char": b}]}
                corpus["blocks"].append(block)
                active["block_ids"].append(bid)
            corpus["sections"].extend(sections)
    return corpus, original_spans


def dialogue_cases(dial_data: dict, split: str):
    inputs, references = [], []
    for domain, dialogs in dial_data.items():
        for dialog in dialogs:
            turns = dialog["turns"]
            for i, (user, target) in enumerate(zip(turns, turns[1:])):
                if user["role"] != "user" or target["role"] != "agent":
                    continue
                qid = f"mdd:{split}:{domain}:{dialog['dial_id']}:{user['turn_id']}"
                history = [{"role": t["role"], "content": t["utterance"], "turn_id": t["turn_id"]}
                           for t in turns[:i]]
                inputs.append({"qid": qid, "question": user["utterance"], "history": history})
                act = target["da"]
                refs = target.get("references", [])
                references.append({"qid": qid, "split": split, "domain": domain,
                    "dialogue_id": dialog["dial_id"], "user_turn_id": user["turn_id"],
                    "target_turn_id": target["turn_id"], "reference_answer": target["utterance"],
                    "dialogue_act": act, "response_type": "clarify" if act == "query_condition" else
                    "abstain" if act == "respond_no_solution" else "answer",
                    "references": refs, "comparison_view": act != "respond_no_solution"})
    return inputs, references


def prepare(output: Path, archive: Path | None = None, *, enforce_counts=True) -> dict:
    import io
    import urllib.request
    import zipfile
    if output.exists():
        raise FileExistsError("Choose a new benchmark preparation directory")
    blob = archive.read_bytes() if archive else urllib.request.urlopen(ARCHIVE_URL, timeout=60).read()
    digest = hashlib.sha256(blob).hexdigest()
    if enforce_counts and digest != ARCHIVE_SHA256:
        raise ValueError("Official archive changed; review its version before preparing")
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = ["README.md", "multidoc2dial_doc.json", *[f"multidoc2dial_dial_{s}.json" for s in ("train", "validation", "test")]]
    raw = {name: z.read("multidoc2dial/" + name) for name in names}
    document_data = json.loads(raw["multidoc2dial_doc.json"])["doc_data"]
    if enforce_counts and {k: len(v) for k, v in document_data.items()} != DOMAIN_COUNTS:
        raise ValueError("Unexpected document inventory")
    corpus, native_spans = import_documents(document_data)
    output.mkdir(parents=True)
    (output / "raw").mkdir()
    (output / "raw" / "multidoc2dial.zip").write_bytes(blob)
    for name, value in raw.items():
        (output / "raw" / name).write_bytes(value)
    for kind, rows in corpus.items():
        write_jsonl(output / "corpus" / (kind + ".jsonl"), rows)
    write_jsonl(output / "corpus" / "native_spans.jsonl", native_spans)
    summary = {}
    for split in ("train", "validation"):
        data = json.loads(raw[f"multidoc2dial_dial_{split}.json"])["dial_data"]
        inputs, refs = dialogue_cases(data, split)
        summary[split] = {"dialogues": sum(map(len, data.values())), "cases": len(inputs),
            "comparison_cases": sum(r["comparison_view"] for r in refs),
            "response_types": dict(Counter(r["response_type"] for r in refs)),
            "domains": {k: len(v) for k, v in data.items()}}
        write_jsonl(output / "evaluation" / f"{split}.inputs.jsonl", inputs)
        write_jsonl(output / "evaluation" / f"{split}.references.jsonl", refs)
        write_json(output / "evaluation" / f"{split}.comparison.json",
                   [r["qid"] for r in refs if r["comparison_view"]])
    if enforce_counts and (summary["train"]["dialogues"], summary["validation"]["dialogues"],
        summary["validation"]["cases"], summary["validation"]["comparison_cases"]) != (3474, 661, 4427, 4201):
        raise ValueError("Unexpected dialogue counts; incomplete preparation must not be used")
    manifest = {"schema": "multidoc2dial-preparation-v1", "source_url": ARCHIVE_URL, "archive_sha256": digest,
        "documents": {k: len(v) for k, v in document_data.items()}, "splits": summary,
        "test_status": "preserved_not_evaluation_ambiguous_release_description",
        "citation": "Feng et al. 2021. MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents. EMNLP.",
        "artifacts": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_preparation(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, digest in manifest["artifacts"].items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha256(p) != digest:
            raise ValueError("Preparation checksum mismatch: " + relative)
    inputs = read_jsonl(root / "evaluation" / "validation.inputs.jsonl")
    for row in inputs:
        if set(row) != {"qid", "question", "history"}:
            raise ValueError("Unexpected evaluation input fields")
    return {"valid": True, "documents": manifest["documents"], "splits": manifest["splits"]}


def export_evaluation(root: Path, output: Path, split="validation", comparison=False):
    validate_preparation(root)
    if split not in {"train", "validation"}:
        raise ValueError("Unverified test release is not an evaluation split")
    if output.exists():
        raise FileExistsError(output)
    refs = read_jsonl(root / "evaluation" / f"{split}.references.jsonl")
    selected = {r["qid"] for r in refs if not comparison or r["comparison_view"]}
    write_jsonl(output / "inputs.jsonl", [r for r in read_jsonl(root / "evaluation" / f"{split}.inputs.jsonl") if r["qid"] in selected])
    write_jsonl(output / "references.jsonl", [r for r in refs if r["qid"] in selected])
    return {"split": split, "cases": len(selected), "reference_field": "reference_answer", "scored": False}
