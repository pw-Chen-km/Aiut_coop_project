"""Inspect interchangeable navigation builds and optionally run the same QA set.

Reporting never builds a missing arm or calls a model unless QA is requested.
The shared corpus is checked from file bytes, not trusted manifest labels.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics

from doc2skill.adaptive import validate_tree
from doc2skill.config import fingerprint, load_config, read_jsonl, sha256, write_json, write_jsonl
from doc2skill.comparison_pipeline import COMMON_FILES
from .bundle import local_asset


ARMS = ("source", "llm", "corpus2skill")
COMMON_REQUIRED = COMMON_FILES


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_arm(root, strategy):
    root = Path(root).resolve()
    status = _read_json(root / "build_status.json")
    if status.get("status") != "complete":
        raise ValueError("Build is not complete: " + str(status.get("status", "unknown")))
    if status.get("strategy") != strategy:
        raise ValueError("Build strategy differs from the requested arm")
    prepared = _read_json(root / "prepared_manifest.json")
    if prepared.get("schema_version") != "comparison-preparation-v1" or prepared.get("status") != "complete":
        raise ValueError("Expected a completed comparison preparation")
    if not isinstance(prepared.get("llm_configuration"), dict) or not prepared["llm_configuration"]:
        raise ValueError("Preparation must record its description model configuration")
    if status.get("llm_configuration") != prepared["llm_configuration"]:
        raise ValueError("Builder model configuration differs from shared preparation")
    if not isinstance(prepared.get("descriptor_policy"), dict) or not prepared["descriptor_policy"]:
        raise ValueError("Preparation must record its common descriptor policy")
    hashes = prepared.get("common_hashes")
    if not isinstance(hashes, dict) or not COMMON_REQUIRED <= hashes.keys():
        raise ValueError("Preparation does not cover every shared corpus artifact")
    if not isinstance(prepared.get("fingerprint"), str) or not re.fullmatch(r"[0-9a-f]{64}", prepared["fingerprint"]):
        raise ValueError("Invalid preparation fingerprint")
    for relative, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid common artifact checksum")
        if sha256(local_asset(root, relative)) != digest:
            raise ValueError("Shared corpus artifact changed: " + relative)
    if prepared["fingerprint"] != fingerprint(hashes):
        raise ValueError("Preparation fingerprint does not match the common artifact checksums")
    paths = _read_json(root / "skill_paths.json")
    tree = validate_tree(_read_json(local_asset(root, paths["adaptive_tree"])))
    sections = {s["section_id"]: s for s in read_jsonl(root / "sections.jsonl")}
    if tree["section_documents"] != {sid: s["doc_id"] for sid, s in sections.items()}:
        raise ValueError("Navigation registry differs from the shared corpus")
    if paths.get("node_skills") != {nid: n["md_path"] for nid, n in tree["nodes"].items()}:
        raise ValueError("Node Markdown paths differ from the navigation registry")
    for node in tree["nodes"].values():
        if not isinstance(node.get("title"), str) or not node["title"].strip() or not node["md_path"].endswith(".md"):
            raise ValueError("Navigation nodes require a title and Markdown path")
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict) or not isinstance(metadata.get("summary"), str):
            raise ValueError("Navigation nodes require summary metadata")
        for field in ("content_type", "question_intents", "aliases"):
            if not isinstance(metadata.get(field), list) or any(not isinstance(v, str) for v in metadata[field]):
                raise ValueError("Invalid node metadata field: " + field)
    md = {nid: local_asset(root, node["md_path"]).read_text(encoding="utf-8")
          for nid, node in tree["nodes"].items()}
    if any(not text.strip() for text in md.values()):
        raise ValueError("A navigation node has empty Markdown")
    units = read_jsonl(root / "units.jsonl")
    if (not units or any(not isinstance(u.get("unit_id"), str) or not u["unit_id"] for u in units)
            or len({u["unit_id"] for u in units}) != len(units)):
        raise ValueError("Units must have unique, nonempty identities")
    for unit in units:
        if unit["section_id"] not in sections or unit["doc_id"] != sections[unit["section_id"]]["doc_id"]:
            raise ValueError("Unit is outside its canonical source section")
    documents = {d["doc_id"]: d for d in read_jsonl(root / "documents.jsonl")}
    if any(s["doc_id"] not in documents for s in sections.values()):
        raise ValueError("Canonical section refers to an unknown document")
    return {"root": root, "status": status, "prepared": prepared, "tree": tree,
            "sections": sections, "documents": documents, "units": units, "markdown": md}


def _paths(tree):
    paths = {}
    def visit(nid, chain):
        paths[nid] = [*chain, nid]
        for child in tree["nodes"][nid]["children"]:
            visit(child, paths[nid])
    visit(tree["root_id"], [])
    return paths


def _metrics(arm):
    tree = arm["tree"]
    nodes, paths = tree["nodes"], _paths(tree)
    leaves = [nid for nid, n in nodes.items() if not n["children"]]
    groups = [n for n in nodes.values() if n.get("kind") == "group"]
    cross = sum(len(n["scope"]["doc_ids"]) > 1 for n in groups)
    levels = Counter(len(p) - 1 for p in paths.values())
    return {"nodes": len(nodes), "leaves": len(leaves), "max_depth": max(levels),
            "mean_leaf_depth": statistics.mean(len(paths[n]) - 1 for n in leaves),
            "max_children": max(len(n["children"]) for n in nodes.values()),
            "secondary_edges": sum(len(n.get("secondary_children", [])) for n in nodes.values()),
            "max_navigation_children": max(len(n["children"]) + len(n.get("secondary_children", [])) for n in nodes.values()),
            "max_level_width": max(levels.values()), "nodes_per_depth": dict(sorted(levels.items())),
            "group_nodes": len(groups), "cross_document_groups": cross,
            "cross_document_group_ratio": cross / len(groups) if groups else None,
            "sections": len(tree["section_documents"]), "units": len(arm["units"]),
            "markdown_characters": sum(len(s) for s in arm["markdown"].values()),
            "build_elapsed_seconds": arm["status"].get("elapsed_seconds"),
            "llm_cost": deepcopy(arm["status"].get("llm_cost", {}))}


def _page_label(unit, section):
    if unit.get("pages"):
        return ", ".join(str(p) for p in unit["pages"])
    start, end = section.get("page_start"), section.get("page_end")
    if isinstance(start, int):
        return str(start + 1) if end is None or end == start else f"{start + 1}–{end + 1}"
    return "unknown"


def _mapping_rows(strategy, arm):
    tree, paths = arm["tree"], _paths(arm["tree"])
    owners = {sid: nid for nid, n in tree["nodes"].items() for sid in n["own_section_ids"]}
    rows = []
    for unit in arm["units"]:
        sid, did = unit["section_id"], unit["doc_id"]
        nid = owners[sid]
        rows.append({"strategy": strategy, "unit_id": unit["unit_id"], "section_id": sid, "doc_id": did,
                     "document_title": arm["documents"].get(did, {}).get("title", did),
                     "unit_title": unit.get("title", sid), "pages": _page_label(unit, arm["sections"][sid]),
                     "owner_node_id": nid, "path_node_ids": json.dumps(paths[nid]),
                     "secondary_entrances": json.dumps([
                         {'from_node_id': source, 'to_node_id': target}
                         for source, node in tree['nodes'].items()
                         for target in node.get('secondary_children', [])
                         if sid in tree['nodes'][target]['scope']['section_ids']]),
                     "path_titles": " → ".join(tree["nodes"][n]["title"] for n in paths[nid])})
    return rows


def _esc(value):
    return html.escape(str(value), quote=True)


def _csv_value(value):
    # CSV may be opened in Excel. PDF titles must stay text, never formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _render_tree(arm, nid):
    tree, nodes = arm["tree"], arm["tree"]["nodes"]
    node = nodes[nid]
    owned = set(node["own_section_ids"])
    members = [u for u in arm["units"] if u["section_id"] in owned]
    content = [f'<details class="node" {"open" if nid == tree["root_id"] else ""}>',
               f'<summary>{_esc(node["title"])} <small>{len(node["scope"]["section_ids"])} sections · '
               f'{len(node["scope"]["doc_ids"])} documents</small></summary>',
               f'<p class="node-id">{_esc(nid)} · {_esc(node.get("kind", "node"))}</p>',
               f'<p>{_esc(node.get("metadata", {}).get("summary", ""))}</p>']
    for field in ("content_type", "question_intents", "aliases"):
        values = node.get("metadata", {}).get(field, [])
        if values:
            content.append(f'<p><b>{_esc(field)}:</b> {_esc("; ".join(values))}</p>')
    if members:
        content.append('<ul class="members">')
        for unit in members:
            did, sid = unit["doc_id"], unit["section_id"]
            title = arm["documents"].get(did, {}).get("title", did)
            page = _page_label(unit, arm["sections"][sid])
            content.append(f'<li><b>{_esc(unit.get("title", sid))}</b><br>{_esc(title)} · p. {_esc(page)}'
                           f'<br><small>{_esc(unit["unit_id"])} · {_esc(sid)}</small></li>')
        content.append('</ul>')
    content.append(f'<details class="markdown"><summary>Actual navigation Markdown</summary>'
                   f'<pre>{_esc(arm["markdown"][nid])}</pre></details>')
    for child in node["children"]:
        content.append(_render_tree(arm, child))
    for child in node.get("secondary_children", []):
        content.append(f'<p>Additional entry: {_esc(arm["tree"]["nodes"][child]["title"])} '
                       f'<small>{_esc(child)}</small></p>')
    content.append('</details>')
    return "\n".join(content)


def _html(report, loaded):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Navigation structure comparison</title><style>',
             'body{font:15px/1.5 system-ui,sans-serif;margin:24px;color:#172333;background:#f4f6fa}'
             'h1{margin-bottom:6px}p{margin:8px 0}.columns{display:grid;grid-template-columns:repeat(3,minmax(320px,1fr));gap:16px}'
             'article{background:white;border:1px solid #cad1db;border-radius:10px;padding:16px;min-width:0}'
             '.node{margin:10px 0 10px 10px;border-left:2px solid #d7dfeb;padding-left:10px}'
             'summary{cursor:pointer;font-weight:650}small,.node-id{color:#536170;font-size:12px}'
             'summary small{display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f8;padding:10px;font-size:12px}'
             '.members{padding-left:20px}.members li{margin-bottom:10px}.error{color:#8c2525;background:#ffeded;padding:10px}'
             'table{border-collapse:collapse;margin:12px 0}td,th{padding:5px 10px;text-align:left;border-bottom:1px solid #dce2ea}'
             'button{padding:7px 12px;margin:5px 5px 12px 0;cursor:pointer}.markdown{margin:10px 0}'
             '@media(max-width:1100px){.columns{grid-template-columns:1fr}}',
             '</style><h1>Navigation structure comparison</h1>',
             '<p>Same source units; alternate grouping strategies. Open nodes to inspect summaries, source members and the actual skill text.</p>',
             f'<p><b>All three arms accepted:</b> {_esc(report["all_three_accepted"])} · '
             f'<b>Valid builds:</b> {report["valid_arm_count"]}/{report["expected_arm_count"]}</p>',
             '<p>Depth counts edges from the root. Cross-document ratio counts only group nodes; a missing group denominator is unavailable.</p>',
             '<button onclick="document.querySelectorAll(\'details.node\').forEach(e=>e.open=true)">Expand trees</button>',
             '<button onclick="document.querySelectorAll(\'details.node\').forEach(e=>e.open=false)">Collapse trees</button>',
             '<div class="columns">']
    for strategy in ARMS:
        arm_report = report["arms"][strategy]
        parts.extend([f'<article><h2>{_esc(strategy)}</h2>', f'<p>Status: {_esc(arm_report["status"])}</p>'])
        if arm_report.get("error"):
            parts.append(f'<p class="error">{_esc(arm_report["error"])}</p>')
        if strategy in loaded:
            metrics = arm_report["metrics"]
            parts.append('<table>')
            for field in ("nodes", "leaves", "max_depth", "max_children", "max_navigation_children", "secondary_edges", "max_level_width", "cross_document_group_ratio"):
                parts.append(f'<tr><th>{_esc(field)}</th><td>{_esc(metrics[field])}</td></tr>')
            parts.append('</table>')
            parts.append('<details><summary>Offline build cost</summary><pre>' +
                         _esc(json.dumps({"elapsed_seconds": metrics["build_elapsed_seconds"],
                                         "llm_cost": metrics["llm_cost"]}, indent=2, ensure_ascii=False)) + '</pre></details>')
            if arm_report.get("qa"):
                parts.append('<details><summary>QA results</summary><pre>' + _esc(json.dumps(arm_report["qa"], indent=2, ensure_ascii=False)) + '</pre></details>')
            parts.append(_render_tree(loaded[strategy], loaded[strategy]["tree"]["root_id"]))
        parts.append('</article>')
    parts.append('</div></html>')
    return "\n".join(parts)


def _qa_rows(path):
    rows = read_jsonl(path) if Path(path).suffix.lower() == ".jsonl" else _read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("questions", rows.get("qa"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("QA input must contain a nonempty question list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str) or not row["qid"] or row["qid"] in seen:
            raise ValueError("QA rows require unique, nonempty qid strings")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("QA rows require a question")
        seen.add(row["qid"])
        if "reference_answer" in row and (not isinstance(row["reference_answer"], str) or not row["reference_answer"].strip()):
            raise ValueError("reference_answer must be a nonempty string when supplied")
        for field in ("gold_section_ids", "gold_doc_ids"):
            if field in row and (not isinstance(row[field], list) or not row[field]
                                 or any(not isinstance(x, str) or not x for x in row[field])):
                raise ValueError(field + " must be a nonempty list of IDs")
    return rows


def _score_answer(question, result, scorer):
    if "reference_answer" not in question:
        return {"answer_score_status": "unlabeled", "answer_score": None}
    if scorer is None:
        return {"answer_score_status": "not_requested", "answer_score": None}
    answer = result.get("answer")
    if result.get("status") != "ok" or not isinstance(answer, str) or not answer.strip():
        return {"answer_score_status": "execution_failed", "answer_score": 0.0}
    try:
        value = scorer(question=question["question"], answer=answer,
                       reference_answer=question["reference_answer"], history=deepcopy(question.get("history", [])))
        score = value.get("score") if isinstance(value, dict) else value
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Answer scorer must return a finite score between 0 and 1")
        reason = value.get("reason") if isinstance(value, dict) else None
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Answer scorer reason must be a string")
        return {"answer_score_status": "scored", "answer_score": float(score),
                **({"answer_score_reason": reason} if reason is not None else {})}
    except Exception as exc:
        # A judge outage is not evidence that the answer is wrong. Keep the
        # denominator fixed and mark the aggregate unavailable until resolved.
        return {"answer_score_status": "scorer_failed", "answer_score": None,
                "answer_score_error": {"type": type(exc).__name__, "message": str(exc)}}


def _score_qa(question, result, *, substantive_section_ids=None, section_documents=None, answer_scorer=None):
    rounds = result.get("rounds") or [{"stages": result.get("stages", {})}]
    routes = [r.get("stages", {}).get("navigation", {}) for r in rounds]
    selected = {sid for r in routes if r.get("status") == "ok" for sid in r.get("scope", {}).get("section_ids", [])}
    docs = {did for r in routes if r.get("status") == "ok" for did in r.get("scope", {}).get("doc_ids", [])}
    if substantive_section_ids is not None:
        # Routing-only source containers are not extra evidence. Counting them
        # would penalize source trees even when the same body units are selected.
        selected &= set(substantive_section_ids)
        if section_documents is not None:
            docs = {section_documents[sid] for sid in selected}
    score = {"qid": question["qid"], "status": result.get("status", "failed"),
             "answerable": result.get("answerable"), "navigation_ok": bool(routes) and all(r.get("status") == "ok" for r in routes),
             "navigation_calls": sum(r.get("navigation_calls", 0) for r in routes),
             "navigation_input_tokens": sum(t.get("input_tokens_estimate", 0) for r in routes for t in r.get("trace", [])),
             "selected_sections": len(selected), "selected_documents": len(docs), "error": result.get("error")}
    for name, chosen in (("section", selected), ("document", docs)):
        gold = question.get("gold_section_ids" if name == "section" else "gold_doc_ids")
        if gold:
            gold = set(gold)
            score[name + "_scope_recall"] = len(gold & chosen) / len(gold)
            score[name + "_scope_precision"] = len(gold & chosen) / len(chosen) if chosen else 0.0
            score[name + "_scope_complete"] = int(gold <= chosen)
    return {**score, **_score_answer(question, result, answer_scorer)}


def _qa_summary(rows):
    total = len(rows)
    report = {"questions": total, "ok": sum(r["status"] == "ok" for r in rows),
              "failed": sum(r["status"] != "ok" for r in rows),
              "navigation_ok_rate": sum(r["navigation_ok"] for r in rows) / total,
              "answerable_rate": sum(r.get("answerable") is True for r in rows) / total,
              "mean_navigation_calls": statistics.mean(r["navigation_calls"] for r in rows),
              "mean_navigation_input_tokens": statistics.mean(r["navigation_input_tokens"] for r in rows),
              "answer_quality": "Not graded; answerable is the model's self-report, not correctness."}
    for kind in ("section", "document"):
        eligible = [r for r in rows if kind + "_scope_recall" in r]
        report[kind + "_gold_denominator"] = len(eligible)
        for metric in ("recall", "precision", "complete"):
            field = kind + "_scope_" + metric
            report[field] = statistics.mean(r[field] for r in eligible) if eligible else None
    eligible = [r for r in rows if r.get("answer_score_status") != "unlabeled"]
    statuses = [r["answer_score_status"] for r in eligible]
    complete = bool(eligible) and all(s in ("scored", "execution_failed") for s in statuses)
    report.update(answer_gold_denominator=len(eligible),
                  answer_score=statistics.mean(r["answer_score"] for r in eligible) if complete else None,
                  answer_scorer_failures=statuses.count("scorer_failed"),
                  answer_execution_failures=statuses.count("execution_failed"),
                  answer_scoring_complete=complete)
    if complete:
        report["answer_quality"] = "Common injected scorer; execution failures and empty answers score zero."
    elif "scorer_failed" in statuses:
        report["answer_quality"] = "Unavailable: scorer failed; labeled-question denominator is unchanged."
    return report


def _run_qa(arm, questions, config, session_factory, answer_scorer=None):
    settings = deepcopy(config)
    settings["bundle_dir"] = str(arm["root"])
    results, runtime_manifest = [], None
    try:
        with session_factory(settings) as session:
            runtime_manifest = deepcopy(getattr(session, "manifest", None))
            for question in questions:
                try:
                    kwargs = {"qid": question["qid"]}
                    if "history" in question:
                        kwargs["history"] = question["history"]
                    result = session.agent.answer(question["question"], **kwargs)
                except Exception as exc:
                    result = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
                results.append({"qid": question["qid"], "result": result})
    except Exception as exc:
        for question in questions[len(results):]:
            results.append({"qid": question["qid"], "result": {"status": "failed", "error": {
                "type": type(exc).__name__, "message": "Session unavailable: " + str(exc)}}})
    substantive = {u["section_id"] for u in arm["units"]}
    scores = [_score_qa(q, r["result"], substantive_section_ids=substantive,
                        section_documents=arm["tree"]["section_documents"], answer_scorer=answer_scorer)
              for q, r in zip(questions, results)]
    return results, scores, {**_qa_summary(scores), "runtime_manifest": runtime_manifest}


def compare_bundles(bundle_dirs, output_dir, *, qa_config=None, qa_path=None, session_factory=None,
                    answer_scorer=None, answer_scorer_id=None):
    """Write JSON, CSV and standalone HTML; invalid arms remain in the denominator.

    ``qa_path`` accepts arbitrary JSON/JSONL rows with qid and question, plus
    optional history, gold_section_ids, gold_doc_ids and reference_answer. Gold
    never enters the agent. Optional ``answer_scorer`` is one common callback:
    scorer(question=..., answer=..., reference_answer=..., history=...) -> float
    in [0, 1], or {score: float, reason: str}. Supply a versioned scorer ID. There
    is no default judge/model call. Unlabeled questions are excluded identically
    in all arms; failed executions receive zero on the fixed labeled denominator.
    """
    if not isinstance(bundle_dirs, dict) or set(bundle_dirs) - set(ARMS):
        raise ValueError("Expected a mapping of source, llm and corpus2skill build directories")
    if (qa_config is None) != (qa_path is None):
        raise ValueError("qa_config and qa_path must be supplied together")
    if answer_scorer is not None:
        if qa_path is None or not callable(answer_scorer):
            raise ValueError("answer_scorer requires QA input and must be callable")
        if not isinstance(answer_scorer_id, str) or not answer_scorer_id.strip():
            raise ValueError("An explicit versioned answer_scorer_id is required")
    elif answer_scorer_id is not None:
        raise ValueError("answer_scorer_id requires an answer_scorer callback")
    output = Path(output_dir).resolve()
    for directory in bundle_dirs.values():
        if directory is not None:
            source = Path(directory).resolve()
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("Comparison output must not overlap an input build")
    if output.exists():
        raise FileExistsError("Comparison output already exists; choose a new report directory")
    questions = _qa_rows(qa_path) if qa_path is not None else None
    loaded, arms = {}, {}
    for strategy in ARMS:
        directory = bundle_dirs.get(strategy)
        if directory is None or not Path(directory).exists():
            arms[strategy] = {"status": "missing", "error": "Build directory was not supplied or does not exist"}
            continue
        try:
            loaded[strategy] = _load_arm(directory, strategy)
            arms[strategy] = {"status": "valid", "metrics": _metrics(loaded[strategy]),
                              "build": loaded[strategy]["status"], "directory": str(loaded[strategy]["root"])}
        except Exception as exc:
            loaded.pop(strategy, None)
            arms[strategy] = {"status": "invalid", "error": type(exc).__name__ + ": " + str(exc)}
            try:
                build = _read_json(Path(directory) / "build_status.json")
                if build.get("status") != "complete":
                    arms[strategy]["status"] = "failed" if build.get("status") == "failed" else "incomplete"
                    arms[strategy]["build"] = build
            except (OSError, ValueError, TypeError):
                pass
    identities = {(a["prepared"]["fingerprint"], json.dumps(a["prepared"]["common_hashes"], sort_keys=True))
                  for a in loaded.values()}
    common_identical = len(identities) == 1 if loaded else False
    methods = {json.dumps({"llm_configuration": a["prepared"]["llm_configuration"],
                           "descriptor_policy": a["prepared"]["descriptor_policy"]}, sort_keys=True)
               for a in loaded.values()}
    method_identical = len(methods) == 1 if loaded else False
    comparable = common_identical and method_identical
    if len(identities) > 1 or len(methods) > 1:
        for strategy in loaded:
            arms[strategy]["status"] = "incomparable"
            arms[strategy]["error"] = "Completed arms differ in shared corpus hashes, preparation fingerprints, model settings or descriptor policy"
    accepted = len(loaded) == len(ARMS) and comparable
    report = {"schema_version": "navigation-comparison-v1", "all_three_accepted": accepted,
              "expected_arm_count": len(ARMS), "valid_arm_count": len(loaded),
              "comparable_arm_count": len(loaded) if comparable else 0,
              "common_artifacts_identical": common_identical, "arms": arms,
              "common_model_and_descriptor_policy_identical": method_identical,
              "qa_requested": questions is not None,
              "answer_scorer_id": answer_scorer_id,
              "notes": ["Structural acceptance does not establish which strategy is better.",
                        "Every source unit has one primary path; paths CSV includes every valid arm and every unit.",
                        "QA scope counts and precision use substantive sections from shared units; empty source containers are excluded.",
                        "Pages in units.jsonl are displayed as provided; section page indices are converted to one-based pages."]}
    if loaded and comparable:
        shared = next(iter(loaded.values()))["prepared"]
        report["shared_preparation_cost"] = {"elapsed_seconds": shared.get("elapsed_seconds"),
                                              "llm_cost": deepcopy(shared.get("llm_cost", {}))}
    if questions is not None:
        config = load_config(qa_config) if isinstance(qa_config, (str, Path)) else deepcopy(qa_config)
        if not isinstance(config, dict):
            raise ValueError("qa_config must be a configuration object or path")
        if loaded and comparable:
            canonical = next(iter(loaded.values()))
            substantive = {u["section_id"] for u in canonical["units"]}
            for question in questions:
                if set(question.get("gold_section_ids", [])) - canonical["sections"].keys():
                    raise ValueError("QA gold_section_ids contain an unknown section")
                if set(question.get("gold_section_ids", [])) - substantive:
                    raise ValueError("QA gold_section_ids must refer to substantive source units, not empty sections")
                if set(question.get("gold_doc_ids", [])) - canonical["documents"].keys():
                    raise ValueError("QA gold_doc_ids contain an unknown document")
        if session_factory is None:
            from .factory import create_comparison_session
            session_factory = create_comparison_session
    output.mkdir(parents=True, exist_ok=False)
    if questions is not None:
        for strategy in ARMS:
            if strategy in loaded and comparable:
                results, scores, summary = _run_qa(loaded[strategy], questions, config, session_factory, answer_scorer)
                write_jsonl(output / (strategy + "_qa_results.jsonl"), results)
                write_jsonl(output / (strategy + "_qa_scores.jsonl"), scores)
                arms[strategy]["qa"] = summary
            else:
                scores = [_score_qa(q, {"status": "failed", "error": {"message": "Build unavailable or incomparable"}},
                                    answer_scorer=answer_scorer) for q in questions]
                arms[strategy]["qa"] = {**_qa_summary(scores), "status": "not_run", "reason": "Build unavailable or incomparable"}
        report["qa_questions_per_arm"] = len(questions)
        report["all_three_qa_completed"] = accepted and all(arms[s]["qa"]["failed"] == 0 for s in ARMS)
        if answer_scorer is not None:
            report["all_three_answer_scoring_completed"] = all(arms[s]["qa"]["answer_scoring_complete"] for s in ARMS)
    mapping = [row for strategy in ARMS if strategy in loaded for row in _mapping_rows(strategy, loaded[strategy])]
    fields = ["strategy", "unit_id", "section_id", "doc_id", "document_title", "unit_title", "pages", "owner_node_id", "path_node_ids", "path_titles", "secondary_entrances"]
    with (output / "unit_paths.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in mapping)
    report["artifacts"] = {"report": "comparison_report.json", "review": "comparison.html", "unit_paths": "unit_paths.csv"}
    write_json(output / "comparison_report.json", report)
    (output / "comparison.html").write_text(_html(report, loaded), encoding="utf-8")
    return report
