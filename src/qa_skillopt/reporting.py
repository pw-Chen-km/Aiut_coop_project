"""Read-only audit reporting for navigation-only SkillOPT runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .io import read_json, read_jsonl


def optional(path, default=None):
    return read_json(path) if Path(path).is_file() else default


def _result_for(path: Path, qid: str) -> dict:
    payload = optional(path / "qa_results.json", {})
    return next((row for row in payload.get("results", []) if row.get("qid") == qid), {})


def _lane(work: Path, raw_path: Path) -> tuple[str, str, Path]:
    evaluation = raw_path.parents[2]
    relative = evaluation.relative_to(work).as_posix()
    if relative == "upstream/selection_eval_baseline":
        lane = "upstream_navigation_baseline"
    elif relative.endswith("/rollout"):
        lane = "optimization_rollout"
    elif relative.endswith("/selection_eval"):
        lane = "upstream_navigation_candidate"
    else:
        lane = relative.replace("/", "_")
    payload = optional(evaluation / "qa_results.json", {})
    inferred = "optimization" if relative.endswith("/rollout") else "validation"
    return lane, payload.get("phase", inferred), evaluation


def inspect_run(root):
    root = Path(root).resolve()
    questions, calls, targets = [], [], []
    for job_file in sorted(root.glob("job_*.json")):
        job = read_json(job_file)
        number = job_file.stem.removeprefix("job_")
        work = root / ("step_" + number)
        result = optional(work / "result.json", {})
        if not result.get("summary"):
            baseline = optional(work / "upstream/selection_eval_baseline/qa_results.json", {})
            if "hard_accuracy" in baseline:
                result = {**result, "summary": {"baseline_selection_hard": baseline["hard_accuracy"]}}
        checks = [read_json(path) for path in sorted(work.glob("candidate_check_*.json"))]
        reflections = [read_json(path) for path in sorted(work.glob("upstream/steps/*/reflection_selection.json"))]
        targets.append({"step": number, "target": job["active_target"],
                        "config": optional(work / "upstream/config.json", {}),
                        "result": result, "checks": checks, "reflections": reflections,
                        "history": optional(work / "upstream/history.json", []),
                        "preflight": optional(work / "preflight.json", {}), "directory": str(work)})
        for raw_path in sorted(work.glob("**/predictions/*/raw_trajectory.json")):
            raw = read_json(raw_path)
            lane, phase, evaluation = _lane(work, raw_path)
            questions.append({"target_step": number, "target": job["active_target"],
                              "phase": phase, "lane": lane, "raw": raw,
                              "metrics": optional(raw_path.parent / "navigation_metrics.json", {}),
                              "optimizer_view": optional(raw_path.parent / "navigation_trajectory.json"),
                              "judge": optional(raw_path.parent / "model_judgment.json", {}),
                              "error": optional(raw_path.parent / "unscored.json", {}),
                              "result": _result_for(evaluation, raw.get("qid")),
                              "path": str(raw_path)})
        for request_path in work.glob("offline_calls/*/request.json"):
            request = read_json(request_path)
            calls.append({"target_step": number, "role": request.get("role"), "request": request,
                          "response": optional(request_path.parent / "response.json", {}),
                          "path": str(request_path.parent)})
    return {"directory": str(root), "targets": targets, "questions": questions,
            "calls": sorted(calls, key=lambda row: row["request"].get("time_unix", 0)),
            "selected": optional(root / "selected.json"),
            "interrupted": optional(root / "interrupted.json")}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _sensitive_strings(rows):
    values = []
    for row in rows:
        answer = row.get("answer")
        if isinstance(answer, str):
            values.append(("answer", answer))
        elif isinstance(answer, dict):
            values.extend(("answer", value) for value in answer.values() if isinstance(value, str))
        values.extend(("claim", claim.get("text", "")) for claim in row.get("claims", []))
        values.extend(("evidence_quote", item.get("quote", "")) for item in row.get("evidence", []))
    return [(kind, " ".join(value.split())) for kind, value in values
            if isinstance(value, str) and len(" ".join(value.split())) >= 16]


def leakage_audit(run, optimization_rows, validation_rows):
    allowed = {
        "overall": {"schema_version", "user_utterance", "selected_document_ids_by_round",
                    "required_document_id_sets", "missing_document_ids", "extra_document_ids",
                    "document_route_complete"},
        "document": {"schema_version", "user_utterance", "active_document_id",
                     "selected_section_ids_by_round", "required_section_id_sets",
                     "missing_section_ids", "extra_section_ids", "section_route_complete"},
    }
    forbidden = {"qid", "answer", "answerable", "claims", "evidence", "quote", "citations",
                 "retrieved_text", "context_items", "judge_reason", "reference_answer"}
    trajectory_errors = []
    for row in run["questions"]:
        view = row.get("optimizer_view")
        if view is None:
            continue
        kind = "overall" if row["target"] == "overall" else "document"
        keys = set(_walk_keys(view))
        if set(view) != allowed[kind] or keys & forbidden:
            trajectory_errors.append({"target": row["target"], "path": row["path"],
                                      "unexpected_keys": sorted((set(view) - allowed[kind]) | (keys & forbidden))})
    optimizer_texts = ["\n".join(message.get("content", "") for message in call["request"].get("payload", {}).get("messages", []))
                       for call in run["calls"] if call["role"] == "optimizer"]
    sensitive = _sensitive_strings(optimization_rows)
    content_hits = []
    for index, text in enumerate(optimizer_texts, 1):
        normalized = " ".join(text.split())
        for kind, value in sensitive:
            if value in normalized:
                content_hits.append({"optimizer_call": index, "kind": kind,
                                     "value_sha256": hashlib.sha256(value.encode()).hexdigest()})
    validation_hits = []
    for index, text in enumerate(optimizer_texts, 1):
        normalized = " ".join(text.split())
        for row in validation_rows:
            question = " ".join(row["question"].split())
            if question and question in normalized:
                validation_hits.append({"optimizer_call": index,
                                        "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
    serialized_forbidden = []
    pattern = re.compile(r'"(?:qid|answer|answerable|claims|evidence|quote|citations|reference_answer)"\s*:')
    for index, text in enumerate(optimizer_texts, 1):
        if pattern.search(text):
            serialized_forbidden.append(index)
    failed = trajectory_errors or content_hits or validation_hits or serialized_forbidden
    return {"status": "failed" if failed else "passed",
            "navigation_trajectory_count": sum(row.get("optimizer_view") is not None for row in run["questions"]),
            "optimizer_call_count": len(optimizer_texts), "trajectory_schema_errors": trajectory_errors,
            "sensitive_content_hits": content_hits, "validation_question_hits": validation_hits,
            "serialized_forbidden_field_calls": serialized_forbidden}


def _pct(value):
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_cells(metrics):
    return [_pct(metrics.get(key)) for key in
            ("route_complete", "route_f1", "evidence_recall", "evidence_complete", "primary_score")]


def render_report(run, *, preparation_dir, previous=()):
    prep = Path(preparation_dir).resolve()
    optimization = read_jsonl(prep / "optimization.jsonl")
    validation = read_jsonl(prep / "validation.jsonl")
    phases = {phase: read_json(prep / f"{phase}_manifest.json")
              for phase in ("optimization", "validation", "test")}
    nav_manifest = optional(prep / "navigation_manifest.json", {
        "target_label_counts": {"optimization": {}, "validation": {}},
        "eligible_targets": [], "skipped_targets": [],
        "bundle_manifest_sha256": "not-recorded", "card_snapshot_sha256": {}})
    audit = leakage_audit(run, optimization, validation)
    selected, interrupted = run["selected"], run["interrupted"]
    status = "completed" if selected else "interrupted" if interrupted else "incomplete"
    roles = Counter(call["role"] for call in run["calls"])
    usage = Counter()
    for call in run["calls"]:
        response = call["response"].get("data", {})
        usage["input_tokens"] += response.get("prompt_eval_count", 0)
        usage["output_tokens"] += response.get("eval_count", 0)

    lines = ["# QURSOR × SkillOPT Navigation-only 重跑報告", "",
             f"狀態：**{status}**；Run：`{Path(run['directory']).name}`。", "",
             "## 1. 這次真正優化的東西", "",
             "SkillOPT 只接收使用者問句與 selected/required routing IDs。Overall 只負責選文件；Document skill 只負責選章節。",
             "答案、claims、evidence quote、取回原文和 judge reason 不進 optimizer。GT evidence 只在外部映射成 routing labels 與計分。", "",
             "- SkillOPT：v0.2.0 / `e4ea6a6771e797ef820cdd8bfea64c57e0481065`。",
             "- Optimization batch：20；reflection minibatch：5；最多兩個 cycle。",
             "- Primary：50% route complete + 25% route F1 + 25% evidence recall。",
             "- 接受前先要 navigation primary 嚴格上升，再要求 40 題 final-answer accuracy 不下降。",
             "- 120 題 held-out test 未執行。", "",
             "## 2. 資料邊界與 target 充足性", "",
             "| Phase | 分配 | 可用 | 用途 |", "| --- | ---: | ---: | --- |"]
    purposes = {"optimization": "routing trajectory 反思", "validation": "candidate gate，不反思",
                "test": "本次不讀取不執行"}
    for phase in ("optimization", "validation", "test"):
        lines.append(f"| {phase} | {phases[phase]['assigned_count']} | {phases[phase]['eligible_count']} | {purposes[phase]} |")
    lines += ["", "| Target | optimization labels | validation labels | 決定 |",
              "| --- | ---: | ---: | --- |"]
    skipped = {row["target"]: row for row in nav_manifest.get("skipped_targets", [])}
    targets = sorted(set(nav_manifest["target_label_counts"]["optimization"]) |
                     set(nav_manifest["target_label_counts"]["validation"]),
                     key=lambda value: (value != "overall", value))
    for target in targets:
        decision = "run" if target in nav_manifest.get("eligible_targets", []) else skipped[target]["reason"]
        lines.append(f"| `{target}` | {nav_manifest['target_label_counts']['optimization'].get(target, 0)} | "
                     f"{nav_manifest['target_label_counts']['validation'].get(target, 0)} | {decision} |")

    lines += ["", "## 3. Target 結果", "",
              "| Target | 實際 opt/val | Upstream baseline→best | 外部 navigation gate | answer guard | 最終接受 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for target in run["targets"]:
        result = target["result"]
        summary = result.get("summary", {})
        den = result.get("target_denominators", {})
        nav = result.get("navigation_comparison", {})
        answer = result.get("answer_guard", {})
        base = summary.get("baseline_gate_score", summary.get("baseline_selection_hard"))
        best = summary.get("best_gate_score", result.get("best_validation_score"))
        lines.append(f"| `{target['target']}` | {den.get('optimization', '—')}/{den.get('validation', '—')} | "
                     f"{_cell(base)} → {_cell(best)} | {nav.get('status', '—')} | {answer.get('status', '—')} | "
                     f"{result.get('accepted', False)} |")
        comparison = result.get("navigation_comparison", {})
        if comparison.get("baseline") and comparison.get("candidate"):
            lines += ["", f"### `{target['target']}` navigation metrics", "",
                      "| Version | route complete | route F1 | evidence recall | evidence complete | primary |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for name in ("baseline", "candidate"):
                cells = _metric_cells(comparison[name])
                lines.append("| " + name + " | " + " | ".join(cells) + " |")
            lines += ["", f"第一輪 route complete：{_pct(comparison['baseline'].get('first_round_route_complete'))} → "
                      f"{_pct(comparison['candidate'].get('first_round_route_complete'))}；上表 route complete 是兩輪累積結果。", ""]
        if answer.get("status") not in {None, "not_run_no_upstream_candidate"}:
            lines += [f"Answer accuracy：{_pct(answer.get('baseline_accuracy'))} → "
                      f"{_pct(answer.get('candidate_accuracy'))}；改善 qids={answer.get('improved_qids', [])}；"
                      f"退步 qids={answer.get('regressed_qids', [])}。", ""]

    lines += ["## 4. Candidate 修改與拒絕原因", ""]
    check_count = 0
    for target in run["targets"]:
        for index, check in enumerate(target["checks"], 1):
            check_count += 1
            side = check.get("sidecar", {})
            grounding = side.get("grounding", {})
            lines += [f"### Step {target['step']} / candidate {index} / `{target['target']}`", "",
                      f"- valid：`{check.get('valid')}`；errors：`{check.get('errors', [])}`",
                      f"- grounding：`{grounding.get('status')}`；reason：{_cell(grounding.get('reason', '—'))}",
                      f"- source refs：`{grounding.get('source_refs', [])}`"]
            edits = side.get("card_field_edits", [])
            if edits:
                lines += ["", "| card | field | before | after |", "| --- | --- | --- | --- |"]
                for edit in edits:
                    lines.append(f"| `{edit.get('card_id')}` | `{edit.get('field')}` | "
                                 f"{_cell(edit.get('before'))} | {_cell(edit.get('after'))} |")
            lines.append("")
    if not check_count:
        lines += ["沒有 candidate 到達 validator；因此沒有 MD 修改可接受或匯出。", ""]

    lines += ["## 5. Leakage audit", "",
              f"結果：**{audit['status']}**；navigation trajectories={audit['navigation_trajectory_count']}；"
              f"optimizer calls={audit['optimizer_call_count']}。", "",
              f"- trajectory schema errors：`{audit['trajectory_schema_errors']}`",
              f"- answer/claim/evidence exact-content hits：`{audit['sensitive_content_hits']}`",
              f"- validation-question hits：`{audit['validation_question_hits']}`",
              f"- serialized forbidden-field calls：`{audit['serialized_forbidden_field_calls']}`", "",
              "Grounding verifier 可讀原始 corpus，但與 optimizer role 分開；其 source refs 只留在私有 audit sidecar。", "",
              "## 6. 軌跡與失敗隔離", "",
              "| Target | Lane | calls | route success | route failure | answer judged |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    grouped = Counter()
    for row in run["questions"]:
        result = row.get("result", {})
        grouped[(row["target"], row["lane"], "calls")] += 1
        if result.get("navigation_scorable"):
            grouped[(row["target"], row["lane"], "route_success" if result.get("hard") else "route_failure")] += 1
        if "hard" in row.get("judge", {}):
            grouped[(row["target"], row["lane"], "answer_judged")] += 1
    for target, lane in sorted({(key[0], key[1]) for key in grouped}):
        lines.append(f"| `{target}` | {lane} | {grouped[(target, lane, 'calls')]} | "
                     f"{grouped[(target, lane, 'route_success')]} | {grouped[(target, lane, 'route_failure')]} | "
                     f"{grouped[(target, lane, 'answer_judged')]} |")
    lines += ["", "Dense retrieval miss、context trimming、generation failure 不會被標成 routing failure。"
              "Routing 對但答案錯，仍屬 navigation success；routing 不完整但答案對，仍屬 navigation failure。", "",
              "## 7. 版本與呼叫紀錄", "",
              f"Offline calls by role：`{dict(roles)}`；tokens：`{dict(usage)}`。", "",
              f"Bundle manifest SHA-256：`{nav_manifest['bundle_manifest_sha256']}`", "",
              f"Card baseline hashes：`{nav_manifest['card_snapshot_sha256']}`", ""]
    for target in run["targets"]:
        if target["preflight"]:
            lines.append(f"- Step {target['step']}：model=`{target['preflight'].get('model')}`；"
                         f"digest=`{target['preflight'].get('digest')}`；context={target['preflight'].get('context_length')}。")
    if interrupted:
        lines += ["", f"中斷：`{interrupted.get('error_type')}` — {_cell(interrupted.get('reason'))}。"]
        for row in run["questions"]:
            if row.get("error"):
                lines.append(f"- `{row['raw'].get('qid')}`：{_cell(row['error'].get('reason', row['error']))}")

    lines += ["", "## 8. 與舊錯誤 run 的關鍵差異", "",
              "| 舊 run | 本次 |", "| --- | --- |",
              "| hard/success/failure 來自最終答案 | hard 只是 target route_complete |",
              "| optimizer 可見 reference answer、claims、quote | optimizer 只見問句與 routing IDs/缺口 |",
              "| skill 可被改成解題手冊 | 只允許六個 navigation-card fields |",
              "| final-answer accuracy 是唯一 gate | navigation primary 是主 gate；answer 只防退步 |", ""]
    if previous:
        lines.append("供比對的舊 run：" + ", ".join(f"`{Path(old['directory']).name}`" for old in previous) + "。")
    lines += ["", "## 結論", ""]
    accepted = selected.get("last_accepted_version", 0) if selected else 0
    lines.append(f"本次最終接受 {accepted} 個 navigation update。"
                 "只有同時通過結構、corpus grounding、navigation gate 和 answer non-regression 的版本才會進 selected snapshot。")
    lines += ["", "這是 prototype evaluation：自動 oracle/judge 尚未經 AIUT SME 校準，不稱 frozen benchmark。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--preparation-dir", required=True)
    parser.add_argument("--previous-run", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = render_report(inspect_run(args.run_dir), preparation_dir=args.preparation_dir,
                           previous=[inspect_run(path) for path in args.previous_run])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(json.dumps({"report": str(output.resolve())}))


if __name__ == "__main__":
    main()
