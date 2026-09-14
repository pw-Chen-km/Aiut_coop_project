"""Source-only evaluation, without mutable chunk/qrel or oracle-ID shortcuts."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.metrics import aggregate_query_metrics, evaluate_query


SOURCE_METRIC_PREFIXES = (
    "evidence_set_recall@", "complete_evidence_all_hops@", "source_atom_recall@",
    "complete_source_atoms_all_hops@", "source_recall@", "per_hop_recall@",
    "macro_hop_recall@", "weakest_hop_recall@",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _source_only_qa(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    result["hard_negatives"] = []
    for evidence in result.get("evidence", ()):
        span = evidence.get("source_span")
        if not isinstance(span, Mapping) or span.get("coordinate_space", "block_text") != "block_text":
            raise ValueError("source-only evaluation requires gold block_text source_span anchors")
        if not evidence.get("block_id") or not evidence.get("doc_id"):
            raise ValueError("gold source anchors require document and block identities")
        if int(span.get("start_char", -1)) < 0 or int(span.get("end_char", -1)) <= int(span.get("start_char", -1)):
            raise ValueError("gold source span offsets are invalid")
        # EvidenceMatcher's last fallback indexes QA chunk IDs. Removing them
        # disables over-credit when new/unmapped items happen to reuse old IDs.
        evidence.pop("chunk_id", None)
        evidence.pop("span", None)
    return result


def _source_only_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for rank, item in enumerate(items, 1):
        spans = item.get("source_spans") or []
        cleaned_spans = []
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError("returned source spans must be objects")
            start = span.get("start_char", span.get("source_start_char"))
            end = span.get("end_char", span.get("source_end_char"))
            if (not span.get("block_id") or start is None or end is None
                    or int(start) < 0 or int(end) <= int(start)
                    or span.get("coordinate_space", "block_text") != "block_text"):
                raise ValueError("returned source span has invalid frozen block coordinates")
            cleaned_spans.append({"block_id": str(span["block_id"]), "start_char": int(start),
                                  "end_char": int(end), "coordinate_space": "block_text"})
        if cleaned_spans and not item.get("doc_id"):
            raise ValueError("returned source spans require doc_id")
        if not item.get("chunk_id"):
            raise ValueError("retrieved items require a stable chunk_id")
        # Deliberate allowlist excludes evidence_ids, equivalence_groups, qrels.
        result.append({"chunk_id": str(item["chunk_id"]), "doc_id": item.get("doc_id"),
                       "rank": item.get("rank", rank), "score": item.get("score"),
                       "source_spans": cleaned_spans})
    return result


def evaluate_runs(
    qa_path: str | Path,
    runs: Sequence[Mapping[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 20),
    alignment_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score original frozen source anchors at 100% union coverage.

    ``runs`` is one ``{qid, items:[...]}`` record per query, or a JSONL path.
    Failed, empty and missing runs score as zero, never disappear from the denominator.
    No existing corpus/QA file is modified. Only an explicit output path is
    written, and existing files are refused to preserve experiment snapshots.
    """
    qa_path = Path(qa_path)
    qa = _read_jsonl(qa_path)
    if isinstance(runs, (str, Path)):
        runs = _read_jsonl(runs)
    qa_ids = [str(item.get("qid", "")) for item in qa]
    if any(not identifier for identifier in qa_ids) or len(set(qa_ids)) != len(qa_ids):
        raise ValueError("QA IDs must be nonempty and unique")
    runs_by_id = {}
    statuses_by_id: dict[str, str] = {}
    discarded_failed_items = 0
    for run in runs:
        qid = str(run.get("qid", ""))
        if qid not in set(qa_ids):
            raise ValueError(f"unknown run qid: {qid!r}")
        if qid in runs_by_id:
            raise ValueError(f"duplicate run qid: {qid!r}")
        status = str(run.get("status", "ok"))
        statuses_by_id[qid] = status
        items = run.get("items", [])
        if status != "ok":
            # Fail closed for explicit failure/timeout/setup failures and unknown
            # statuses, even if a caller bypasses the CLI's export_run guard.
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                discarded_failed_items += len(items)
            items = []
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError("run items must be a sequence")
        runs_by_id[qid] = items
    queries = []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched_items = 0
    total_items = 0
    for record in qa:
        qid = str(record["qid"])
        items = _source_only_items(runs_by_id.get(qid, []))
        total_items += len(items)
        unmatched_items += sum(not item["source_spans"] for item in items)
        metrics = evaluate_query(_source_only_qa(record), items, qrels=(), cutoffs=cutoffs, min_span_coverage=1.0)
        filtered = {key: value for key, value in metrics.items()
                    if key == "qid" or key.startswith(SOURCE_METRIC_PREFIXES)}
        diagnostics = dict(metrics["diagnostics"])
        diagnostics.pop("hard_negative_intrusions", None)
        diagnostics["run_status"] = statuses_by_id.get(qid, "missing_run")
        if any(method in {"qrels", "chunk_fallback", "explicit_evidence"}
               for method in diagnostics["matching_method_by_rank"]):
            raise AssertionError("non-source matching escaped source-only evaluation")
        filtered["diagnostics"] = diagnostics
        queries.append(filtered)
        strata[f"question_type:{record.get('question_type', 'unknown')}"].append(filtered)
        strata["visual_diagnostic:" + str(bool(record.get("visual_diagnostic"))).lower()].append(filtered)
        strata[f"difficulty:{record.get('difficulty', 'unknown')}"].append(filtered)
        strata[f"hop_count:{record.get('reasoning', {}).get('hop_count', 'unknown')}"].append(filtered)
        documents = sorted({str(item["doc_id"]) for item in record.get("evidence", ()) if item.get("doc_id")})
        for document in documents:
            strata[f"document:{document}"].append(filtered)
    reviews = Counter(str(item.get("review", {}).get("status", "unknown")) for item in qa)
    multihop = [item for item in qa if item.get("question_type") == "multihop"]
    split_path = qa_path.parent / "split_diagnostics.json"
    split_status = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {"release_status": "not_available"}
    split_summary = {key: value for key, value in split_status.items() if key != "pairs"}
    report = {
        "evaluation_schema_version": "doc2skill-source-eval-v1",
        "query_count": len(qa), "queries": queries,
        "aggregate": aggregate_query_metrics(queries),
        "strata": {key: {"query_count": len(values), "aggregate": aggregate_query_metrics(values)}
                   for key, values in sorted(strata.items())},
        "missing_run_qids": [qid for qid in qa_ids if qid not in runs_by_id],
        "execution_status": {
            "counts": dict(Counter(statuses_by_id.get(qid, "missing_run") for qid in qa_ids)),
            "failed_run_qids": [qid for qid in qa_ids if qid in statuses_by_id and statuses_by_id[qid] != "ok"],
            "discarded_items_from_failed_runs": discarded_failed_items,
        },
        "strata_policy": "Document strata overlap for multi-document questions; their counts must not be summed as disjoint partitions.",
        "min_source_atom_coverage": 1.0,
        "matching_policy": "canonical source spans only; full interval union; no explicit gold IDs, qrels, or chunk fallback",
        "omitted_metrics": ["chunk_recall", "mrr", "ndcg", "hard_negative_intrusion", "answer_accuracy"],
        "alignment": {"returned_items": total_items, "items_without_mapped_source_spans": unmatched_items,
                      "corpus_alignment_report": dict(alignment_report) if alignment_report is not None else None,
                      "interpretation": "Scores measure retrieval plus legacy-coordinate alignment. Unmapped or ambiguous source text limits measurable recall; corpus character coverage is not gold-evidence coverage."},
        "dataset_readiness": {"review_status_counts": dict(reviews),
                              "unverified_multihop_questions": sum(not item.get("reasoning", {}).get("hop_necessity", {}).get("verified", False) for item in multihop),
                              "split_diagnostics": split_summary,
                              "benchmark_freeze_claimed": False},
        "limitations": [
            "A source-span run must first be aligned to the frozen canonical extraction; this scorer does not verify claimed offsets against PDF bytes.",
            "Unmapped evidence receives no credit and remains in the query denominator.",
            "With incomplete alignment, low source recall or method differences must not be attributed to retrieval quality alone; inspect gold-evidence alignment and failed trajectories before drawing research conclusions.",
            "Equivalent provenance is limited to alternatives already registered in the gold annotations.",
            "Pending SME review and split adjudication are not resolved by this evaluation.",
        ],
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return report
