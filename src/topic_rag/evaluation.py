"""Source recall and exact evidence-span recall; never lexical answer F1."""
import statistics


def interval_covered(start, end, intervals):
    cursor = start
    for left, right in sorted(intervals):
        if left > cursor:
            break
        if right > cursor:
            cursor = right
        if cursor >= end:
            return True
    return False


def evidence_metrics(chunks, evidence, cutoffs=(5, 10, 20)):
    evidence = list({e["evidence_id"]: e for e in evidence}.values())
    gold_sources = {e["source_id"] for e in evidence}
    metrics = {}
    for k in cutoffs:
        selected = chunks[:k]
        hits = []
        for atom in evidence:
            spans = [(c["source_start"], c["source_end"]) for c in selected
                     if c["source_id"] == atom["source_id"]]
            if interval_covered(atom["start"], atom["end"], spans):
                hits.append(atom["evidence_id"])
        matched_sources = gold_sources & {c["source_id"] for c in selected}
        metrics[str(k)] = {
            "matched": len(hits), "eligible": len(evidence), "matched_evidence_ids": hits,
            "evidence_recall": len(hits) / len(evidence) if evidence else None,
            "source_recall": len(matched_sources) / len(gold_sources) if gold_sources else None,
            "any_hit": bool(hits), "complete": bool(evidence) and len(hits) == len(evidence),
        }
    return metrics


def summarize(rows):
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [r for r in rows if r["variant"] == variant]
        ok = [r for r in group if not r.get("error")]
        retrieved = [r for r in group if "evidence" in r]
        summary = {"variant": variant, "question_count": len(group), "success_count": len(ok),
                   "error_count": len(group) - len(ok), "retrieval_success_count": len(retrieved)}
        for kind in ("answer", "clarify", "abstain", "not_run"):
            summary[kind + "_count"] = sum(r.get("response_type") == kind for r in ok)
        for k in (5, 10, 20):
            records = [r["evidence"][str(k)] for r in retrieved]
            summary[f"evidence_recall_macro_at_{k}"] = statistics.mean(
                [r["evidence_recall"] for r in records]) if records else None
            total = sum(r["eligible"] for r in records)
            summary[f"evidence_recall_micro_at_{k}"] = sum(r["matched"] for r in records) / total if total else None
            summary[f"source_recall_at_{k}"] = statistics.mean([r["source_recall"] for r in records]) if records else None
            # Also expose a conservative denominator including retrieval errors.
            summary[f"recall_macro_errors_as_zero_at_{k}"] = sum(r["evidence_recall"] for r in records) / len(group)
        summary["any_evidence_hit_rate_at_5"] = statistics.mean([r["evidence"]["5"]["any_hit"] for r in retrieved]) if retrieved else None
        summary["complete_evidence_rate_at_5"] = statistics.mean([r["evidence"]["5"]["complete"] for r in retrieved]) if retrieved else None
        for component in ("query_embedding", "topic_scoring", "retrieval", "answer", "total"):
            values = [r["latency_ms"][component] for r in ok]
            summary[f"mean_{component}_ms"] = statistics.mean(values) if values else None
        # No automatic accuracy claim: a separate blind semantic review is needed.
        summary["answer_accuracy"] = None
        summary["accuracy_status"] = "awaiting_semantic_review"
        output.append(summary)
    return output

