# Extracted unchanged from Corpus2Skill b0108ce22d434983cf97abc2ca455aebc6b625dd
# corpus2skill/summarizer.py; MIT, see LICENSE in this directory.

_REPARTITION_PROMPT = (
    "You are auditing a clustering of items at level {level} of a navigable "
    "knowledge hierarchy. Each cluster has a label, a short summary, and a list "
    "of member item IDs with brief titles.\n\n"
    "Two failure modes to look for:\n"
    "  (a) CONFUSABLE: two or more clusters whose labels/summaries describe "
    "essentially the same topic — a query that fits one would plausibly fit "
    "the other.\n"
    "  (b) MIXED: a single cluster whose members visibly belong to multiple "
    "distinct sub-topics that should be split.\n\n"
    "For each problematic GROUP (a set of confusable cluster IDs, or a single "
    "mixed cluster ID), output a fresh partition of its members by ID. Use ONLY "
    "the IDs already listed in that group; do not invent IDs or move items "
    "across non-flagged boundaries.\n\n"
    "Return ONLY a JSON object with this shape:\n"
    '{{ "changes": [ '
    '{{ "old_cluster_ids": ["L1-C2", "L1-C7"], '
    '   "new_partition": {{ "label_a": ["id1","id2"], "label_b": ["id3","id4"] }} '
    "}} ] }}\n"
    "If no changes are needed, return {{ \"changes\": [] }}.\n\n"
    "Clusters at level {level}:\n{clusters_block}\n\n"
    "JSON:"
)

def _build_clusters_block(
    cluster_records: list[dict],
    max_members: int = 12,
) -> str:
    """Format clusters for the repartition prompt.

    Prompt-size safety lives upstream:
      * ``repartition_level`` enforces ``max_clusters_per_call`` (default 60).
      * Cluster summaries are themselves produced with bounded ``max_tokens``
        in ``_cluster_summary_prompt`` (typically 500–1500 chars).
      * Member titles are first-lines of child summaries (set in
        ``clustering.py``), naturally short.
    So we render summaries and titles verbatim — no per-field truncation.
    """
    lines = []
    for rec in cluster_records:
        cid = rec["cluster_id"]
        label = rec.get("label", "")
        summary = rec.get("summary") or ""
        members = rec.get("members", [])[:max_members]
        lines.append(f"- cluster_id: {cid}")
        lines.append(f"  label: {label}")
        lines.append(f"  summary: {summary}")
        lines.append(f"  members ({len(rec.get('members', []))} total, showing up to {max_members}):")
        for m in members:
            mid = m.get("id", "")
            title = m.get("title") or ""
            lines.append(f"    - {mid}: {title}")
    return "\n".join(lines)
