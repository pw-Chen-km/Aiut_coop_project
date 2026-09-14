"""Human-readable inspection queues; never label automated extraction SME-approved."""
from __future__ import annotations

from collections import Counter
from pathlib import Path


def write_inspection_report(corpus, root, alignment=None):
    root = Path(root)
    rows = ["# Doc2Skill corpus inspection", "",
            "Status: automated extraction; chapter boundaries, recovered text and tables require human inspection.",
            "Navigation metadata/skills are produced only by the separately configured LLM stage.", "",
            "| Document | Pages | Sections | Source units | Chunks | Figures |", "|---|---:|---:|---:|---:|---:|"]
    for doc in corpus["documents"]:
        counts = {key: sum(r["doc_id"] == doc["doc_id"] for r in corpus[key])
                  for key in ("sections", "source_units", "chunks", "figures")}
        rows.append(f"| {doc['title']} | {doc['page_count']} | {counts['sections']} | {counts['source_units']} | {counts['chunks']} | {counts['figures']} |")
    if alignment:
        rows += ["", "## Legacy evidence alignment", "",
                 "These are source-coordinate compatibility diagnostics, not retrieval performance.", "",
                 "```json", __import__("json").dumps(alignment["report"], ensure_ascii=False, indent=2), "```", "",
                 "Ambiguous/unmapped text gets no benchmark credit. Scores are alignment-limited until this mapping is audited."]
    for doc in corpus["documents"]:
        rows += ["", "## " + doc["title"], "", "Review the native section inventory against the PDF:", ""]
        for section in corpus["sections"]:
            if section["doc_id"] != doc["doc_id"]:
                continue
            start, end = section.get("page_start"), section.get("page_end")
            pages = f"pp. {start + 1}–{end + 1}" if start is not None and end is not None else "page unknown"
            rows.append(f"- [ ] {pages} · {' / '.join(section.get('path', []))} · `{section['section_id']}`")
        flagged = [b for b in corpus["blocks"] if b["doc_id"] == doc["doc_id"] and
                   (b.get("reading_order_unresolved") or "recover" in b.get("kind", "") or b.get("page_index") is None)]
        rows += ["", f"Blocks requiring special inspection: {len(flagged)}.", ""]
        for block in flagged:
            rows.append(f"- [ ] `{block['block_id']}` · {block.get('kind')} · {block['text'][:150]}")
    target = root / "inspection_report.md"
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return target
