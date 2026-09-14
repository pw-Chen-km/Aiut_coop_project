"""Hierarchical clustering with branching ratio p.

Builds a tree bottom-up:
  Level 0: n documents (leaves)
  Level 1: ceil(n/p) clusters of ~p documents each
  Level 2: ceil(n/p²) clusters of ~p level-1 clusters each
  ...
  Level t: stop when num_clusters <= max_top

At each level, clusters are formed by K-means on embeddings. Three
extensions are layered on top of the basic K-means partition:

  - **Soft assignment**: each item also gets an optional secondary parent
    when the gap between its top-2 centroid similarities is below a margin.
  - **Summary+raw embedding**: the caller-provided embed_fn also receives
    a ``context`` block of sampled member excerpts, so the embedding sees
    both the LLM-abstractive summary and surface-lexical signal.
  - **LLM repartition**: a single LLM hop per level flags confusable or
    mixed clusters and proposes a fresh partition by item IDs. Each
    cluster_id is repartitioned at most once per compile.

After clustering, an LLM summary is generated for each cluster, and that
summary (plus sampled member excerpts) is embedded as the data point for
the next level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ClusterNode:
    """A node in the hierarchy tree."""
    node_id: str
    level: int
    label: str = ""
    summary: str = ""
    embedding: np.ndarray | None = None
    children: list["ClusterNode"] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    doc_texts: list[str] = field(default_factory=list)

    # Soft assignment bookkeeping
    primary_parent: "ClusterNode | None" = None
    secondary_parent: "ClusterNode | None" = None
    # Children whose primary_parent is elsewhere but who appear here as stubs.
    secondary_children: list["ClusterNode"] = field(default_factory=list)

    # Entity tagging (filled by the entity-extraction pass).
    named_entities: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    related_skill_paths: list[dict] = field(default_factory=list)

    # Exemplar documents (doc_ids of representative leaves).
    exemplar_doc_ids: list[str] = field(default_factory=list)

    # Leaf-only: per-document summary cards {title, one_line, phrases}.
    doc_cards: dict[str, dict] = field(default_factory=dict)


def _kmeans_labels(normed: np.ndarray, k: int) -> np.ndarray:
    """Run K-means (or fall back to agglomerative) and return cluster labels."""
    from sklearn.cluster import KMeans, AgglomerativeClustering

    try:
        km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
        return km.fit_predict(normed)
    except Exception:
        agg = AgglomerativeClustering(n_clusters=k)
        return agg.fit_predict(normed)


def cluster_level_soft(
    embeddings: np.ndarray,
    k: int,
    min_size: int = 1,
    soft_margin: float = 0.05,
) -> tuple[list[list[int]], dict[int, int]]:
    """K-means partition with an optional secondary cluster per item.

    Returns:
        primary_assignments: list of cluster member-index lists (each item in
            exactly one primary cluster, same as classical K-means).
        secondary_assignments: dict[item_index -> secondary_cluster_index]. An
            item appears here only when the gap between its top-1 and top-2
            centroid cosine similarity is below ``soft_margin``.
    """
    n = embeddings.shape[0]
    k = min(k, n)
    if k <= 1:
        return [list(range(n))], {}

    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    labels = _kmeans_labels(normed, k)

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    primary = [v for v in clusters.values() if len(v) >= min_size]
    orphans: list[int] = []
    for v in clusters.values():
        if len(v) < min_size:
            orphans.extend(v)

    if not primary:
        primary = [list(range(n))]

    centroids = []
    for cl in primary:
        c_emb = normed[cl].mean(axis=0)
        centroids.append(c_emb / (np.linalg.norm(c_emb) + 1e-10))
    centroids_arr = np.array(centroids)

    for oidx in orphans:
        sims = centroids_arr @ normed[oidx]
        best = int(np.argmax(sims))
        primary[best].append(oidx)

    secondary: dict[int, int] = {}
    if soft_margin > 0 and len(primary) >= 2:
        # Recompute centroids after orphan re-attachment.
        centroids = []
        for cl in primary:
            c_emb = normed[cl].mean(axis=0)
            centroids.append(c_emb / (np.linalg.norm(c_emb) + 1e-10))
        centroids_arr = np.array(centroids)

        item_to_primary = {}
        for ci, members in enumerate(primary):
            for m in members:
                item_to_primary[m] = ci

        sims_all = normed @ centroids_arr.T  # (n, k)
        for i in range(n):
            row = sims_all[i]
            order = np.argsort(-row)
            top1, top2 = int(order[0]), int(order[1])
            pi = item_to_primary.get(i)
            if pi is None:
                continue
            if top1 != pi:
                # K-means split didn't pick the nearest centroid; treat the
                # nearest as the secondary signal.
                if row[pi] - row[top1] >= -soft_margin:
                    continue
                secondary[i] = top1
                continue
            gap = row[top1] - row[top2]
            if gap < soft_margin:
                secondary[i] = top2

    return primary, secondary


def cluster_level(embeddings: np.ndarray, k: int, min_size: int = 1) -> list[list[int]]:
    """Backward-compatible hard-partition wrapper."""
    primary, _ = cluster_level_soft(embeddings, k, min_size, soft_margin=0.0)
    return primary


def _sampled_member_excerpts(
    member_indices: list[int],
    current_nodes: list[ClusterNode],
    embeddings: np.ndarray,
    n_samples: int = 5,
    excerpt_chars: int = 400,
) -> str:
    """Pick the n_samples most-central members and join their summary heads.

    Used for context-aware cluster embedding: the embedding model sees
    both the abstractive summary and surface-lexical snippets from the
    actual member content, restoring discriminative tokens when summaries
    collapse onto near-identical wording.
    """
    if not member_indices:
        return ""
    member_embs = embeddings[member_indices]
    normed = member_embs / (np.linalg.norm(member_embs, axis=1, keepdims=True) + 1e-10)
    mean = normed.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-10)
    sims = normed @ mean
    order = np.argsort(-sims)
    top = [member_indices[int(j)] for j in order[: max(n_samples, 1)]]

    chunks = []
    for idx in top:
        node = current_nodes[idx]
        snippet = (node.summary or "").strip()[:excerpt_chars]
        if snippet:
            chunks.append(snippet)
    return "\n---\n".join(chunks)


def _aggregate_doc_content(
    children: list[ClusterNode],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """Walk descendants and collect (doc_ids, doc_texts, doc_cards)."""
    ids: list[str] = []
    texts: list[str] = []
    cards: dict[str, dict] = {}
    for child in children:
        ids.extend(child.doc_ids)
        texts.extend(child.doc_texts)
        if child.doc_cards:
            cards.update(child.doc_cards)
    return ids, texts, cards


def _apply_repartition(
    clusters: list[dict],
    changes: list[dict],
    current_nodes: list[ClusterNode],
    embeddings: np.ndarray,
    repartitioned_cluster_ids: set[str],
    level: int,
) -> tuple[list[dict], list[int]]:
    """Apply LLM-proposed repartition changes to ``clusters`` in-place-ish.

    Returns:
        new_clusters: the updated cluster list (clusters that survived plus
            new clusters produced by the partition).
        affected_new_indices: indices into new_clusters of clusters that
            need fresh summarization.
    """
    if not changes:
        return clusters, []

    id_to_cluster_idx = {c["cluster_id"]: i for i, c in enumerate(clusters)}
    node_id_to_global = {n.node_id: i for i, n in enumerate(current_nodes)}

    drop_indices: set[int] = set()
    new_clusters: list[dict] = []
    affected_new_indices: list[int] = []
    repart_counter = 0

    for change in changes:
        old_ids = change.get("old_cluster_ids") or []
        new_part = change.get("new_partition") or {}
        if not old_ids or not new_part:
            continue

        # Skip if any flagged cluster_id was already repartitioned in a
        # prior level (cap: once per compile).
        if any(oid in repartitioned_cluster_ids for oid in old_ids):
            continue
        if not all(oid in id_to_cluster_idx for oid in old_ids):
            continue

        union_indices: set[int] = set()
        for oid in old_ids:
            union_indices.update(clusters[id_to_cluster_idx[oid]]["member_indices"])
        union_node_ids = {current_nodes[i].node_id for i in union_indices}

        # Aggregate secondary memberships from the affected old clusters so
        # we can redistribute them to the new clusters by item identity.
        old_secondary_lookup: dict[int, list[str]] = {}
        for oid in old_ids:
            for sec_idx in clusters[id_to_cluster_idx[oid]].get(
                "secondary_member_indices", []
            ):
                old_secondary_lookup.setdefault(sec_idx, []).append(oid)

        new_assignments: list[tuple[str, list[int]]] = []
        seen_indices: set[int] = set()
        for new_label, raw_id_list in new_part.items():
            if not isinstance(raw_id_list, list):
                continue
            mapped: list[int] = []
            for raw in raw_id_list:
                if raw not in union_node_ids:
                    continue
                gi = node_id_to_global.get(raw)
                if gi is None or gi in seen_indices:
                    continue
                mapped.append(gi)
                seen_indices.add(gi)
            if mapped:
                new_assignments.append((new_label, mapped))

        leftover = union_indices - seen_indices
        if not new_assignments:
            continue
        if leftover:
            # Attach leftover items to the new cluster whose centroid they're
            # most similar to.
            normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
            new_centroids = []
            for _, mems in new_assignments:
                c_emb = normed[mems].mean(axis=0)
                new_centroids.append(c_emb / (np.linalg.norm(c_emb) + 1e-10))
            new_centroids_arr = np.array(new_centroids)
            for li in leftover:
                sims = new_centroids_arr @ normed[li]
                best = int(np.argmax(sims))
                new_assignments[best][1].append(li)

        for oid in old_ids:
            drop_indices.add(id_to_cluster_idx[oid])
            repartitioned_cluster_ids.add(oid)

        for new_label, mems in new_assignments:
            cid = f"L{level}-CR{repart_counter}"
            repart_counter += 1
            new_clusters.append({
                "cluster_id": cid,
                "member_indices": mems,
                "secondary_member_indices": [],
                "summary": None,
                "repartitioned_from": list(old_ids),
            })
            repartitioned_cluster_ids.add(cid)

    if not new_clusters:
        return clusters, []

    surviving = [c for i, c in enumerate(clusters) if i not in drop_indices]
    start = len(surviving)
    merged = surviving + new_clusters
    affected_new_indices = list(range(start, len(merged)))
    return merged, affected_new_indices


def build_hierarchy(
    doc_ids: list[str],
    doc_texts: list[str],
    embeddings: np.ndarray,
    p: int = 10,
    max_top: int = 10,
    min_cluster_size: int = 3,
    summarize_fn=None,
    summarize_batch_fn=None,
    embed_fn: Callable[..., np.ndarray] | None = None,
    soft_assignment: bool = True,
    soft_margin: float = 0.05,
    repartition_fn: Callable[[list[dict], int], dict] | None = None,
    doc_cards: dict[str, dict] | None = None,
) -> list[ClusterNode]:
    """Build a multi-level cluster hierarchy.

    Args:
        doc_ids: document identifiers
        doc_texts: document text content (full text used for the leaf-level
            "summary + raw_text" embedding when ``embed_fn`` accepts a
            ``context=`` keyword).
        embeddings: (n, dim) array of document embeddings (already aware of
            doc_cards when the caller embeds ``concat(card_summary, raw_text)``).
        p: branching ratio — each cluster has ~p children
        max_top: stop when num_clusters <= max_top
        min_cluster_size: minimum items per cluster
        summarize_fn: callable(texts, level) -> str
        summarize_batch_fn: callable(batch, level) -> list[str]
        embed_fn: callable(text: str, context: str | None = None) -> ndarray.
            Internal clusters get ``context=`` filled with sampled member
            excerpts so the cluster embedding sees both summary and content.
        soft_assignment: enable soft / multi-parent K-means.
        soft_margin: gap threshold for triggering a secondary parent.
        repartition_fn: callable([cluster_record, ...], level) -> {"changes": [...]}.
            Each cluster_id is repartitioned at most once across the whole
            compile.
        doc_cards: {doc_id: {title, one_line, phrases}} from the per-document
            card stage; stored on the leaf ClusterNodes for downstream use
            (rich INDEX rows, exemplar listings).
    """
    n = len(doc_ids)
    assert n == embeddings.shape[0]

    leaf_nodes = []
    for i, (did, txt) in enumerate(zip(doc_ids, doc_texts)):
        card = (doc_cards or {}).get(did)
        leaf_summary = txt[:500]
        if card:
            from corpus2skill.summarizer import card_to_summary
            leaf_summary = card_to_summary(card) or leaf_summary
        leaf = ClusterNode(
            node_id=f"doc-{did[:12]}",
            level=0,
            label=did,
            summary=leaf_summary,
            embedding=embeddings[i],
            doc_ids=[did],
            doc_texts=[txt],
        )
        if card:
            leaf.doc_cards = {did: card}
        leaf_nodes.append(leaf)

    current_nodes = leaf_nodes
    current_embeddings = embeddings.copy()
    level = 0
    repartitioned_cluster_ids: set[str] = set()

    while len(current_nodes) > max_top:
        level += 1
        k = max(max_top, math.ceil(len(current_nodes) / p))
        k = min(k, len(current_nodes))

        print(f"  Level {level}: clustering {len(current_nodes)} items → {k} clusters")

        if soft_assignment:
            prim_assign, sec_assign = cluster_level_soft(
                current_embeddings, k, min_cluster_size, soft_margin
            )
        else:
            prim_assign = cluster_level(current_embeddings, k, min_cluster_size)
            sec_assign = {}

        if not prim_assign:
            break

        clusters: list[dict] = []
        for ci, members in enumerate(prim_assign):
            clusters.append({
                "cluster_id": f"L{level}-C{ci}",
                "member_indices": list(members),
                "secondary_member_indices": [],
                "summary": None,
            })
        for item_idx, parent_ci in sec_assign.items():
            if 0 <= parent_ci < len(clusters):
                clusters[parent_ci]["secondary_member_indices"].append(item_idx)

        # ---- summarize primary members ---------------------------------
        def _summarize(clusters_to_run: list[dict]) -> None:
            if not clusters_to_run:
                return
            inputs = [
                [current_nodes[i].summary for i in c["member_indices"]]
                for c in clusters_to_run
            ]
            if summarize_batch_fn:
                summaries = summarize_batch_fn(inputs, level=level)
            elif summarize_fn:
                summaries = [summarize_fn(si, level=level) for si in inputs]
            else:
                summaries = [
                    f"Cluster of {len(c['member_indices'])} items." for c in clusters_to_run
                ]
            for c, s in zip(clusters_to_run, summaries):
                c["summary"] = s

        print(f"    Summarizing {len(clusters)} clusters concurrently...")
        _summarize(clusters)

        # ---- LLM verify+repartition ------------------------------------
        if repartition_fn is not None:
            eligible = [c for c in clusters if c["cluster_id"] not in repartitioned_cluster_ids]
            records = []
            for c in eligible:
                if not c.get("summary"):
                    continue
                records.append({
                    "cluster_id": c["cluster_id"],
                    "label": c["cluster_id"],
                    "summary": c["summary"],
                    "members": [
                        {
                            "id": current_nodes[i].node_id,
                            # First non-empty line of the child summary is the
                            # de-facto title; _build_clusters_block in
                            # summarizer applies a final title_chars cap.
                            "title": next(
                                (
                                    ln.strip()
                                    for ln in (current_nodes[i].summary or "").splitlines()
                                    if ln.strip()
                                ),
                                "",
                            ),
                        }
                        for i in c["member_indices"][:12]
                    ],
                })
            if records:
                try:
                    print(f"    Repartition pass over {len(records)} clusters...")
                    out = repartition_fn(records, level)
                    changes = (out or {}).get("changes") or []
                    if changes:
                        print(f"      LLM proposed {len(changes)} repartition change(s)")
                    clusters, affected = _apply_repartition(
                        clusters, changes, current_nodes,
                        current_embeddings, repartitioned_cluster_ids, level,
                    )
                    if affected:
                        _summarize([clusters[i] for i in affected])
                except Exception as e:
                    print(f"      repartition skipped: {e}")



        # ---- build next-level nodes -----------------------------------
        next_nodes: list[ClusterNode] = []
        next_embeddings: list[np.ndarray] = []
        for ci, c in enumerate(clusters):
            children = [current_nodes[i] for i in c["member_indices"]]
            secondary_children = [
                current_nodes[i] for i in c["secondary_member_indices"]
            ]
            all_doc_ids, all_doc_texts, all_cards = _aggregate_doc_content(children)

            summary = c["summary"] or f"Cluster of {len(c['member_indices'])} items."
            if embed_fn:
                ctx = _sampled_member_excerpts(
                    c["member_indices"], current_nodes, current_embeddings
                )
                try:
                    emb = embed_fn(summary, context=ctx)
                except TypeError:
                    emb = embed_fn(summary)
            else:
                member_embs = current_embeddings[c["member_indices"]]
                emb = member_embs.mean(axis=0)
                emb = emb / (np.linalg.norm(emb) + 1e-10)

            node_id = f"L{level}-C{ci}"
            node = ClusterNode(
                node_id=node_id,
                level=level,
                summary=summary,
                embedding=emb,
                children=children,
                secondary_children=secondary_children,
                doc_ids=all_doc_ids,
                doc_texts=all_doc_texts,
            )
            if all_cards:
                node.doc_cards = all_cards
            for ch in children:
                ch.primary_parent = node
            for sch in secondary_children:
                # Only set if no explicit secondary parent yet (don't overwrite).
                if sch.secondary_parent is None:
                    sch.secondary_parent = node
            next_nodes.append(node)
            next_embeddings.append(emb)

        if not next_nodes:
            break

        current_nodes = next_nodes
        current_embeddings = np.array(next_embeddings, dtype=np.float32)
        print(
            f"    → {len(current_nodes)} clusters (docs per cluster: "
            f"min={min(len(n.doc_ids) for n in current_nodes)}, "
            f"max={max(len(n.doc_ids) for n in current_nodes)}, "
            f"mean={sum(len(n.doc_ids) for n in current_nodes)/len(current_nodes):.0f})"
        )

    return current_nodes


def tree_stats(roots: list[ClusterNode]) -> dict:
    """Compute statistics about the hierarchy tree."""
    def _depth(node: ClusterNode) -> int:
        if not node.children:
            return 0
        return 1 + max(_depth(c) for c in node.children)

    def _count_nodes(node: ClusterNode) -> int:
        return 1 + sum(_count_nodes(c) for c in node.children)

    max_depth = max(_depth(r) for r in roots)
    total_nodes = sum(_count_nodes(r) for r in roots)
    total_docs = sum(len(r.doc_ids) for r in roots)
    total_secondary = 0
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            total_secondary += len(n.secondary_children)
            stack.extend(n.children)

    return {
        "num_top_clusters": len(roots),
        "max_depth": max_depth,
        "total_nodes": total_nodes,
        "total_documents": total_docs,
        "total_secondary_stubs": total_secondary,
    }
