# Corpus2Skill core provenance

Source: https://github.com/dukesun99/Corpus2Skill

Commit: `b0108ce22d434983cf97abc2ca455aebc6b625dd`

`clustering.upstream.py` is the unchanged source used for parity tests.
`clustering.py` is that source with the following explicit comparison changes:

1. Require full initial summaries, removing the implicit 500-character leaf
   truncation. Replace upstream document cards with shared comparison descriptions.
2. Require summary and embedding callbacks. Never fall back to a generic summary,
   centroid embedding, a callback lacking child context, or agglomerative clustering.
3. Add a member-node summary callback so duplicate descriptions retain separate
   identities without putting IDs or source paths into embedding input. The original
   string and batch callback interfaces remain for parity tests.
4. Check aligned nonempty input, complete unique assignments, and level reduction.
5. Fix an upstream membership bug: when every K-means group is smaller than
   `min_cluster_size`, upstream first puts every index in one group, then appends
   every orphan index again. The adapted source clears the orphan list in that
   branch; the intended single group contains each index exactly once. Parity tests
   cover unchanged normal behavior and document this intentional divergence.

The K-means settings (`n_init=3`, `max_iter=100`, `random_state=42`), normalization,
small-group nearest-centroid assignment, level schedule, and selection of the five
most-central child descriptions with 400-character excerpts remain upstream.
The adapter executes this actual vendored core. Soft assignment (margin 0.05)
and LLM repartition are now enabled by default. `repartition_prompt.py` contains
the unchanged `_REPARTITION_PROMPT` and `_build_clusters_block` extracted from
`corpus2skill/summarizer.py` at the same pinned commit. The local model transport
uses this prompt with schema validation and at most one output repair. Unlike
upstream, errors are propagated rather than silently skipping failed requests.
The upstream 60-cluster skip and 12 displayed members per cluster are retained
and audited. Unshown members use upstream nearest-centroid leftover assignment.
Shown members must be assigned exactly once; overlapping changes are rejected.

Repartition is completed before attaching final navigation parents, so discarded
preliminary summaries cannot leave orphan groups or duplicate body ownership.
The upstream routine clears secondary links on replaced groups; this behavior
is retained (no extra reclustering or inferred links). Secondary links that
survive are exported as `secondary_children` pointing to canonical nodes.
Primary ownership is still unique; navigation scopes include all reachable
primary and secondary content, deduplicated. The common navigator can follow
both types of entrance using its existing descend/retrieve operations.

This remains a core adaptation, not a complete reproduction of the upstream
compiler, document-card generation, entity index, related links or skill writer.

The accompanying MIT license applies to both copies of the upstream source.
