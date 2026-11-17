# Phase-one experiment protocol

## Question and scope

Does query-topic information improve evidence retrieval and grounded answers
relative to BM25+E5 RRF on the same E-Manual passage pool? Compare hard filtering
with soft reranking without changing the answer model. This phase is single-turn.
History dependency detection, rewriting, LambdaMART, production APIs and deployment
are outside this experiment.

The prepared test pool contains 66 distinct questions, not 132 independent
questions. Keep all 66 in every method, including hard-selection failures. Do not
remove questions because a method fails. Two malformed raw records are excluded
before any method runs, solely because their evidence IDs do not exist in the
provided sentence map. Duplicate valid records still cover those two questions.

The data supplies 538 distinct evidence associations across questions. Several
questions have multiple model responses; retain every valid reference and union
the annotated relevant evidence. Preserve source IDs and exact character offsets.
Do not require an invented section hierarchy, and do not repair evidence with
fuzzy text matching. Citation evaluation uses supplied passages, not PDF pages.

## Fixed variables

- Qwen 4B served checkpoint, quantization, model identity and server process.
- Source pool, chunks, labels, tokenizer and E5 model hashes.
- Chunk budget: at most 384 E5 tokens including title and `passage:` prefix.
- BM25: k1=1.2, b=0.75, fixed corpus-wide IDF; same tokenizer in all conditions.
- Candidates: top-20 BM25 and top-20 E5; final answer context: top-5.
- RRF smoothing: 60; baseline/hard weights 0.5/0.5; soft weights exactly 1/3 each.
- Answer prompt, temperature=0, seed=0, thinking disabled, max answer 512 tokens.
- Context limit configured as 8192; overflow is recorded as an error, never
  silently truncates retrieved passages. This config must match the deployed model.

## Topic annotations and decision

Ten labels were derived by reading the source corpus; sources have one or more
labels. Chunks inherit their source labels. Label reasons, excerpts and source
hashes are saved. These annotations are assistant-reviewed, not independent human
ground truth. Questions do not yet have independently verified topic labels.

The query task is a **single primary-topic categorical distribution**. Multiple
plausible topics can share probability mass, but this is not ten independent
binary probabilities. Hard selects the highest-probability topic; ties use numeric
option order. Soft sums each chunk's associated topic probabilities.

For 10 options, option `10]` may have more tokens than `1]`. Preserve the full
unnormalized sequence likelihood used by the paper. There is no per-token length
normalization, temperature calibration, or option-order ensemble in this phase.
Record these as limitations rather than quietly changing the method.

## Conditions

1. Baseline: global BM25/E5, union candidates, two-way RRF.
2. Hard: keep chunks whose labels contain argmax topic, then retrieve/rank in the
   permitted set. No fallback to global retrieval; an empty set yields abstain.
   Global BM25 statistics remain fixed, so the experiment changes the search mask
   rather than changing both the search mask and IDF.
3. Soft: global candidates identical to baseline. Compute topic mass, rank it
   within the union, then three-way RRF. Missing lexical/dense ranks contribute
   zero. Tied topic masses use competition ranks (1,1,3...), and final ties use
   chunk ID. Topic weighting cannot recover a source outside this union.

The soft weights sum to exactly 1. Use `1/3`, not a rounded stored `0.333`.
Broad multi-label sources gain mass by design: 51 sources have one label, 39 have
two, and 12 have three or more. This is a possible confound; report results by
label cardinality before claiming topic semantics caused the improvement.

## Execution and fairness

The runner uses one query E5 vector and one Jev distribution per question. The
same distribution is reused for hard and soft; its measured cost is attributed to
both methods' standalone latency. Randomize method order per question with a
fixed seed. Record server identity before each question and at completion.
This checks model ID/path and exposed server metadata, not the remote GGUF bytes.
For reproducible runs, the operator must also pin the model image/checkpoint;
replacing weights behind an unchanged alias is not detectable from these endpoints.

Only corpus embeddings have been built. No query embeddings, Jev decisions or
Qwen benchmark answers have been generated during preparation. Unit/integration
tests use synthetic vectors and fake model replies.

After approval, first run retrieval-only, then full generation. Never tune labels,
prompts or weights using these test outcomes and report the rerun as untouched
test performance. For subsequent tuning, introduce a separate development split.
Any broken server scoring support must be fixed explicitly; a generated JSON label
is not an acceptable silent replacement for candidate likelihood scoring.

## Metrics and answer review

- Evidence recall@5/@10/@20: fraction of annotated relevant sentence spans fully
  covered by retrieved chunks. Adjacent selected chunks may jointly cover a span.
- Source recall@5/@10/@20: fraction of annotated source IDs retrieved; less strict
  than evidence recall and reported under its own name.
- Any-hit and complete-evidence question rates at 5.
- Answer/clarify/abstain counts and errors.
- Query embedding, topic scoring, retrieval, answer and attributed total latency.

The JSONL exports full references, evidence and generated answers. Review answers
against actual source evidence and the question; reference responses with false
adherence are not truth. Suggested rubric: 2=correct and sufficiently complete;
1=partly correct or materially incomplete; 0=wrong, unsupported, or abstains despite
sufficient evidence. Mark genuinely unjudgeable questions separately with reasons.
For a confirmed answerable set, report strict accuracy as `count(score=2)/N` and
partial-credit separately. Do not silently drop model failures from the denominator.
Blind method/model labels during review. Accuracy stays null until that review is
performed; lexical F1 is never used as accuracy.

Use paired question-level comparisons; report per-question changes, not just a
mean. Confidence intervals and repeat timing runs can be added after the first
approved run. First-pass timings include variable server/cache effects; this HTTP
reference implementation cannot establish the paper's latency advantage.

## Follow-up phases

After this controlled comparison, consider (separately): generated-label classifier
versus Jev efficiency; chunk-specific rather than inherited labels; topic-only
candidate expansion; out-of-domain queries; conversational dependency/rewriting.
None is mixed into the present baseline comparison.
