# Jev decision scoring implementation

The classifier assigns a topic distribution to the user query. It does not answer
the user's manual question. Answers are generated in a later, separate step from
retrieved evidence. The taxonomy uses consecutive numeric strings `1` through `10`
consistently in the prompt, annotations and output dictionaries.

## Prompt and score

`prompts/query_topic_jev.txt` contains the task, user query and numeric topic list.
The server applies its own model chat template with thinking disabled. The scorer
then appends the assistant prefix `Best answer: [`.

For option k, construct the **complete suffix** `k]`. Tokenize the full prompt plus
suffix and verify it preserves the exact prompt token prefix. If tokenization
merges across the boundary, stop with a scoring error; do not align by guessing.

For suffix token sequence t(k):

```text
L(k) = sum_j log P(t(k,j) | prompt, t(k,<j))
p(k) = exp(L(k) - max L) / sum_i exp(L(i) - max L)
```

This is the paper's joint candidate likelihood computation, including the closing
bracket. It is not the probability of only the first numeric token, and it is not
ordinary text generation of a JSON class. The output `probabilities` is constructed
by our code from likelihoods, for example `{"1": ..., "2": ..., "10": ...}`.
The model itself is never instructed to invent `T01` keys or numerical probabilities.

## HTTP transport

1. `/apply-template`: obtain the native rendered prompt with reasoning disabled.
2. `/tokenize`: obtain prompt and suffix token IDs; parse model special tokens.
3. Build a trie over complete candidate suffixes; reuse shared prefixes.
4. At each trie node call `/completion` with its prompt token IDs, `n_predict=1`,
   `temperature=-1`, `post_sampling_probs=false`, and `n_probs`.
5. Read raw `top_logprobs` by token ID, not by ambiguous decoded token text.
6. Sum per-suffix conditional log probabilities and softmax over all options.

`temperature=-1` follows llama.cpp's raw simple-softmax mode. Do not add grammar,
allowed-token constraints, logit bias or post-sampling filtering. Those would change
the distribution being measured. The one generated token per HTTP call is discarded;
the scorer uses the returned first-step probability distribution. Thus this backend
is not literally zero-decode, nor the paper's parallel GPU implementation.

Default `n_probs=256`; retry the same node with up to 1024 candidates if a required
token is absent. Every required token must have a finite, nonpositive log probability.
If the deployed server omits scores/IDs or still omits a required token, record a
scoring error. Do not insert zero probabilities or switch to JSON classification.

The HTTP trie saves duplicated prefix requests logically, but its nodes are
evaluated sequentially and cache reuse depends on the server. We reproduce the
scoring equation, **not optimized parallel KV-cache performance**. The trace stores
candidate token IDs, token log probabilities, joint scores, request count and wall
time. No live deployed server compatibility or speed claim has been validated yet.

## Tests and inspection points

- `tests/test_jev.py`: demonstrates that adding `]` can change candidate ordering;
  handles shared `1`/`10` prefixes; checks missing-score failures and raw HTTP fields.
- `src/topic_rag/topic_scoring.py`: `suffix_paths`, `score_trie`, `JevScorer`.
- `src/topic_rag/model_client.py`: native template and tokenization; rejects an
  unclosed `<think>` block and ambiguous multiple active models.

Reference: [LLM-as-Jev v3](https://arxiv.org/html/2610.02076v3), Sections 3.2–3.3 and
Appendix A; transport details follow the [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
