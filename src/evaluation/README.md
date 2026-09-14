# Answer and citation evaluation

This layer evaluates generated answers independently of the existing retrieval metrics. It is deterministic, dependency-free, and never calls an LLM.

## Prediction JSONL contract

Each line follows `schemas/predictions.schema.json` and has this shape:

```json
{"schema_version":"qursor-prediction-v1","qid":"qursor:p1:test:q000001","answer":"The flash icon.","citations":[{"evidence_id":"ev:flash"}],"claims":[{"claim_id":"p1","text":"The flash icon opens the options.","correctness":{"label":true,"review_status":"reviewed","reviewers":["sme-01"]},"entailment":{"label":true,"review_status":"reviewed","reviewers":["sme-02"]}}]}
```

`answer`, `citations`, and `claims` are required; the arrays may be empty. A claim judgment has `review_status=pending|reviewed`. Only a boolean label with `review_status=reviewed` and at least one reviewer is scored. A provisional label under `pending` is retained for diagnostics but ignored as truth.

## Metrics

- `normalized_em`: case-folded, Unicode-normalized exact match after punctuation, English articles, and extra whitespace are removed. The best canonical-answer/alias match is used.
- `token_f1`: bag-of-token F1, maximized across the canonical answer and aliases.
- `claim_correctness`: fraction of reviewed claim correctness labels that are true.
- `citation_precision`, `citation_recall`, `citation_f1`: evidence-ID scoring against the best allowed gold evidence set, including declared equivalence groups.
- `faithfulness`: fraction of reviewed claim entailment labels that are true.
- `joint_success@K`: `normalized_em == 1` and the existing retrieval report has `complete_evidence_all_hops@K == 1` for the same `qid`.

If no reviewed correctness or entailment labels exist, the corresponding score is `null` and its denominator is zero. Per-query `denominators` and aggregate denominators are always reported. Claim correctness and faithfulness aggregate over reviewed labels; the other metrics macro-average over queries with defined scores.

Joint Success only reads an existing retrieval metrics report. It does not recompute or modify retrieval metrics. If the report or its `complete_evidence_all_hops@K` value is missing, Joint Success is `null` with denominator zero.

## CLI

```bash
python -m evaluation.evaluate_answers \
  --qa data/phase1_candidate/qa.jsonl \
  --predictions runs/generator.predictions.jsonl \
  --retrieval-report reports/metrics/hybrid.metrics.json \
  --joint-cutoff 10 \
  --output reports/metrics/generator.answer-citation.json
```

After installation, the same command is available as `qegs-evaluate-answers`. Omit `--retrieval-report` when answer/citation metrics are needed without Joint Success.
