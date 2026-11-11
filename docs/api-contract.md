# API contract

The company backend owns user authentication, conversation persistence, request ordering, and idempotency. The QA service is stateless and requires the backend to provide the relevant completed Q/A history on every request.

## Answer a question

```http
POST /v1/qa
Content-Type: application/json
```

```json
{
  "conversation_id": 456,
  "context": [
    {
      "Q": "What is Kanban?",
      "A": "Kanbans consist of one or more steps..."
    }
  ],
  "question": "How can I verify that a step was completed?"
}
```

Input requirements:

- `conversation_id`: required JSON integer. Numeric strings and booleans are rejected.
- `context`: required array; `[]` is valid for the first question.
- `context[].Q`: required non-blank string, maximum 20,000 characters.
- `context[].A`: required non-blank string, maximum 20,000 characters.
- `question`: required non-blank string, maximum 20,000 characters.
- Field names are case-sensitive. Unknown fields are rejected.

The service trims surrounding whitespace and excludes entire Q/A pairs whose `A` equals `I can't answer this`, using an exact case-insensitive comparison. Filtering happens before history budgets or rewrite-trigger checks. These pairs are not passed to query rewriting, retrieval-history selection, or answer generation. Other answers and clarification questions are converted into ordered user/assistant messages; the newest complete pairs are kept within the configured budgets. The current `question` is always preserved.

When `QUERY_REWRITE_MODE=conditional` (the default), the service uses a small deterministic list of reference and ellipsis patterns to decide whether the history is needed to form a search query. A triggered request may make one additional internal Qwen call to produce a standalone query. This query is used only for retrieval; the original question remains the answer question and the public response is unchanged. Rewrite failure falls back to retrieval with the original question.

Response:

```json
{
  "conversation_id": 456,
  "response_type": "answer",
  "answer": "A step is complete after all jobs in its sequence have executed...",
  "sources": [
    {
      "document": "Qursor-MAP-userguide-Kanbans-20250121",
      "pages": [18],
      "excerpt": "After completing a step..."
    }
  ]
}
```

The public response contains exactly `conversation_id`, `response_type`, `answer`, and `sources`. `response_type` is one of `answer`, `clarify`, or `abstain`:

- `answer`: return the generated document-supported answer and its sources.
- `clarify`: return a concrete follow-up question in the user's language, with `sources: []`.
- `abstain`: return exactly `I can't answer this`, with `sources: []`.

For `answer`, `sources` includes only passages actually cited by Qwen, sorted by retrieval RRF score from highest to lowest. Equal scores preserve retrieval rank. The order does not follow Qwen's citation list. Each source still contains only `document`, `pages`, and `excerpt`; scores are not exposed and passages from the same PDF are not merged.

Clarification example:

```json
{
  "conversation_id": 456,
  "response_type": "clarify",
  "answer": "Which Qursor service do you want to restart?",
  "sources": []
}
```

The service returns the follow-up and waits; it does not initiate another request itself. When the user replies, the backend includes the clarification Q/A pair in `context` and supplies the reply as the new `question`. The input remains `{conversation_id, context, question}`; `context` items still accept only `Q` and `A`.

Abstention example:

```json
{
  "conversation_id": 456,
  "response_type": "abstain",
  "answer": "I can't answer this",
  "sources": []
}
```

Integration change: `response_type` is a new public response field, and `clarify` no longer returns the fixed refusal text. Consumers should branch on `response_type`, not infer the status only from `sources` being empty. Store the status with the response if needed; no server-side conversation storage is added.

The service does not persist any input or output. Reusing a `conversation_id` does not restore prior history.

## Internal top-k ablation endpoint

For server-side benchmark runs only, the RAG image also exposes:

```http
POST /internal/ablation/qa?final_k=2
Content-Type: application/json
```

The JSON body and response are identical to `/v1/qa`. `final_k` is a required
integer from `1` to `20` and is applied only to that request. The public
`/v1/qa` endpoint remains fixed at final top-5. The endpoint does not enable
model switching, persist results, or restart containers; the external runner
must record its own results. It is intended for access from the deployment
network, not as a general public API.

## Health

```http
GET /healthz
GET /readyz
```

`/readyz` verifies that the immutable corpus, E5 runtime, and local Qwen server are available. It includes the active corpus version.

## Errors

Errors use FastAPI's JSON format:

```json
{"detail": "QA request failed: ..."}
```

- `422`: missing, blank, incorrectly cased, or unknown request fields.
- `502`: local retrieval or Qwen generation failed.
- `503`: the corpus, E5 runtime, or Qwen server is not ready.

Generation/format failures remain errors; they are not normalized into `abstain` or `clarify`.

The former conversation POST, GET, and DELETE endpoints are removed and return HTTP 404.
