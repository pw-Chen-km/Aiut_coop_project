# Architecture

The runtime has two self-contained services: a FastAPI image containing the immutable SQLite/vector corpus and multilingual E5 model, and a llama.cpp image containing the Qwen GGUF. It has no mutable database and stores no conversation state.

For each request, the backend supplies a `conversation_id`, completed Q/A history, and the current question. Before applying history budgets or rewrite rules, the API drops entire Q/A pairs whose answer exactly matches `I can't answer this` after trimming and case folding. It retains normal answers and clarification follow-ups, keeping the newest complete pairs within fixed budgets. In `conditional` mode, a deterministic rule checks whether the current question contains a reference or omitted context such as “after that” or “the previous step”. Only then does the API call Qwen once to create a standalone search query. The original question remains a retrieval path.

For a successful rewrite request, BM25 and E5 retrieve top-20 candidates for both queries and combine them with weighted RRF using `rrf_k=60`. The rewritten query is primary, with 80% total weight (`rewrite_bm25=0.40`, `rewrite_e5=0.40`); the original question is secondary, with 20% total weight (`original_bm25=0.10`, `original_e5=0.10`). All weights are positive. The public endpoint exposes the final top-5 passages to Qwen. If rewriting is requested but usable history is missing or rewriting fails, the API returns a clarification question. In `disabled` mode, the previous current-question-plus-bounded-history retrieval query is preserved for baseline comparison.

The standard non-rewrite path uses BM25 top-20 plus E5 top-20, weighted RRF, and final top-5.

The API returns the supplied `conversation_id`, `response_type`, answer, and public source metadata. `answer` responses include cited sources; `clarify` returns a concrete follow-up question without sources; `abstain` returns `I can't answer this` without sources. The company backend persists that response and is responsible for concurrent request ordering and retry deduplication. To continue after `clarify`, it passes the clarification turn in the next request's context together with the user's reply.

Navigation skills, graph retrieval, NER, cloud fallback, multi-round retrieval, and server-side conversation storage are absent from the runtime.

PDF text, backend-supplied conversation history, and questions are untrusted input. The answer prompt prevents instructions in those inputs from changing system behavior.
