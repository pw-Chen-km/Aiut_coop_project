# Navigation RAG A Service

This is the production handoff package for the source-hierarchy (A) navigation skill. It is independent from the research repository and contains no B/C artifacts.

Set the model endpoint, model digest, tokenizer files and embedding worker settings in `config/qa_agent.toml`, then run `./run.ps1`. The browser service listens on `8080`; the Qwen model endpoint must be a separate port such as `11440`. Open `http://127.0.0.1:8080` for the browser test console. Each message shows its timestamp and the answer's elapsed time. The service selects the runtime that matches the bundle's pinned Qwen embedding identity automatically.

The stable HTTP contract is `POST /v1/answer` with `{qid, question, history?}` and the existing qa-agent result JSON. Health endpoints are `/health/live` and `/health/ready`.

Validate the immutable skill bundle before deployment:

```powershell
$env:PYTHONPATH='src'; python -m qa_agent.cli validate-bundle --bundle-dir skill-bundle
```

Qwen generation and Qwen3-Embedding-8B are provisioned by the company server; weights and credentials are intentionally excluded.

The `skill-bundle` directory is the private A artifact and must be transferred with this repository. Future navigation algorithms can be added behind the service adapter while keeping `/v1/answer` unchanged.
