# Navigation RAG A Service

This is the production handoff package for the source-hierarchy (A) navigation skill. It is independent from the research repository and contains no B/C artifacts.

Set the model endpoint, model digest, tokenizer files and local embedding model in `config/qa_agent.toml`, then run `./run.ps1`. The checked Qwen 27B digest and 131,072-token context from the A build are filled in; replace them if the company server exposes a different model build. The browser service listens on `8080`; the Qwen model endpoint must be a separate port such as `11440`. Open `http://127.0.0.1:8080` for the browser test console. Each message shows its timestamp and the answer's elapsed time. The service selects the comparison runtime for the pinned BGE-M3 embedding bundle automatically.

The stable HTTP contract is `POST /v1/answer` with `{qid, question, history?}` and the existing qa-agent result JSON. Health endpoints are `/health/live` and `/health/ready`.

Validate the immutable skill bundle before deployment:

```powershell
$env:PYTHONPATH='src'; python -m qa_agent.cli validate-bundle --bundle-dir skill-bundle
```

Qwen generation is provisioned by the company server. BGE-M3 is a local Hugging Face model and must be downloaded or copied into the configured cache before starting the service; model weights and credentials are intentionally excluded.

For a machine with Hugging Face access, provision the pinned embedding files with:

```powershell
hf download BAAI/bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181 --local-dir models/bge-m3
```

The `skill-bundle` directory is the private A artifact and must be transferred with this repository. Future navigation algorithms can be added behind the service adapter while keeping `/v1/answer` unchanged.
