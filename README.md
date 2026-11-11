# Qursor RAG Algorithm Core

This repository contains the source code for the Qursor enterprise document QA algorithm.

Included:

- BM25 retrieval
- multilingual E5 dense retrieval integration
- weighted RRF fusion
- history filtering and query rewriting
- Qwen answer generation client and response validation
- PDF parsing, section-aware chunking and corpus build code
- unit tests and API documentation

Runtime assets are intentionally external to this repository. The following are not included:

- Qwen or E5 model files
- PDFs, SQLite corpus files and vector files
- Docker and Compose deployment files
- CI/CD configuration
- benchmark outputs and conversation data

The runtime expects a prepared corpus and local E5 model through `CORPUS_DIR` and
`E5_MODEL_DIR`. Qwen is accessed through `QWEN_ENDPOINT`.

The production retrieval path is:

```text
BM25 top-20 + E5 top-20
        ↓
weighted RRF (rrf_k=60)
        ↓
final top-5 passages
        ↓
Qwen answer generation
```

When history rewriting succeeds, the rewritten query has 80% of the RRF weight and
the original question has 20%.

This repository is the algorithm source. Deployment assets are maintained separately.
