"""Environment-only service configuration."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    e5_model_dir: Path
    qwen_endpoint: str
    qwen_model: str
    retrieval_history_chars: int = 2400
    answer_history_chars: int = 9000
    query_rewrite_mode: str = "conditional"
    query_rewrite_timeout_seconds: float = 30.0
    query_rewrite_max_tokens: int = 96
    qwen_timeout_seconds: float = 120.0
    qwen_max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "Settings":
        rewrite_mode = os.getenv("QUERY_REWRITE_MODE", "conditional").strip().lower()
        if rewrite_mode not in {"conditional", "disabled"}:
            raise ValueError("QUERY_REWRITE_MODE must be conditional or disabled")
        return cls(
            corpus_dir=Path(os.getenv("CORPUS_DIR", "/app/data/corpus/active")),
            e5_model_dir=Path(os.getenv("E5_MODEL_DIR", "/app/models/e5/multilingual-e5-small")),
            qwen_endpoint=os.getenv("QWEN_ENDPOINT", "http://host.docker.internal:8073/v1"),
            qwen_model=os.getenv("QWEN_MODEL", "Qwen3.5-4B-Q4_K_M"),
            retrieval_history_chars=int(os.getenv("RETRIEVAL_HISTORY_CHARS", "2400")),
            answer_history_chars=int(os.getenv("ANSWER_HISTORY_CHARS", "9000")),
            query_rewrite_mode=rewrite_mode,
            query_rewrite_timeout_seconds=float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "30")),
            query_rewrite_max_tokens=int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "96")),
            qwen_timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", "120")),
            qwen_max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "512")),
        )
