"""Version-pinned, CPU-only learned E5 encoder. No hashing fallback."""
from __future__ import annotations

import re
from typing import Sequence


class E5Encoder:
    def __init__(self, config: dict, *, local_only: bool = False, tokenizer_only: bool = False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the doc2skill optional dependencies before embedding") from exc
        self.config = config
        self.model_id = config.get("model", "intfloat/multilingual-e5-small")
        requested_revision = config.get("revision") or "main"
        if local_only and not re.fullmatch(r"[0-9a-f]{40}", requested_revision):
            raise ValueError("Offline inference requires the resolved 40-character embedding revision from the build")
        options = {"revision": requested_revision, "local_files_only": local_only,
                   "trust_remote_code": False, "cache_dir": config.get("cache_dir")}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
        self.revision = self.tokenizer.init_kwargs.get("_commit_hash") or requested_revision
        self.model = None
        if not tokenizer_only:
            import torch
            torch.set_num_threads(int(config.get("threads", 4)))
            self.model = AutoModel.from_pretrained(self.model_id, **options).to("cpu").eval()
            self.revision = getattr(self.model.config, "_commit_hash", None) or self.revision
        if not re.fullmatch(r"[0-9a-f]{40}", str(self.revision)):
            # Tokenizer metadata on some Transformers versions omits the commit.
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(self.model_id, "tokenizer_config.json", revision=requested_revision,
                                           cache_dir=config.get("cache_dir"))
            if isinstance(cached, str):
                from pathlib import Path
                self.revision = Path(cached).parent.name
        if not re.fullmatch(r"[0-9a-f]{40}", str(self.revision)):
            raise ValueError("Cannot establish immutable embedding/tokenizer revision")
        self.provenance = {"model": self.model_id, "revision": self.revision, "device": "cpu",
                           "query_prefix": "query: ", "passage_prefix": "passage: ",
                           "pooling": "attention-mask mean + L2", "dtype": "float32"}

    def _encode(self, texts: Sequence[str], prefix: str):
        import numpy as np
        import torch
        if self.model is None:
            raise RuntimeError("Tokenizer-only instance cannot generate embeddings")
        prepared = [text if text.startswith(prefix) else prefix + text for text in texts]
        for text in prepared:
            if len(self.tokenizer.encode(text, add_special_tokens=True)) > 512:
                raise ValueError("Embedding input exceeds 512 tokens; truncation is forbidden")
        arrays = []
        batch_size = int(self.config.get("batch_size", 16))
        for offset in range(0, len(prepared), batch_size):
            batch = self.tokenizer(prepared[offset:offset + batch_size], padding=True, truncation=False,
                                   return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                arrays.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(arrays) if arrays else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]):
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts: Sequence[str]):
        return self._encode(texts, "query: ")
