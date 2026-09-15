"""Pinned local BGE encoding for comparison; no truncation or E5 impersonation.

BGE-M3 follows the BGE dense interface but has a multilingual 8,192-token
context and does not use the BGE v1.5 query instruction.  Keep those settings
explicit so changing the model cannot silently change the vector contract.
"""
from __future__ import annotations

import re
from .embedding import E5Encoder


class BGEEncoder(E5Encoder):
    def __init__(self, config, *, tokenizer_only=False):
        from transformers import AutoModel, AutoTokenizer
        self.config = dict(config)
        self.model_id = config['model']
        self.revision = config['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise ValueError('BGE requires an immutable revision')
        model_source = config.get('local_path') or self.model_id
        options = dict(revision=self.revision, local_files_only=True,
                       trust_remote_code=False, cache_dir=config.get('cache_dir'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **options)
        resolved_tokenizer = self.tokenizer.init_kwargs.get('_commit_hash')
        if resolved_tokenizer and resolved_tokenizer != self.revision:
            raise ValueError('Loaded BGE tokenizer revision differs from the pinned revision')
        self.model = None
        if not tokenizer_only:
            import torch
            torch.set_num_threads(int(config.get('threads', 4)))
            self.model = AutoModel.from_pretrained(model_source, **options).to('cpu').eval()
            resolved_model = getattr(self.model.config, '_commit_hash', None)
            if resolved_model and resolved_model != self.revision:
                raise ValueError('Loaded BGE model revision differs from the pinned revision')
        self.is_m3 = self.model_id.lower().rstrip('/').endswith('bge-m3')
        self.max_tokens = 8192 if self.is_m3 else 512
        self.query_prefix = '' if self.is_m3 else 'Represent this sentence for searching relevant passages: '
        self.provenance = dict(model=self.model_id, revision=self.revision, device='cpu',
            query_prefix=self.query_prefix, passage_prefix='', pooling='CLS + L2', dtype='float32',
            max_tokens=self.max_tokens, dimensions=int(self.model.config.hidden_size) if self.model else None,
            encoder='bge')

    def _encode(self, texts, prefix):
        import numpy as np
        import torch
        if self.model is None:
            raise RuntimeError('Tokenizer-only encoder cannot generate vectors')
        prepared = [prefix + text for text in texts]
        for i, text in enumerate(prepared):
            count = len(self.tokenizer.encode(text, add_special_tokens=True))
            if count > self.max_tokens:
                raise ValueError(f'Embedding input {i} has {count} tokens; limit={self.max_tokens}; no truncation')
        arrays = []
        for start in range(0, len(prepared), int(self.config.get('batch_size', 8))):
            batch = self.tokenizer(prepared[start:start + int(self.config.get('batch_size', 8))],
                                   padding=True, truncation=False, return_tensors='pt')
            with torch.inference_mode():
                cls = self.model(**batch).last_hidden_state[:, 0]
                arrays.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
        return np.concatenate(arrays) if arrays else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

    def encode_documents(self, texts):
        return self._encode(texts, '')

    def encode_queries(self, texts):
        return self._encode(texts, self.query_prefix)


def comparison_encoder(config, *, tokenizer_only=False):
    name = config.get('encoder', 'bge')
    if name == 'bge':
        return BGEEncoder(config, tokenizer_only=tokenizer_only)
    if name == 'e5':
        return E5Encoder(config, tokenizer_only=tokenizer_only, local_only=True)
    if name == 'qwen3_remote':
        from .comparison_qwen_embedding import QwenEmbeddingEncoder
        return QwenEmbeddingEncoder(config, tokenizer_only=tokenizer_only)
    raise ValueError(f'Unknown comparison encoder: {name}')
