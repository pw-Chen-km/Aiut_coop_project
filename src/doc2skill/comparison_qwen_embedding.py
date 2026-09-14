"""Pinned Qwen3 embeddings through a private, length-checking worker."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

import numpy as np


class QwenEmbeddingEncoder:
    def __init__(self, config, *, tokenizer_only=False):
        from transformers import AutoTokenizer
        self.config = dict(config)
        self.revision = config['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise ValueError('Qwen embedding requires an immutable revision')
        self.endpoint = config['endpoint'].rstrip('/')
        from .llm import is_loopback_endpoint
        if not is_loopback_endpoint(self.endpoint):
            raise ValueError('Embedding worker must use a private loopback/SSH endpoint')
        info = self._request('/info')
        self.provenance = info['embedding']
        if (self.provenance['model'] != config['model'] or
                self.provenance['revision'] != self.revision or
                self.provenance['max_tokens'] != 32768 or
                self.provenance['dimensions'] != 4096):
            raise ValueError('Remote embedding identity/capacity mismatch')
        directory = Path(config['tokenizer_dir'])
        if not {'tokenizer.json', 'tokenizer_config.json', 'config.json'} <= info['tokenizer_hashes'].keys():
            raise ValueError('Embedding tokenizer identity is incomplete')
        for name, digest in info['tokenizer_hashes'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Local/remote embedding tokenizer mismatch: ' + name)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.tokenizer_only = tokenizer_only

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf8')
        request = urllib.request.Request(self.endpoint + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self.config.get('timeout_seconds', 1800)) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError('Embedding worker failed: ' + exc.read().decode('utf8', errors='replace')) from exc

    def _encode(self, texts, prefix):
        if self.tokenizer_only:
            raise RuntimeError('Tokenizer-only encoder cannot generate vectors')
        prepared = [prefix + t for t in texts]
        counts = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in prepared]
        for i, count in enumerate(counts):
            if count > self.provenance['max_tokens']:
                raise ValueError(f'Embedding input {i} has {count} tokens; limit=32768; no truncation')
        if not prepared:
            return np.empty((0, 4096), dtype=np.float32)
        # The remote GPU worker releases this task's Ollama model. Restore the
        # same loaded context after the embedding child has exited; the LLM
        # adapter will independently verify its digest/context before use.
        settings = getattr(self, 'config', {})
        generation_endpoint = settings.get('generation_endpoint')
        loaded = None
        if generation_endpoint:
            from .llm import is_loopback_endpoint
            if not is_loopback_endpoint(generation_endpoint):
                raise ValueError('Generation restore requires a loopback endpoint')
            def ollama(path, payload=None):
                data = None if payload is None else json.dumps(payload).encode()
                req = urllib.request.Request(generation_endpoint.rstrip('/') + path, data=data,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=900) as response:
                    return json.load(response)
            loaded = next((m for m in ollama('/api/ps')['models']
                           if m.get('name') == settings['generation_model']), None)
        result = self._request('/embed', {'texts': prepared, 'revision': self.revision})
        if loaded:
            context = loaded.get('context_length')
            if not isinstance(context, int) or context <= 0:
                raise ValueError('Cannot restore unverifiable generation context')
            ollama('/api/generate', {'model': settings['generation_model'], 'stream': False,
                'keep_alive': '30m', 'options': {'num_ctx': context}})
            restored = next((m for m in ollama('/api/ps')['models']
                             if m.get('name') == settings['generation_model']), {})
            if restored.get('context_length') != context or restored.get('digest') != loaded.get('digest'):
                raise ValueError('Generation identity/context changed after embedding')
        if result.get('token_counts') != counts or result.get('embedding') != self.provenance:
            raise ValueError('Remote embedding token counts or identity changed')
        vectors = np.asarray(result['vectors'], dtype=np.float32)
        if vectors.shape != (len(prepared), 4096) or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding shape/values')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError('Embedding vectors must be L2 normalized')
        return vectors

    def encode_documents(self, texts):
        return self._encode(texts, '')

    def encode_queries(self, texts):
        return self._encode(texts, self.provenance['query_prefix'])
