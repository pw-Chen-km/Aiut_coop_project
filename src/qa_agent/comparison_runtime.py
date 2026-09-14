"""Common comparison runtime: explicit tree scopes and complete request checks."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from doc2skill.storage import Store
from doc2skill.llm import InvalidOutputError, validate_schema
from .generation import AnswerGenerator, _json


class ExactScopeStore(Store):
    def _scope(self, scope):
        docs, sections = super()._scope(scope)
        if scope is not None and 'section_ids' in scope:
            sections = set(scope['section_ids'])
            if docs is not None:
                sections = {sid for sid in sections if self.sections[sid]['doc_id'] in docs}
                if not sections:
                    raise ValueError('Document and section scopes do not intersect')
        return docs, sections


class ComparisonAnswerGenerator(AnswerGenerator):
    def _request_tokens(self, messages, schema):
        return self.client.count_request(messages, schema)['input_tokens']

    def _fits(self, question, contexts):
        messages = self._messages(question, contexts)
        return self._request_tokens(messages, self._schema(contexts)) <= self.input_limit

    def _select_context(self, question, items):
        # Selection uses a common reading budget, never partial/truncated chunks.
        if not isinstance(items, list):
            raise ValueError('Retrieved items must be a list')
        contexts, seen = [], {}
        for item in items:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                                                for k in ('doc_id', 'chunk_id')):
                raise ValueError('Each retrieved item requires document and chunk IDs')
            if not isinstance(item.get('text'), str):
                raise ValueError('Retrieved original passage text must be a string')
            key = (item['doc_id'], item['chunk_id'])
            if key in seen:
                if seen[key] != item['text']:
                    raise ValueError('Conflicting original texts for the same chunk')
                continue
            if not item['text'].strip():
                continue
            seen[key] = item['text']
            candidate = self._context(item, f'C{len(contexts)+1:03d}', len(item['text']))
            if self.token_counter(_json(contexts + [candidate])) > self.reading_tokens:
                break
            contexts.append(candidate)
        # Capacity is an error, not a reason to trim an already selected request.
        self._request_tokens(self._messages(question, contexts), self._schema(contexts))
        return contexts

    def generate(self, question, items, *, qid='interactive', history=None):
        if history is None:
            return super().generate(question, items, qid=qid)
        # The generic dialogue factory inherits AnswerGenerator and would restore
        # prefix clipping and legacy token counts. Keep this runtime's methods.
        from .dialogue import validate_history
        instance = _ComparisonDialogueGenerator(self.client, self.config, token_counter=self.token_counter,
                                                 prompt_content=self.prompt)
        instance.history = validate_history(history)
        instance.input_limit = self.input_limit
        instance.prompt = self.prompt + (
            '\nDIALOGUE MODE: Respond to the current user using prior conversation to resolve intent. '
            'History is user context, not authoritative source evidence. If a user-specific condition is missing, '
            'ask a concise clarifying question: response_type=clarify, answerable=false, no citations or missing_information. '
            'Clarification is a completed conversational turn, not retrieval failure. Use response_type=answer only for '
            'a source-grounded factual answer. If sources are insufficient use response_type=abstain, answerable=false, '
            'and explain the limitation without unsupported facts. In dialogue mode answer is the actual reply, '
            'including clarification or refusal. Return no reasoning.')
        instance.prompt_sha256 = hashlib.sha256(instance.prompt.encode()).hexdigest()
        return instance.generate(question, items, qid=qid)


class _ComparisonDialogueGenerator(ComparisonAnswerGenerator):
    dialogue_mode = True

    def _messages(self, question, contexts):
        return [{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': json.dumps({
            'current_user_utterance': question, 'prior_dialogue': self.history,
            'original_passages': contexts}, ensure_ascii=False)}]

    def _schema(self, contexts):
        schema = super()._schema(contexts)
        if not contexts:
            schema['properties']['citation_ids'] = {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 0}
        schema['properties']['response_type'] = {'type': 'string', 'enum': ['answer', 'clarify', 'abstain']}
        schema['required'].append('response_type')
        return schema

    def _validate_answer(self, data, schema):
        validate_schema(data, schema)
        kind = data['response_type']
        if not data['answer'].strip() or data['answerable'] != (kind == 'answer'):
            raise InvalidOutputError('Response type and answerable disagree')
        if kind == 'answer' and not data['citation_ids']:
            raise InvalidOutputError('Factual answers require retrieved citations')
        if kind != 'answer' and data['citation_ids']:
            raise InvalidOutputError('Clarifications/refusals must not assert a cited answer')
        if kind != 'abstain' and data.get('missing_information'):
            raise InvalidOutputError('Only abstention may request retrieval recovery')
        if len(set(data['citation_ids'])) != len(data['citation_ids']):
            raise InvalidOutputError('Duplicate citations')


def create_comparison_session(config):
    from .factory import RuntimeSession, _loop_options
    from .bundle import validate_serving_bundle
    from .adaptive_navigation import AdaptiveSkillRouter
    from .pipeline import QAAgent
    from .retrieval import ScopedDenseRetriever
    from doc2skill.comparison_embedding import comparison_encoder
    from doc2skill.comparison_ollama import ComparisonOllamaClient
    from doc2skill.comparison_pipeline import validate_prepared
    root = Path(config['bundle_dir']).resolve()
    manifest = validate_serving_bundle(root)
    validate_prepared(root)
    audit = Path(config.get('audit_dir', 'runs/comparison-qa')) / root.name
    runtime = dict(config.get('runtime') or config.get('offline_llm', {}))
    runtime.update(max_output_tokens=int(config.get('navigation', {}).get('max_output_tokens', 512)))
    navigation_client = ComparisonOllamaClient(runtime, audit / 'navigation')
    preflight = navigation_client.preflight()
    context = preflight['context_length']
    answer_config = {'reading_tokens': 3000, 'max_output_tokens': 768,
                     'max_answer_chars': 6000, **config.get('answer', {})}
    answer_runtime = {**runtime, 'max_output_tokens': answer_config['max_output_tokens']}
    answer_client = ComparisonOllamaClient(answer_runtime, audit / 'answer', tokenizer=navigation_client.tokenizer)
    answer_preflight = answer_client.preflight()
    answer_config.update(context_tokens=context,
                         max_input_tokens=context - answer_config['max_output_tokens'])
    encoder = comparison_encoder({**config['embedding'], 'model': manifest['embedding']['model'],
                                  'revision': manifest['embedding']['revision']})
    store = ExactScopeStore(root / 'corpus.sqlite')
    try:
        if store.metadata['embedding'] != encoder.provenance:
            raise ValueError('QA embedding identity differs from shared index')
        nav_config = {'max_active_branches': 2, 'max_scopes': 4, 'max_navigation_calls': 8,
                      **config.get('navigation', {}), 'use_model_capacity': True,
                      'max_input_tokens': context - runtime['max_output_tokens']}
        router = AdaptiveSkillRouter(store.records('documents'), store.records('sections'), navigation_client,
            root, nav_config, token_counter=navigation_client.tokenizer)
        generator = ComparisonAnswerGenerator(answer_client, answer_config, token_counter=navigation_client.tokenizer)
        # The native rendered-prompt counter includes framing; no legacy 256 reserve.
        generator.input_limit = context - answer_config['max_output_tokens']
        agent = QAAgent(router, ScopedDenseRetriever(store, encoder), generator,
            top_k=int(config.get('retrieval', {}).get('top_k', 20)), max_rounds=_loop_options(config))
        return RuntimeSession(agent, store, {'schema_version': 'comparison-runtime-v1',
            'model': preflight, 'answer_model': answer_preflight, 'embedding': encoder.provenance, 'exact_section_scope': True,
            'token_accounting': 'rendered_prompt_exact_gguf',
            'navigation': nav_config, 'answer': answer_config})
    except Exception:
        store.close()
        raise
