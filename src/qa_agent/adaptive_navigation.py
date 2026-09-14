"""Finite traversal over registry scopes; Markdown is editable presentation only."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from pathlib import Path
from doc2skill.adaptive import envelope, validate_tree, navigation_children
from doc2skill.llm import validate_schema
from .bundle import local_asset


class AdaptiveSkillRouter:
    def __init__(self, documents, sections, client, artifact_dir, config, *, token_counter,
                 skill_content=None, navigation_md_overrides=None):
        if not callable(token_counter):
            raise ValueError('Adaptive routing requires a real token counter')
        if hasattr(client, 'local_only') and not client.local_only:
            raise ValueError('Online router must use local transport')
        if skill_content is not None:
            raise ValueError('Adaptive policy is registry-controlled; edit navigation MD instead')
        self.root = Path(artifact_dir)
        self.paths = json.loads((self.root / 'skill_paths.json').read_text())
        self.tree = validate_tree(json.loads(local_asset(self.root, self.paths['adaptive_tree']).read_text()))
        self.nodes = self.tree['nodes']
        if self.tree['section_documents'] != {s['section_id']: s['doc_id'] for s in sections}:
            raise ValueError('Navigation registry differs from corpus')
        self.config = {'max_input_tokens': 7000, 'max_active_branches': 2, 'max_scopes': 4,
                       'max_navigation_calls': 8, **config}
        for k in ('max_input_tokens', 'max_active_branches', 'max_scopes', 'max_navigation_calls'):
            if type(self.config[k]) is not int or self.config[k] < 1:
                raise ValueError('Invalid navigation resource limit')
        self.client, self.token_counter = client, token_counter
        self.navigation_md_overrides = copy.deepcopy(navigation_md_overrides or {})
        if set(self.navigation_md_overrides) - self.nodes.keys() or any(
            not isinstance(s, str) or not s.strip() for s in self.navigation_md_overrides.values()):
            raise ValueError('Overrides must name existing node IDs with nonempty Markdown')
        for nid in self.nodes:
            self.read_md(nid)

    def read_md(self, nid):
        if nid in self.navigation_md_overrides:
            return self.navigation_md_overrides[nid]
        return local_asset(self.root, self.nodes[nid]['md_path']).read_text(encoding='utf-8')

    def navigation_override_hashes(self):
        return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in self.navigation_md_overrides.items()}

    def route(self, question, *, qid='interactive', feedback=None):
        started = time.monotonic()
        result = {'qid': qid, 'question': question, 'status': 'failed', 'trace': [], 'decisions': []}
        pending, terminals, calls, repairs = [self.tree['root_id']], [], 0, 0
        visited = set()
        try:
            while pending:
                nid = pending.pop(0); node = self.nodes[nid]
                if nid in visited:
                    continue
                visited.add(nid)
                if not navigation_children(node):
                    terminals.append(nid)
                    result['decisions'].append({'node_id': nid, 'action': 'retrieve', 'node_ids': [nid],
                                                'automatic_leaf': True, 'scope': node['scope']})
                    continue
                capacity = self.config['max_active_branches'] - len(pending)
                messages, schema = envelope(node, self.read_md(nid), question, feedback, capacity)
                while True:
                    if calls >= self.config['max_navigation_calls']:
                        raise ValueError('Navigation call limit reached')
                    if self.config.get('use_model_capacity'):
                        tokens = self.client.count_request(messages, schema)['input_tokens']
                    else:
                        tokens = self.token_counter(json.dumps({'messages': messages, 'schema': schema}, ensure_ascii=False))
                    if tokens > self.config['max_input_tokens']:
                        raise ValueError('Navigation request budget exceeded; no truncation')
                    entry = {'stage': 'navigate_node', 'node_id': nid, 'messages': copy.deepcopy(messages),
                             'schema': schema, 'input_tokens_estimate': tokens, 'status': 'failed'}
                    result['trace'].append(entry); calls += 1; tick = time.monotonic()
                    try:
                        response = self.client.complete(messages, schema, 'navigate_node', no_repair=True)
                        entry['elapsed_seconds'] = time.monotonic() - tick
                        entry['response'] = response
                        data = response['data']; validate_schema(data, schema)
                        ids = data['node_ids']
                        if data['action'] == 'retrieve' and ids != [nid]:
                            raise ValueError('Retrieve must use the current node')
                        if data['action'] == 'descend' and (nid in ids or not set(ids) <= set(navigation_children(node))):
                            raise ValueError('Descend must select children')
                        entry['status'] = 'ok'
                        break
                    except (ValueError, KeyError) as exc:
                        entry['elapsed_seconds'] = time.monotonic() - tick
                        entry['status'] = 'invalid_output'
                        if repairs >= 1:
                            raise ValueError('Invalid navigation after one repair') from exc
                        repairs += 1
                        messages += [{'role': 'user', 'content': 'Choose retrieve with the current node ID, or descend with child IDs. Return the schema.'}]
                result['decisions'].append({'node_id': nid, **data, 'scope': node['scope'], 'automatic_leaf': False})
                if data['action'] == 'retrieve':
                    terminals.append(nid)
                else:
                    pending += [i for i in ids if i not in visited and i not in pending]
                if len(terminals) + len(pending) > self.config['max_scopes']:
                    raise ValueError('Too many terminal retrieval scopes')
            selected = set(s for n in terminals for s in self.nodes[n]['scope']['section_ids'])
            result.update(status='ok', selected_node_ids=terminals, scope={
                'section_ids': sorted(selected), 'doc_ids': sorted({self.tree['section_documents'][s] for s in selected})},
                candidate_scope_sections=len(selected), format_repairs=repairs)
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result.update(navigation_calls=calls, elapsed_seconds=time.monotonic() - started)
        return result
