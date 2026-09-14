"""Gold stays here; only routing IDs, decisions and costs leave for reflection."""
from collections import defaultdict
from itertools import product
from hashlib import sha256
import json
from .trajectory import evidence_coverage


def item_spans(items):
    """Exact v2 projection; HTML has real text coordinates, not invented pages."""
    output = defaultdict(list)
    for item in items:
        text = item['text']; covered = bytearray(len(text))
        for s in item.get('source_spans', []):
            a, b, x, y = (s.get(k) for k in ('chunk_start_char', 'chunk_end_char', 'source_start_char', 'source_end_char'))
            if any(type(v) is not int for v in (a, b, x, y)) or not 0 <= a < b <= len(text) or x < 0 or y-x != b-a:
                raise ValueError('Invalid v2 source span')
            page = s.get('page_index')
            if not (type(page) is int and page >= 0) and not (page is None and s.get('source_type') == 'html'):
                raise ValueError('Missing PDF page or native HTML provenance')
            covered[a:b] = b'\1'*(b-a)
            output[item['doc_id']].append({'block_id': s['block_id'], 'page_index': page,
                'start_char': x, 'end_char': y, 'text': text[a:b]})
        if any(not text[i].isspace() and not covered[i] for i in range(len(text))):
            raise ValueError('Unanchored retrieved text')
    return output


def required_scope_sets(query, tree, blocks=()):
    """Each evidence occurrence is an alternative; its covered blocks are joint.

    Do not mistake two chunks containing the same quote for two required hops.
    Resolve required section owners from the immutable block registry when given.
    """
    owners = {b['block_id']: b['section_id'] for b in blocks}
    choices = defaultdict(set)
    for ev in query.get('evidence', []):
        key = ev.get('equivalence_group') or ev['evidence_id']
        for t in ev.get('targets', []):
            spans = t.get('source_spans', [])
            if owners:
                if any(s['block_id'] not in owners for s in spans):
                    continue
                ids = {owners[s['block_id']] for s in spans}
            else:
                ids = set(t.get('section_ids', []))
            if ids and ids <= tree['section_documents'].keys():
                choices[key].add(frozenset(ids))
    alternatives = set()
    for route in query.get('routes', []):
        keys = route.get('required_keys', [])
        if not keys or any(not choices[k] for k in keys):
            return None  # Unknown alternatives are not navigation failures.
        combinations = 1
        for k in keys:
            combinations *= len(choices[k])
        if combinations > 100000:
            raise ValueError('Oracle alternative expansion exceeds explicit resource budget')
        for combination in product(*(choices[k] for k in keys)):
            alternatives.add(frozenset().union(*combination))
    return [sorted(s) for s in sorted(alternatives, key=lambda x: tuple(sorted(x)))] or None


def scope_score(selected, alternatives):
    values = []
    for alternative in alternatives:
        required = set(alternative); hit = len(required & set(selected))
        p = hit / len(selected) if selected else 0.
        r = hit / len(required)
        values.append({'complete': int(required <= set(selected)), 'precision': p, 'recall': r,
                       'f1': 2*p*r/(p+r) if p+r else 0.})
    return max(values, key=lambda v: (v['complete'], v['f1'], v['recall']))


def navigation_cost(trajectory):
    """Costs do not need a gold label, so unknown-oracle cases still count."""
    calls, tokens = 0, 0
    for row in trajectory.get('rounds', []):
        for entry in row.get('stages', {}).get('navigation', {}).get('trace', []):
            calls += 1
            usage = entry.get('response', {}).get('usage', {})
            inp = usage.get('prompt_tokens', usage.get('input_tokens', entry.get('input_tokens_estimate')))
            out = usage.get('completion_tokens', usage.get('output_tokens'))
            if type(inp) is not int or type(out) is not int or min(inp, out) < 0:
                raise ValueError('Navigation token accounting unavailable')
            tokens += inp + out
    return {'navigation_calls': calls, 'navigation_tokens': tokens}


def navigation_feedback(qa, trajectory, query, tree, alternatives):
    if trajectory.get('status') != 'ok':
        raise ValueError('QA execution failure: candidate comparison stopped')
    selected, reached, pooled, rounds = set(), set(), {}, []
    calls, tokens, seen_exposures = 0, 0, set()
    for index, row in enumerate(trajectory.get('rounds', []), 1):
        nav = row.get('stages', {}).get('navigation', {})
        actual = set(row.get('scope', {}).get('section_ids', []))
        if not actual <= tree['section_documents'].keys():
            raise ValueError('Trajectory contains unknown source scope')
        selected |= actual
        decisions = []
        for decision in nav.get('decisions', []):
            nid = decision['node_id']; reached.add(nid)
            node = tree['nodes'][nid]
            chosen = decision['node_ids']
            relevant = [sorted(c for c in node['children'] if set(tree['nodes'][c]['scope']['section_ids']) & set(a))
                        for a in alternatives]
            decisions.append({'node_id': nid, 'action': decision['action'], 'selected_node_ids': chosen,
                'acceptable_relevant_child_sets': relevant, 'automatic_leaf': decision.get('automatic_leaf', False)})
        traces = nav.get('trace', [])
        calls += len(traces)
        for entry in traces:
            response = entry.get('response', {})
            usage = response.get('usage', {})
            inp = usage.get('prompt_tokens', usage.get('input_tokens', entry.get('input_tokens_estimate')))
            out = usage.get('completion_tokens', usage.get('output_tokens'))
            if type(inp) is not int or type(out) is not int or min(inp, out) < 0:
                raise ValueError('Navigation token accounting unavailable')
            tokens += inp + out
        for item in row.get('items', []):
            key = (item['doc_id'], item['chunk_id'])
            if key in pooled and pooled[key]['text'] != item['text']:
                raise ValueError('Conflicting retrieved source text')
            pooled[key] = item
        score = scope_score(selected, alternatives)
        exposures = [sha256(json.dumps({k: c.get(k) for k in (
            'doc_id', 'chunk_id', 'excerpt_start_char', 'excerpt_end_char')}, sort_keys=True).encode()).hexdigest()
            for c in row.get('context_items', [])]
        repeated = sum(k in seen_exposures for k in exposures)
        seen_exposures.update(exposures)
        rounds.append({'round': index, 'decisions': decisions,
                       'selected_scope_node_ids': nav.get('selected_node_ids', [d['node_id'] for d in decisions if d['action'] == 'retrieve']),
                       'selected_original_section_count': len(actual),
                       'cumulative_route_complete': score['complete'], 'navigation_calls': len(traces),
                       'new_evidence_count': len(row.get('new_evidence_keys', [])),
                       'read_exposure_keys': exposures, 'repeated_exposure_count': repeated})
    spans = item_spans(list(pooled.values()))
    ev = evidence_coverage(query, spans_by_document=spans)
    if ev.get('status') != 'known':
        raise ValueError('Evidence coordinates or scoring denominator unavailable')
    route = scope_score(selected, alternatives)
    relevant_nodes = {nid for nid, n in tree['nodes'].items()
                      if any(set(n['scope']['section_ids']) & set(a) for a in alternatives)}
    # Being downstream of an unchosen branch is an observation, never blame.
    view = {'user_utterance': qa['question'], 'history': qa.get('history', []),
            'rounds': rounds, 'acceptable_required_section_sets': alternatives,
            'not_reached_relevant_node_ids': sorted(relevant_nodes - reached),
            'not_reached_means': 'unobserved; not a downstream routing failure',
            'route_complete': route['complete'], 'scope_f1': route['f1'],
            'navigation_calls': calls, 'navigation_tokens': tokens}
    return {'route_complete': route['complete'], 'scope_f1': route['f1'],
            'scope_precision': route['precision'], 'scope_recall': route['recall'],
            'first_round_route_complete': rounds[0]['cumulative_route_complete'] if rounds else 0,
            'evidence_recall': ev['recall'], 'evidence_complete': int(ev['complete']),
            'navigation_calls': calls, 'navigation_tokens': tokens, 'optimizer_view': view}


def aggregate(rows, answers, costs=None):
    if not rows or not answers:
        raise ValueError('Empty navigation/answer evaluation denominator')
    result = {key: sum(r[key] for r in rows)/len(rows) for key in (
        'route_complete', 'scope_f1', 'evidence_recall', 'evidence_complete',
        'navigation_calls', 'navigation_tokens', 'first_round_route_complete')}
    result.update(Q=.5*result['route_complete']+.25*result['scope_f1']+.25*result['evidence_recall'],
                  eligible_ids=sorted(r['qid'] for r in rows), answer_ids=sorted(answers),
                  answer_accuracy=sum(answers.values())/len(answers))
    if costs is not None:
        if set(costs) != set(answers):
            raise ValueError('Cost evaluation must include the full answer guard set')
        result.update({k: sum(r[k] for r in costs.values())/len(costs) for k in ('navigation_calls', 'navigation_tokens')})
        result['cost_ids'] = sorted(costs)
    return result
