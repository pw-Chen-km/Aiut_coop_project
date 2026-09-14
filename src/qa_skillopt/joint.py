"""Atomic multi-MD worksets and navigation quality/cost acceptance (v2)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import re
from doc2skill.adaptive import envelope, validate_tree

TOLERANCE = 1e-9
HEADER = '<!-- navigation-workset-v2 -->\n'
MARKER = re.compile(r'^<!-- BEGIN (n_[a-f0-9]{16}) -->\n(.*?)^<!-- END \1 -->\n', re.M | re.S)


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pack_workset(skills):
    return HEADER + ''.join(f'<!-- BEGIN {nid} -->\n{md.rstrip()}\n<!-- END {nid} -->\n'
                            for nid, md in skills.items())


def unpack_workset(text, expected_ids):
    if not text.startswith(HEADER):
        raise ValueError('Missing workset boundary')
    pos, result = len(HEADER), {}
    for match in MARKER.finditer(text, pos):
        nid, md = match.groups()
        if match.start() != pos or nid in result or not md.strip():
            raise ValueError('Partial, duplicate, or empty MD in workset')
        result[nid] = md; pos = match.end()
    if pos != len(text) or set(result) != set(expected_ids):
        raise ValueError('Workset must preserve every selected MD exactly once')
    return result


def compare_metrics(before, after):
    if not before.get('eligible_ids') or before['eligible_ids'] != after.get('eligible_ids'):
        raise ValueError('Navigation denominator changed or is empty')
    if not before.get('answer_ids') or before['answer_ids'] != after.get('answer_ids'):
        raise ValueError('Answer denominator changed or is empty')
    if before.get('cost_ids') != after.get('cost_ids'):
        raise ValueError('Cost denominator changed')
    for row in (before, after):
        for key in ('Q', 'route_complete', 'evidence_complete', 'answer_accuracy', 'navigation_calls', 'navigation_tokens'):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError('Uncertain or unavailable comparison: ' + key)
    if any(after[k] < before[k] - TOLERANCE for k in ('route_complete', 'evidence_complete', 'answer_accuracy')):
        return {'accepted': False, 'reason': 'quality_non_regression_failed'}
    delta = after['Q'] - before['Q']
    if delta > TOLERANCE:
        return {'accepted': True, 'reason': 'quality_improvement'}
    costs = ('navigation_calls', 'navigation_tokens')
    if abs(delta) <= TOLERANCE and all(after[k] <= before[k] + TOLERANCE for k in costs) and any(
            after[k] < before[k] - TOLERANCE for k in costs):
        return {'accepted': True, 'reason': 'equal_quality_lower_cost'}
    return {'accepted': False, 'reason': 'no_quality_or_efficiency_improvement'}


class JointGate:
    """Called by the real trainer after a complete validation rollout."""
    def __init__(self, metrics):
        self.metrics, self.history = metrics, []

    def __call__(self, candidate_skill, cand_hard, current_skill, current_score,
                 best_skill, best_score, best_step, global_step, **kwargs):
        from skillopt.evaluation.gate import GateResult
        before, after = self.metrics[content_hash(current_skill)], self.metrics[content_hash(candidate_skill)]
        actual = .5 * cand_hard + .5 * kwargs.get('cand_soft', 0.)
        if abs(actual - after['Q']) > TOLERANCE:
            raise ValueError('Upstream gate and external navigation score disagree')
        decision = compare_metrics(before, after)
        self.history.append({'before_hash': content_hash(current_skill), 'candidate_hash': content_hash(candidate_skill),
                             **decision, 'before': deepcopy(before), 'after': deepcopy(after)})
        if decision['accepted']:
            return GateResult('accept_new_best', candidate_skill, after['Q'], candidate_skill, after['Q'], global_step)
        return GateResult('reject', current_skill, current_score, best_skill, best_score, best_step)

    def selected_is_accepted(self, initial, selected):
        if initial == selected:
            return False
        return any(r['accepted'] and r['candidate_hash'] == content_hash(selected) for r in self.history)


class WorksetValidator:
    def __init__(self, tree, snapshot, targets, counter, *, max_input_tokens=7000,
                 reserve_tokens=1536, questions=(), audit=None):
        validate_tree(tree)
        self.tree, self.snapshot, self.targets = deepcopy(tree), deepcopy(snapshot), set(targets)
        self.counter, self.maximum, self.reserve = counter, max_input_tokens, reserve_tokens
        self.questions, self.audit = list(questions), audit
        self.reports = []

    def __call__(self, before, after, patch, application_reports):
        report = {'valid': False, 'changed_nodes': [], 'grounding': []}
        try:
            old, new = unpack_workset(before, self.targets), unpack_workset(after, self.targets)
            for nid, md in new.items():
                node = self.tree['nodes'][nid]
                known = set(re.findall(r'\bn_[a-f0-9]{16}\b', md))
                if known - self.tree['nodes'].keys() or not set(node['children']) <= known:
                    raise ValueError('Unknown reference or missing child navigation entry')
                # Local paths are not routing authority. Reject links to a new
                # file, URL, or shell rather than interpreting them as actions.
                links = re.findall(r'\]\(([^)]+)\)', md)
                if set(links) - set(re.findall(r'\]\(([^)]+)\)', old[nid])):
                    raise ValueError('New executable/external path link')
                if re.search(r'\bq\d{4,}\b|\bqid\s*[:=]|ignore\s+(all\s+|the\s+)?(previous|prior|system)\s+instructions|<\|(?:system|assistant|im_start|im_end)\|>', md, re.I):
                    raise ValueError('Deterministic lookup/instruction-injection marker')
                if re.search(r'\b(?:if|when)\b[^\n]{0,80}\b(?:query|question)\b[^\n]{0,80}'
                             r'\b(?:contains|equals|matches|is exactly|starts with)\b|'
                             r'\b(?:exact[- ]match|keyword[- ]lookup|question[- ]to[- ]answer|qid[- ]lookup)\b', md, re.I):
                    raise ValueError('Question-to-ID lookup rule')
                normal = ' '.join(md.casefold().split())
                if any(len(q.strip()) >= 12 and ' '.join(q.casefold().split()) in normal
                       and ' '.join(q.casefold().split()) not in ' '.join(old[nid].casefold().split()) for q in self.questions):
                    raise ValueError('Complete optimization question copied into skill')
                messages, schema = envelope(node, md, '')
                size = self.counter(json.dumps({'messages': messages, 'schema': schema}, ensure_ascii=False)) + self.reserve
                if size > self.maximum:
                    raise ValueError('Navigation request exceeds fixed budget')
                if md != old[nid]:
                    report['changed_nodes'].append(nid)
            report['valid'] = True
        except (ValueError, KeyError, TypeError) as exc:
            report['reason'] = str(exc)
        # Semantic judgments remain private audit only, including outages.
        if report['valid'] and self.audit:
            for nid in report['changed_nodes']:
                try:
                    value = self.audit(nid, old[nid], new[nid])
                except Exception as exc:
                    value = {'status': 'unavailable', 'reason': str(exc)}
                report['grounding'].append({'node_id': nid, **value})
        self.reports.append(deepcopy(report))
        return report
