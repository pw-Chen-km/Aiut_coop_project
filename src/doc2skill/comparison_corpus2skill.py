"""Local-model transport for the pinned upstream verify/repartition prompt."""
from copy import deepcopy
import json

from .comparison_requests import request, object_schema
from ._vendor.corpus2skill.repartition_prompt import _REPARTITION_PROMPT, _build_clusters_block


def repartition(client, records, level, audit, directory=None):
    # Drop the adapter's private str-subclass identity before model/audit transport.
    records = json.loads(json.dumps(records, ensure_ascii=False))
    entry = {'level': level, 'records': deepcopy(records), 'status': 'running'}
    audit.setdefault('repartition', []).append(entry)
    # Upstream algorithm guard, distinct from model capacity checking.
    if len(records) > 60:
        entry.update(status='skipped', reason='upstream_max_clusters_per_call', limit=60)
        return {'changes': []}
    if client is None:
        raise ValueError('Corpus2Skill repartition requires a model client')
    by_id = {r['cluster_id']: {m['id'] for m in r['members']} for r in records}
    members = sorted(set().union(*by_id.values()))
    schema = object_schema({'changes': {'type': 'array', 'items': object_schema({
        'old_cluster_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                            'items': {'type': 'string', 'enum': list(by_id)}},
        'new_partition': {'type': 'object', 'additionalProperties': {
            'type': 'array', 'minItems': 1, 'uniqueItems': True,
            'items': {'type': 'string', 'enum': members}}}})}})

    def validate(data):
        used = set()
        for change in data['changes']:
            old = set(change['old_cluster_ids'])
            if old & used:
                raise ValueError('A cluster occurs in multiple repartition changes')
            used.update(old)
            partition = change['new_partition']
            if not partition or any(not label.strip() for label in partition):
                raise ValueError('Repartition requires nonempty group labels')
            assigned = []
            for values in partition.values():
                if not isinstance(values, list) or not values or any(not isinstance(x, str) for x in values):
                    raise ValueError('Repartition members must be nonempty lists of IDs')
                assigned.extend(values)
            expected = set().union(*(by_id[c] for c in old))
            if len(assigned) != len(set(assigned)) or set(assigned) != expected:
                raise ValueError('Repartition must cover each shown member exactly once within flagged clusters')

    prompt = _REPARTITION_PROMPT.format(level=level, clusters_block=_build_clusters_block(records))
    try:
        data = request(client, prompt, {}, schema, 'corpus2skill_repartition',
                       directory=directory, validate=validate)
        entry.update(status='complete', response=deepcopy(data))
        return data
    except Exception as exc:
        entry.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
