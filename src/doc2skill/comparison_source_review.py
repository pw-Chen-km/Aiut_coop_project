"""Separate source-role, heading-fragment and body-owner decisions."""
from copy import deepcopy
from collections import Counter
from .comparison_requests import request, object_schema, keyed_schema
from .comparison_source import _geometry, merge_heading_fragments
from .config import write_json

ROLES = ['heading', 'body', 'administrative', 'table_of_contents', 'page_header', 'page_footer']
STRING = {'type': 'string', 'minLength': 1}


def review_items(items, raw, client, directory, audit):
    result = deepcopy(items)
    texts = [x for x in result if x.get('text', '').strip()
             and x.get('label') not in {'table', 'picture', 'document_index'}]
    aliases = {x['self_ref']: f'R{i:04d}' for i, x in enumerate(texts)}
    pages = sorted({g[0] for x in texts if (g := _geometry(x, raw))})
    if any(_geometry(x, raw) is None for x in texts):
        raise ValueError('Role review requires page geometry for every text item')
    # Every call reads the same immutable snapshot; earlier decisions cannot bias later pages.
    records = [{'id': aliases[x['self_ref']], 'text': x['text'], 'label': x.get('label'),
                'heading_level': x.get('level'), 'geometry': _geometry(x, raw),
                'layer': x.get('_layer'), 'source_role': x.get('_source_role')}
               for x in texts]
    changes = {}
    value = object_schema({'role': {'type': 'string', 'enum': ROLES},
                           'confirmed': {'type': 'boolean'}, 'reason': STRING})
    for page in pages:
        target = [r['id'] for r in records if r['geometry'][0] == page]
        def valid(data):
            pending = [key for key, decision in data.items() if not decision['confirmed']]
            if pending:
                raise ValueError('Unresolved source roles: ' + ', '.join(pending))
        changes.update(request(client,
            'Classify source text roles, not topics. Decide ONLY target IDs. Use position, native label, '
            'heading size and neighboring pages. A real repeated section title can remain a heading; '
            'a running version banner is furniture. Contact/legal/registration text is administrative '
            'content unless layout proves it is a running header/footer. TOC entries are not section '
            'boundaries. Do not rewrite text, invent headings or decide section parents. '
            'Return confirmed=false when evidence is insufficient.',
            {'target_page': page, 'target_ids': target,
             'records': [r for r in records if abs(r['geometry'][0] - page) <= 1],
             'repeated_text_pages': {r['id']: sorted({s['geometry'][0] for s in records
                  if s['text'].strip() == r['text'].strip()}) for r in records if r['id'] in target}},
            keyed_schema(target, value), f'source:roles:page-{page}', directory=directory, validate=valid))
    for x in texts:
        decision = changes[aliases[x['self_ref']]]
        role = decision['role']
        audit['decisions'].append({'native_ref': x['self_ref'], 'before': x.get('label'), **decision})
        x.pop('_source_role', None)
        if role in {'page_header', 'page_footer'}:
            x.update(label=role, _layer='furniture', content_layer='furniture')
        else:
            x.update(_layer='body', content_layer='body', _preserve_heading=True)
            x['label'] = 'section_header' if role == 'heading' else 'text'
            if role == 'heading':
                x['level'] = max(1, int(x.get('level') or 1))
            elif role in {'administrative', 'table_of_contents'}:
                x['_source_role'] = role
    # A restored heading must precede its native body. Keep Docling's column order otherwise.
    restored = [x for x in texts if x['label'] == 'section_header'
                and next(r for r in records if r['id'] == aliases[x['self_ref']])['layer'] == 'furniture']
    for x in restored:
        result.remove(x)
        g = _geometry(x, raw)
        position = next((i for i, y in enumerate(result) if y.get('_layer') != 'furniture'
                        and (yg := _geometry(y, raw)) and
                        (yg[0] > g[0] or yg[0] == g[0] and yg[2] >= g[4])), len(result))
        result.insert(position, x)
    return result


def review_fragments(items, raw, client, directory, audit, explicit=()):
    merge_heading_fragments(items, raw, explicit, audit)
    body = [x for x in items if x.get('_layer') != 'furniture']
    candidates = []
    for a, b in zip(body, body[1:]):
        repair = {'first_ref': a['self_ref'], 'second_ref': b['self_ref'], 'reason': 'Candidate only'}
        try:
            merge_heading_fragments(deepcopy(items), raw, [repair], {'fragment_merges': []})
        except ValueError:
            continue
        candidates.append({**repair, 'first_text': a.get('text'), 'second_text': b.get('text'),
                           'geometry': _geometry(a, raw)})
    if not candidates:
        return
    rows = {f'F{i:04d}': r for i, r in enumerate(candidates)}
    data = request(client, 'Decide whether each pair is ONE printed heading split into two lines. '
                   'Do not merge independent headings or similar topics. Return merge=false when unsure.',
                   rows, keyed_schema(rows, object_schema({'merge': {'type': 'boolean'}, 'reason': STRING})),
                   'source:heading-fragments', directory=directory)
    repairs = [{k: r[k] for k in ('first_ref', 'second_ref')} | {'reason': data[key]['reason']}
               for key, r in rows.items() if data[key]['merge']]
    merge_heading_fragments(items, raw, repairs, audit)


def review_owners(corpus, client, directory):
    sections = corpus['sections']
    sid_alias = {s['section_id']: f'S{i:04d}' for i, s in enumerate(sections)}
    alias_sid = {v: k for k, v in sid_alias.items()}
    blocks = corpus['blocks']
    aliases = {b['block_id']: f'B{i:05d}' for i, b in enumerate(blocks)}
    outline = [{'id': sid_alias[s['section_id']], 'title': s['title'], 'level': s['level'],
                'heading_locations': [b['provenance'] for b in blocks
                                      if b['native_ref'] in s.get('native_heading_refs', [])]}
               for s in sections]
    records = [{'id': aliases[b['block_id']], 'owner': sid_alias[b['section_id']],
                'kind': b['kind'], 'text': b['text'], 'page': b.get('page_index'),
                'provenance': b['provenance'], 'role': b.get('source_role', 'body')}
               for b in blocks]
    targets = [r for r in records if r['kind'] not in {'heading', 'inline_heading'}]
    if any(r['page'] is None for r in targets):
        raise ValueError('Body ownership requires page provenance')
    decisions = {}
    value = object_schema({'section_id': {'type': 'string', 'enum': list(alias_sid)},
                           'confirmed': {'type': 'boolean'}, 'reason': STRING})
    for page in sorted({r['page'] for r in targets}):
        keys = [r['id'] for r in targets if r['page'] == page]
        def valid(data):
            pending = [key for key, value in data.items() if not value['confirmed']]
            if pending:
                raise ValueError('Unresolved body ownership: ' + ', '.join(pending))
        decisions.update(request(client,
            'Assign each target body block to its DIRECT source section. Use printed layout, reading '
            'order, section boundaries and page continuation, not topic similarity. Context blocks '
            'are read-only. Keep parent introductions with their parent, not their children. '
            'Independent administrative/contact material without a heading belongs to the document '
            'root, not a nearby Contents heading. Do not change titles, text, roles or parent links. '
            'Do not invent sections. Return confirmed=false if source ownership cannot be determined.',
            {'outline': outline, 'document_root': sid_alias[sections[0]['section_id']], 'target_ids': keys,
             'records': [r for r in records if r['page'] is not None and abs(r['page'] - page) <= 1]},
            keyed_schema(keys, value), f'source:owners:page-{page + 1}', directory=directory, validate=valid))
    before = Counter((b['block_id'], b['text']) for b in blocks)
    for b in blocks:
        if aliases[b['block_id']] in decisions:
            decision = decisions[aliases[b['block_id']]]
            b['owner_before_review'] = b['section_id']
            b['section_id'] = alias_sid[decision['section_id']]
            b['ownership_reason'] = decision['reason']
    for s in sections:
        s['block_ids'] = [b['block_id'] for b in blocks if b['section_id'] == s['section_id']]
    owners = {b['block_id']: b['section_id'] for b in blocks}
    for unit in corpus['source_units']:
        unit['section_id'] = owners[unit['block_id']]
    for figure in corpus['figures']:
        figure['section_id'] = owners[figure['block_id']]
    assert before == Counter((b['block_id'], b['text']) for b in blocks)
    write_json(directory / 'ownership-decisions.json', {'decisions': decisions, 'aliases': aliases,
               'section_aliases': sid_alias, 'original_text_preserved': True})
    return corpus
