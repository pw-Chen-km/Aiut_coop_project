"""Conservative layout rules, targeted role review, then source-only hierarchy.

No summaries, semantic regrouping, source rewriting, or document-specific rules.
Uncertain decisions retain the original and remain visible in the audit.
"""
from collections import Counter
from copy import deepcopy
import re
from statistics import median

from .comparison_source import _geometry, _inside, repair_items
from .comparison_requests import request, keyed_schema, object_schema
from .structure import _table_rows, recover_native_pdf_cells
from .config import write_json

POLICY = 'source-sections-v1'
HEADINGS = {'title', 'section_header'}


def refine_rules(raw):
    items, audit = repair_items(raw)
    audit.update(refinement_policy=POLICY, rule_changes=[], role_decisions=[], parent_decisions=[], unresolved=[])
    # Grow a local footer component from native nontrivial footer text. A phone
    # number in body text or a distant bottom paragraph is not sufficient.
    seeds = [x for x in items if x.get('label') == 'page_footer' and len(x.get('text','').strip()) > 4
             and (g := _geometry(x, raw)) and g[2]/g[5] >= .87]
    linked = list(seeds)
    for _ in range(len(items)):
        added = []
        for x in items:
            if x in linked or x.get('_layer') == 'furniture' or not x.get('text','').strip():
                continue
            g = _geometry(x, raw)
            if not g or g[2]/g[5] < .87 or x.get('label') not in {'text', 'list_item'}:
                continue
            for seed in linked:
                h = _geometry(seed, raw)
                if g[0] != h[0]:
                    continue
                height = min(h[4]-h[2], 12)
                dx = max(0, g[1]-h[3], h[1]-g[3])
                dy = max(0, g[2]-h[4], h[2]-g[4])
                if g[4]-g[2] <= max(12, height*1.8) and dx <= 12 and dy <= max(12, height*2):
                    added.append((x, seed['self_ref']))
                    break
        if not added:
            break
        for x, evidence in added:
            audit['rule_changes'].append({'id':x['self_ref'], 'before':x['label'], 'after':'page_footer',
                'reason':'Small text connected to an established footer region in bottom 13% of page', 'evidence_ids':[evidence]})
            x.update(label='page_footer', _layer='furniture', content_layer='furniture')
            linked.append(x)
    # A separate numeric marker aligned immediately left of a named heading is
    # not an independent empty section. Retain it after that heading as a marker.
    for x in list(items):
        if x.get('label') not in HEADINGS or not re.fullmatch(r'\d+', x.get('text','').strip()):
            continue
        g = _geometry(x, raw)
        if not g:
            continue
        matches = [y for y in items if y is not x and y.get('label') in HEADINGS
                   and re.search(r'[A-Za-z]', y.get('text','')) and (h := _geometry(y, raw))
                   and h[0] == g[0] and 0 <= h[1]-g[3] <= 30
                   and abs((h[2]+h[4])/2-(g[2]+g[4])/2) <= min(g[4]-g[2],h[4]-h[2])/2]
        if len(matches) == 1:
            target = matches[0]
            audit['rule_changes'].append({'id':x['self_ref'], 'before':x['label'], 'after':'inline_heading',
                'reason':'Separate numeric marker on same line immediately left of a named heading',
                'evidence_ids':[target['self_ref']]})
            x.update(label='inline_heading', _source_role='illustration_label')
            items.remove(x)
            items.insert(items.index(target)+1, x)
    return items, audit


def candidates(items, raw):
    result = {}
    for x in items:
        text = x.get('text','').strip()
        g = _geometry(x, raw)
        if not text or not g or x.get('_source_role') or x.get('_layer') == 'furniture':
            continue
        reasons = []
        if x.get('label') in HEADINGS:
            if re.fullmatch(r'\d+',text):
                reasons.append('isolated numeric heading')
            if text.endswith((':','.')) or len(text.split()) >= 12:
                reasons.append('heading may be a sentence introducing body/list')
            if g[2]/g[5] < .12 and g[4]-g[2] < 11:
                reasons.append('small heading in top margin')
        elif x.get('label') in {'text','list_item'} and g[2]/g[5] > .87:
            reasons.append('body text in bottom margin not resolved by footer rule')
        if reasons:
            result[x['self_ref']] = reasons
    return result


def _toc(items):
    return [{'id':x['self_ref'], 'rows':_table_rows(x), 'provenance':x.get('prov',[])}
            for x in items if x.get('label') == 'document_index']


def review_roles(items, raw, client, directory, audit):
    result = deepcopy(items)
    suspects = candidates(items, raw)
    audit['role_candidates'] = suspects
    records = [{'id':x['self_ref'], 'text':x.get('text',''), 'label':x.get('label'),
                'geometry':_geometry(x, raw), 'level':x.get('level'), 'layer':x.get('_layer'),
                'role':x.get('_source_role')} for x in items if _geometry(x, raw)]
    by_id = {x['self_ref']:x for x in result}
    pages = sorted({_geometry(by_id[key],raw)[0] for key in suspects})
    value = object_schema({'role':{'type':'string','enum':['heading','body','page_header','page_footer','administrative','uncertain']},
        'reason':{'type':'string','minLength':1}, 'evidence_ids':{'type':'array','items':{'type':'string'},'minItems':1}})
    for page in pages:
        targets = [key for key in suspects if _geometry(by_id[key],raw)[0] == page]
        context = [r for r in records if abs(r['geometry'][0]-page) <= 1]
        known = {r['id'] for r in context} | {r['id'] for r in _toc(items)}
        page_value = deepcopy(value)
        page_value['properties']['evidence_ids']['items']['enum'] = sorted(known)
        def valid(data):
            if any(set(d['evidence_ids'])-known for d in data.values()):
                raise ValueError('Evidence must refer to supplied source IDs')
        decisions = request(client,
            'Review ONLY the targeted ambiguous source blocks. This is source section extraction, '
            'not semantic grouping. Use complete source text, geometry (page,left,top,right,bottom,page_height), '
            'neighboring blocks, and TOC. No page images are supplied: do not claim visual inspection. '
            'A sentence introducing an immediate list normally remains body under the existing topic. '
            'Meaningful task headings, even questions or short captions functioning as subheadings, may stay headings. '
            'Repeated document/version names in margins are page headers. Legal/contact material is administrative '
            'unless evidence establishes a footer. Do not demote technical body merely because it mentions a number '
            'or is near the bottom. Do not rewrite text, invent titles or assign parents. '
            'Use uncertain when the available layout/text does not decide. Cite actual input IDs as evidence.',
            {'target_ids':targets, 'candidate_reasons':{k:suspects[k] for k in targets},
             'records':context, 'table_of_contents':_toc(items)},
            keyed_schema(targets,page_value), f'source-sections:roles:page-{page}', directory=directory, validate=valid)
        for key,d in decisions.items():
            x = by_id[key]
            audit['role_decisions'].append({'id':key,'before':x.get('label'),**d})
            if d['role'] == 'uncertain':
                audit['unresolved'].append({'stage':'role','id':key,**d})
                continue
            if d['role']=='heading' and x.get('label') not in HEADINGS:
                # These body candidates were selected ONLY for possible footer
                # cleanup. This stage has no mandate to create new headings.
                audit['unresolved'].append({'stage':'blocked_heading_promotion','id':key,
                    'decision':d,'reason':'Footer review cannot promote existing body into a new section; original retained'})
                continue
            if d['role'] in {'page_header','page_footer'}:
                x.update(label=d['role'],_layer='furniture',content_layer='furniture')
            else:
                x.update(label='section_header' if d['role']=='heading' else 'text')
                if d['role']=='administrative':
                    x['_source_role']='administrative'
    assert Counter((x['self_ref'],x.get('text','')) for x in items) == Counter((x['self_ref'],x.get('text','')) for x in result)
    return result


def _number(title):
    match = re.match(r'^(\d+(?:\s*\.\s*\d+)*)(?:\s*\.\s*|\s+)(?=[A-Za-z])', title)
    return tuple(int(x) for x in re.findall(r'\d+',match[1])) if match else None


def validate_parents(sections, parents):
    order = {s['section_id']:i for i,s in enumerate(sections)}
    root = sections[0]['section_id']
    if set(parents) != set(order) or parents[root] is not None:
        raise ValueError('Every section needs one parent; original root remains root')
    for sid in order:
        if sid == root:
            continue
        if parents[sid] not in order or order[parents[sid]] >= order[sid]:
            raise ValueError(f'{sid}: parent must be an earlier source heading or root')
    # A source hierarchy cannot leave a chapter then resume it after a sibling.
    active = [root]
    for section in sections[1:]:
        sid = section['section_id']
        if parents[sid] not in active:
            raise ValueError(f'{sid}: parent subtree already closed; preserve contiguous source order')
        active = active[:active.index(parents[sid])+1] + [sid]


def parent_candidates(fragment, raw):
    """Flag layout contradictions, never expose every parent as editable."""
    sections=fragment['sections']
    by={s['section_id']:s for s in sections}
    geometry={}
    for s in sections:
        b=next((b for b in fragment['blocks'] if b['section_id']==s['section_id'] and b['kind']=='heading'),None)
        if b:
            g=_geometry({'prov':b['provenance']},raw)
            if g:
                geometry[s['section_id']]=g
    suspects={}
    part_indices=[]
    for i,s in enumerate(sections[1:],1):
        sid=s['section_id'];g=geometry.get(sid);p=geometry.get(s['parent_id'])
        if not g:
            continue
        if p and g[0]>p[0] and g[2]/g[5]<.35 and g[4]-g[2] > (p[4]-p[2])*1.15:
            suspects[sid]=['Heading on a later page is larger than its assigned parent']
        if i+1<len(sections):
            nxt=sections[i+1];h=geometry.get(nxt['section_id'])
            letters=''.join(c for c in s['title'] if c.isalpha())
            if (h and letters and letters.isupper() and len(s['title'].split())>=2
                    and 0<=h[0]-g[0]<=1 and abs(g[1]-h[1])<20
                    and g[4]-g[2]>(h[4]-h[2])*1.15 and s['level']>=nxt['level']):
                suspects.setdefault(sid,[]).append('Larger uppercase part heading has equal/deeper level than following smaller heading')
                part_indices.append(i)
    root=sections[0]['section_id']
    for j,i in enumerate(part_indices):
        end=part_indices[j+1] if j+1<len(part_indices) else len(sections)
        for s in sections[i+1:end]:
            if s['parent_id']==root:
                suspects.setdefault(s['section_id'],[]).append('Root-level heading within an unresolved larger part boundary')
    return suspects


def review_parents(fragment, items, raw, client, directory, audit):
    sections = fragment['sections']
    aliases = {s['section_id']:f'S{i:03d}' for i,s in enumerate(sections)}
    reverse = {v:k for k,v in aliases.items()}
    root = sections[0]['section_id']
    original = {s['section_id']:s['parent_id'] for s in sections}
    numbered = {}
    for s in sections[1:]:
        n = _number(s['title'])
        if n:
            numbered.setdefault(n,[]).append(s['section_id'])
    locks = {root:None}
    for n,ids in numbered.items():
        if len(ids)==1 and len(n)>1 and len(numbered.get(n[:-1],[]))==1:
            parent = numbered[n[:-1]][0]
            if list(aliases).index(parent) < list(aliases).index(ids[0]):
                locks[ids[0]] = parent
    rows = []
    for s in sections:
        own = [b for b in fragment['blocks'] if b['section_id']==s['section_id']]
        # Three complete source blocks, explicitly recorded as an excerpt. No
        # character clipping, fabricated summary, or hidden context truncation.
        excerpts = [b for b in own if b['kind'] not in {'heading','picture','inline_heading'}
                    and b.get('source_role') not in {'table_of_contents','administrative'} and b['text'].strip()][:3]
        rows.append({'id':aliases[s['section_id']], 'title':s['title'], 'current_parent':aliases.get(s['parent_id']),
            'native_level':s['level'], 'heading_provenance':[b['provenance'] for b in own if b['kind']=='heading'],
            'body_excerpt_policy':'first 3 complete nonempty direct source blocks',
            'body_excerpt':[{'id':b['native_ref'],'text':b['text']} for b in excerpts],
            'locked_parent':aliases.get(locks[s['section_id']]) if s['section_id'] in locks else 'not_locked'})
    suspects=parent_candidates(fragment,raw)
    audit['parent_candidates']=suspects
    targets = [s for s in sections[1:] if s['section_id'] not in locks and s['section_id'] in suspects]
    properties = {}
    for s in targets:
        prior = list(aliases.values())[:list(aliases).index(s['section_id'])]
        properties[aliases[s['section_id']]] = object_schema({'parent_id':{'type':'string','enum':prior},
            'action':{'type':'string','enum':['keep','change','uncertain']},'reason':{'type':'string','minLength':1}})
    def combine(data):
        parents = {**original,**locks}
        for key,d in data.items():
            if d['action']=='change':
                parents[reverse[key]] = reverse[d['parent_id']]
            elif reverse[d['parent_id']] != original[reverse[key]]:
                raise ValueError('keep/uncertain must return the current parent')
        validate_parents(sections,parents)
        return parents
    decisions = request(client,
        'Correct ONLY immediate parents of the target source headings. Preserve original order and section boundaries. '
        'No topic-based regrouping, new headings, merges, body moves or title changes. '
        'Use TOC, heading geometry, numbered hierarchy and body excerpts. Native levels/current parents may be wrong. '
        'A new large part heading starts a new section, not a child of the preceding tutorial merely because it follows it. '
        'Keep coherent existing parent relationships unless source evidence supports a correction. '
        'All non-target parents and numbered locks are FIXED. Only listed layout conflicts may change. '
        'Do not invent nesting because topics logically relate or one follows another. '
        'Every parent precedes its child; each subtree must occupy a contiguous interval '
        'of the supplied order. Do not leave a subtree and return to it later. '
        'Return action=keep when current parent is correct, action=change for a supported correction, '
        'or action=uncertain with the current parent when evidence is insufficient. '
        'No images are supplied; do not claim to see typography beyond provided geometry.',
        {'outline':rows,'target_ids':list(properties),'candidate_reasons':{aliases[k]:v for k,v in suspects.items()},'table_of_contents':_toc(items)},
        object_schema(properties),'source-sections:parents',directory=directory,validate=combine) if properties else {}
    parents = combine(decisions)
    by_id = {s['section_id']:s for s in sections}
    for s in sections:
        sid = s['section_id']
        s['parent_before_refinement'] = original[sid]
        s['parent_id'] = parents[sid]
        if sid != root:
            p = by_id[s['parent_id']]
            s.update(path=p['path']+[s['title']],level=p['level']+1)
        d = decisions.get(aliases[sid])
        audit['parent_decisions'].append({'id':sid,'before':original[sid],'after':parents[sid],
            'decision':d,'numbered_lock':sid in locks})
        if d and d['action']=='uncertain':
            audit['unresolved'].append({'stage':'parent','id':sid,**d})
        if d and d['action']=='change' and parents[sid]==original[sid]:
            audit['unresolved'].append({'stage':'inconsistent_parent_action','id':sid,'decision':d,
                'reason':'Model claims change but returns the current parent; no correction occurred'})
    for s in sections:
        own_pages = [p['page_index'] for b in fragment['blocks'] if b['section_id']==s['section_id']
                     for p in b.get('provenance',[]) if p.get('page_index') is not None]
        s.update(page_start=min(own_pages) if own_pages else None,page_end=max(own_pages) if own_pages else None)
    for s in reversed(sections[1:]):
        p=by_id[s['parent_id']]
        for key, fn in [('page_start',min),('page_end',max)]:
            if s[key] is not None:
                p[key]=s[key] if p[key] is None else fn(p[key],s[key])
    return fragment


def refine_document(raw, native_pages, document, parse_version, client, directory):
    from .parsing import normalize_docling_json
    items,audit=refine_rules(raw)
    write_json(directory/'rules.json',audit)
    items=review_roles(items,raw,client,directory/'roles',audit)
    fragment=normalize_docling_json(raw,document,parse_version,reviewed_items=items,reviewed_audit=audit)
    recover_native_pdf_cells(fragment,native_pages)
    # Preserve the existing TOC role for any fallback recovery within its region.
    for b in fragment['blocks']:
        g=_geometry({'prov':b.get('provenance',[])},raw)
        if g and any(_inside(g,region) for region in audit['toc_regions']):
            b['source_role']='table_of_contents'
        if b.get('source_role')=='administrative':
            b['section_id']=fragment['sections'][0]['section_id']
    for s in fragment['sections']:
        s['block_ids']=[b['block_id'] for b in fragment['blocks'] if b['section_id']==s['section_id']]
    owners={b['block_id']:b['section_id'] for b in fragment['blocks']}
    for unit in fragment['source_units']:
        unit['section_id']=owners[unit['block_id']]
    write_json(directory/'roles-applied.json',fragment)
    from .navigation_layout import repair_navigation_layout
    fragment,layout_audit=repair_navigation_layout(fragment,raw)
    audit['navigation_layout']=layout_audit
    audit['unresolved'].extend(layout_audit['pending'])
    write_json(directory/'navigation-layout.json',layout_audit)
    fragment=review_parents(fragment,items,raw,client,directory/'parents',audit)
    owners={b['block_id']:b['section_id'] for b in fragment['blocks']}
    block_ids=[b['block_id'] for b in fragment['blocks']]
    assert len(block_ids)==len(set(block_ids))
    assert Counter(block_ids)==Counter(bid for s in fragment['sections'] for bid in s['block_ids'])
    assert all(u['section_id']==owners[u['block_id']] for u in fragment['source_units'])
    audit['status']='needs_review' if audit['unresolved'] else 'complete'
    audit['original_items_preserved']=True
    write_json(directory/'audit.json',audit)
    write_json(directory/'result.json',fragment)
    return fragment,audit
