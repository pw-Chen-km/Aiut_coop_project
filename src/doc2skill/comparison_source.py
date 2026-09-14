"""Audited source-layout repair before parent-only review; no semantic grouping."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from .config import sha256
from .structure import ordered_items, recover_native_pdf_cells, stable_id

POLICY = 'comparison-source-layout-v1'


def _geometry(item, raw):
    prov = next(iter(item.get('prov', [])), {})
    page = prov.get('page_no')
    size = raw.get('pages', {}).get(str(page), {}).get('size', {})
    box = prov.get('bbox', {})
    height = size.get('height')
    if not height or not all(k in box for k in ('l', 'r', 't', 'b')):
        return None
    if box.get('coord_origin') == 'BOTTOMLEFT':
        top, bottom = height - box['t'], height - box['b']
    elif box.get('coord_origin') == 'TOPLEFT':
        top, bottom = box['t'], box['b']
    else:
        return None
    return page, box['l'], top, box['r'], bottom, height


def _text(item):
    return ' '.join(item.get('text', '').split()).casefold()


def repair_items(raw, fragment_repairs=()):
    """Use repeated margin geometry and native title evidence, retaining all items.

    A unique topical page_header is restored only if it has heading-sized text.
    Repeated body phrases are never excluded. TOC text remains traceable but is
    marked navigation-only. Ambiguous table relationships are not guessed here.
    """
    items, _ = ordered_items(raw)
    items = deepcopy(items)
    for item in items:
        item['_preserve_heading'] = True
    occurrences = defaultdict(set)
    first_headings = {}
    for item in items:
        g = _geometry(item, raw)
        if g and _text(item):
            band = 'top' if g[2] / g[5] < .12 else 'bottom' if g[4] / g[5] > .88 else None
            if band:
                occurrences[(_text(item), band)].add(g[0])
            if band == 'top' and item.get('label') in {'title', 'section_header'} and item.get('_layer') != 'furniture':
                first_headings.setdefault(_text(item), item['self_ref'])
    toc_regions = [g for x in items if x.get('label') == 'document_index'
                   and (g := _geometry(x, raw))]
    toc_pages = {g[0] for g in toc_regions}
    audit = {'policy': POLICY, 'decisions': [], 'toc_pages': sorted(toc_pages),
             'toc_regions': toc_regions,
             'fragment_merges': [], 'limitations': ['No inferred repair of table cell relationships.']}
    for item in items:
        g, text = _geometry(item, raw), _text(item)
        if not g or not text:
            continue
        old_label = item.get('label')
        band = 'top' if g[2] / g[5] < .12 else 'bottom' if g[4] / g[5] > .88 else None
        repeated = band and len(occurrences[(text, band)]) >= 3
        file_identifier = text.count('-') + text.count('_') >= 2 and bool(re.search(r'\d', text))
        first_topic_heading = (band == 'top' and first_headings.get(text) == item['self_ref']
                               and not file_identifier)
        reason = None
        if repeated and not first_topic_heading:
            item.update(label='page_header' if band == 'top' else 'page_footer',
                        _layer='furniture', content_layer='furniture')
            reason = 'Same text appears in the same margin band on at least three pages'
        elif old_label == 'page_header' and g[2] / g[5] < .18 and g[4] - g[2] >= 12:
            # Large short topical headers are source headings, not repeated banners.
            candidate = (len(text.split()) <= 12 and re.search(r'[a-z]', text)
                         and not re.search(r'©|copyright|https?://|@|\d{4}[-_]', text)
                         and not text.endswith(('.', ':', ';')))
            if candidate:
                item.update(label='section_header', level=1, _layer='body', content_layer='body')
                reason = 'Non-running topical page header with heading-sized native geometry'
        in_toc = any(_inside(g, region) for region in toc_regions)
        if (in_toc or g[0] in toc_pages and text in {'contents', 'table of contents'}) and item.get('_layer') != 'furniture':
            item['_source_role'] = 'table_of_contents'
            if item.get('label') in {'section_header', 'title'} and text not in {'contents', 'table of contents'}:
                item['label'] = 'text'
                reason = 'TOC entry is an index entry, not a body section boundary'
        if reason:
            item['_source_repair'] = {'original_label': old_label, 'reason': reason}
            audit['decisions'].append({'native_ref': item['self_ref'], 'page_no': g[0],
                'text': item.get('text', ''), 'before': old_label, 'after': item['label'], 'reason': reason})

    # Promoted furniture was appended after body traversal. Insert it by native
    # page/y-position, keeping existing body reading order (including columns).
    promoted = [x for x in items if x.get('_source_repair', {}).get('original_label') == 'page_header'
                and x.get('label') == 'section_header']
    for item in promoted:
        items.remove(item)
        g = _geometry(item, raw)
        position = next((i for i, other in enumerate(items)
                         if other.get('_layer') != 'furniture' and (og := _geometry(other, raw))
                         and (og[0] > g[0] or og[0] == g[0] and og[2] >= g[4])), len(items))
        items.insert(position, item)
    merge_heading_fragments(items, raw, fragment_repairs, audit)
    return items, audit


def _inside(box, region):
    return (box[0] == region[0] and region[1] <= (box[1] + box[3]) / 2 <= region[3]
            and region[2] <= (box[2] + box[4]) / 2 <= region[4])


def merge_heading_fragments(items, raw, repairs, audit):
    """Separate explicit source repair; only adjacent, same-page heading lines.

    No model chooses merges. Callers provide native refs and a source-layout
    reason, which is validated geometrically and retained for every merge.
    Original item texts and references remain unchanged.
    """
    by_ref = {x['self_ref']: x for x in items}
    used = set()
    for repair in repairs:
        first, second, reason = (repair.get(k) for k in ('first_ref', 'second_ref', 'reason'))
        label = f'Heading fragment repair {first!r} + {second!r}'
        if not isinstance(reason, str) or not reason.strip() or first == second:
            raise ValueError(label + ': distinct refs and a nonempty source-layout reason are required')
        if first not in by_ref or second not in by_ref or {first, second} & used:
            raise ValueError(label + ': unknown or repeated native reference')
        a, b = by_ref[first], by_ref[second]
        ga, gb = _geometry(a, raw), _geometry(b, raw)
        if (a.get('label') not in {'title', 'section_header'} or b.get('label') not in {'title', 'section_header'}
                or a.get('_source_role') or b.get('_source_role') or not ga or not gb):
            raise ValueError(label + ': both items must be body headings with native geometry')
        body = [x for x in items if x.get('_layer') != 'furniture']
        height = max(ga[4] - ga[2], gb[4] - gb[2])
        if (body.index(b) != body.index(a) + 1 or ga[0] != gb[0]
                or abs(ga[1] - gb[1]) > 3 or not -2 <= gb[2] - ga[4] <= height
                or abs((ga[4] - ga[2]) - (gb[4] - gb[2])) > height * .25):
            raise ValueError(label + ': not adjacent aligned lines of the same source heading')
        a['_combined_title'] = a.get('text', '') + ' ' + b.get('text', '')
        a['_fragment_refs'] = [second]
        b['label'] = 'inline_heading'
        used.update((first, second))
        audit['fragment_merges'].append({**repair, 'page_no': ga[0], 'title': a['_combined_title']})


def prepare_source_layout(corpus, settings=None, *, client=None, directory=None):
    """Reproject verified Docling/native exports into a new, audited source layer."""
    from .parsing import normalize_docling_json
    from .pipeline import COLLECTIONS
    settings = settings or {}
    repairs = settings.get('heading_fragment_repairs', [])
    docs = {d['source_sha256']: d for d in corpus['documents']}
    if any(r.get('source_sha256') not in docs for r in repairs):
        raise ValueError('Heading fragment repair names an unknown source_sha256')
    output, audits = {k: [] for k in COLLECTIONS}, []
    for doc in corpus['documents']:
        for path_key, hash_key in [('docling_json_path', 'raw_export_sha256'),
                                   ('native_pdf_pages_path', 'native_pdf_pages_sha256')]:
            if not doc.get(path_key) or not doc.get(hash_key) or sha256(doc[path_key]) != doc[hash_key]:
                raise ValueError(doc['doc_id'] + ': verified raw Docling and native page exports required')
        raw = json.loads(Path(doc['docling_json_path']).read_text(encoding='utf8'))
        selected = [r for r in repairs if r['source_sha256'] == doc['source_sha256']]
        code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                       ('comparison_source.py', 'structure.py', 'parsing.py')}
        version = stable_id('parse', doc['parse_version'], POLICY, selected, code_hashes)
        if settings.get('review_blocks'):
            if client is None or directory is None:
                raise ValueError('Source block review requires model and audit directory')
            from .comparison_source_review import review_items, review_fragments, review_owners
            review_dir = Path(directory) / doc['doc_id']
            items, item_audit = repair_items(raw)
            items = review_items(items, raw, client, review_dir / 'roles', item_audit)
            review_fragments(items, raw, client, review_dir / 'fragments', item_audit, selected)
            fragment = normalize_docling_json(raw, doc, version, reviewed_items=items, reviewed_audit=item_audit)
        else:
            fragment = normalize_docling_json(raw, doc, version, source_layout=True, fragment_repairs=selected)
        native = json.loads(Path(doc['native_pdf_pages_path']).read_text(encoding='utf8'))
        recover_native_pdf_cells(fragment, native['pages'])
        toc = fragment['audit']['source_layout']['toc_regions']
        for block in fragment['blocks']:
            g = _geometry({'prov': block.get('provenance', [])}, raw)
            if g and any(_inside(g, region) for region in toc):
                block['source_role'] = 'table_of_contents'
        if settings.get('review_blocks'):
            review_owners(fragment, client, review_dir / 'owners')
        fragment['documents'][0]['source_layout_policy'] = POLICY
        audits.append(fragment['audit'])
        for k in COLLECTIONS:
            output[k].extend(fragment[k])
    return output, audits
