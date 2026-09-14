"""Import a frozen, reviewed PDF structure without repeating model correction."""
from pathlib import Path
import json
from .config import sha256
from .pipeline import COLLECTIONS


def reviewed_hashes(directory):
    root = Path(directory).resolve()
    paths = sorted(root.glob('[0-9][0-9]/result.json'))
    if not paths:
        raise ValueError('No reviewed source fragments')
    return {str(p): sha256(p) for p in paths}


def load_reviewed(directory, sources):
    hashes = reviewed_hashes(directory)
    expected = {s['doc_id']: s for s in sources}
    corpus = {k: [] for k in COLLECTIONS}
    audits, layouts, seen = [], [], set()
    for path, digest in hashes.items():
        fragment = json.loads(Path(path).read_text(encoding='utf8'))
        if len(fragment['documents']) != 1:
            raise ValueError('Each reviewed fragment must contain exactly one document')
        doc = fragment['documents'][0]
        did = doc['doc_id']
        if did in seen or did not in expected or doc['source_sha256'] != expected[did]['source_sha256']:
            raise ValueError('Reviewed source does not match the requested PDF corpus')
        if sha256(doc['source_path']) != doc['source_sha256'] or sha256(doc['docling_json_path']) != doc['raw_export_sha256']:
            raise ValueError('Reviewed original PDF or native parse changed')
        seen.add(did)
        assignments = []
        for block in fragment['blocks']:
            # Text and memberships remain frozen. Explicit source headings and
            # TOC records are navigation metadata, not duplicate direct body.
            role = ('navigation_only' if block.get('kind') in {'heading', 'inline_heading'}
                    or block.get('source_role') == 'table_of_contents' else 'content')
            block['build_role'] = role
            assignments.append({'block_id': block['block_id'], 'role': role,
                                'section_id': block['section_id'], 'reason': 'Frozen reviewed source role'})
        fragment['chunks'] = []
        for key in COLLECTIONS:
            corpus[key].extend(fragment[key])
        layouts.append({'doc_id': did, 'reviewed_file': path, 'sha256': digest,
                        'audit': fragment.get('audit', {}).get('source_layout', {})})
        audits.append({'doc_id': did, 'block_assignments': assignments,
                       'provenance': {'mode': 'frozen-reviewed-source', 'path': path, 'sha256': digest}})
    if seen != set(expected):
        raise ValueError('Reviewed corpus omitted requested PDFs')
    return corpus, audits, layouts


def write_reviewed_report(corpus, path):
    """Supply legacy report fields on a copy; frozen sections remain unchanged."""
    from copy import deepcopy
    from .normalization import write_structure_report
    view = deepcopy(corpus)
    for section in view['sections']:
        section['original_section_ids'] = [section['section_id']]
        section['normalization_reason'] = 'Reused frozen reviewed structure; no new parent correction'
    write_structure_report(corpus, view, {'provenance': {'mode': 'frozen-reviewed-source'}}, path)
