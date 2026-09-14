"""Explicit MultiDoc2Dial assignments -> existing genuine joint SkillOPT adapter.

No automatic optimization split, no original validation/test question access.
"""
from pathlib import Path

from doc2skill.config import sha256
from qa_agent.bundle import validate_serving_bundle, local_asset
from qa_benchmarks.multidoc_gt import load_inputs, build_native_oracle, qa_records, routing_labels
from .data import digest
from .io import new_directory, read_json, write_new, write_jsonl_new
from .oracle import load_corpus
from .joint_feedback import required_scope_sets


def prepare_multidoc_joint(root, bundle_dir, assignments_path, out_dir, *, view='opening-answer'):
    """Assignments JSON: {optimization: [qid, ...], validation: [qid, ...]}.

    Both groups MUST come from official train. Official validation remains held
    out. Shared documents/evidence are allowed; same dialogue/exact questions are
    not. Near-duplicate semantic review remains an explicitly reported limitation.
    """
    bundle = Path(bundle_dir).resolve()
    validate_serving_bundle(bundle)
    paths = read_json(bundle / 'skill_paths.json')
    if not paths.get('adaptive_tree'):
        raise ValueError('MultiDoc2Dial joint preparation requires a completed adaptive serving bundle')
    tree_file = local_asset(bundle, paths['adaptive_tree'])
    tree = read_json(tree_file); corpus = load_corpus(bundle)
    assignments = read_json(assignments_path)
    if set(assignments) != {'optimization', 'validation'}:
        raise ValueError('Explicit optimization and validation qid lists required; no held-out input')
    for ids in assignments.values():
        if not isinstance(ids, list) or not ids or any(not isinstance(q, str) for q in ids) or len(set(ids)) != len(ids):
            raise ValueError('Phase assignments must be nonempty unique qid lists')
    if set(assignments['optimization']) & set(assignments['validation']):
        raise ValueError('Overlapping phase assignments')
    inputs, references, native, docs, hashes = load_inputs(root, 'train', view)
    by_input = {r['qid']: r for r in inputs}; by_ref = {r['qid']: r for r in references}
    requested = set(assignments['optimization'] + assignments['validation'])
    if not requested <= by_input.keys():
        raise ValueError('Assigned qids must belong to the chosen official train view')
    conversations, questions = {}, {}
    for phase, ids in assignments.items():
        conversations[phase] = {(by_ref[q]['domain'], by_ref[q]['dialogue_id']) for q in ids}
        questions[phase] = {' '.join(by_input[q]['question'].casefold().split()) for q in ids}
    if conversations['optimization'] & conversations['validation']:
        raise ValueError('Same dialogue cannot cross optimization/validation')
    if questions['optimization'] & questions['validation']:
        raise ValueError('Exact duplicate questions cannot cross optimization/validation')
    if tree['section_documents'] != {s['section_id']: s['doc_id'] for s in corpus['sections']}:
        raise ValueError('Fixed tree and current corpus disagree')
    prepared = {}
    for phase, ids in assignments.items():
        selected_inputs, selected_refs = [by_input[q] for q in ids], [by_ref[q] for q in ids]
        oracle = build_native_oracle(selected_refs, native, docs, corpus)
        labels = {q: required_scope_sets(oracle['queries'][q], tree, corpus['blocks']) for q in ids}
        if not any(labels.values()):
            raise ValueError('No exactly aligned routing labels in ' + phase)
        prepared[phase] = (qa_records(selected_inputs, selected_refs, oracle), oracle, labels)
    bundle_hash, tree_hash = sha256(bundle / 'bundle_manifest.json'), sha256(tree_file)
    preparation_id = digest({'source_hashes': hashes, 'bundle': bundle_hash, 'assignments': assignments, 'view': view})
    out = new_directory(out_dir, protected=(root, bundle, assignments_path))
    report = {'schema_version': 'joint-preparation-v2', 'dataset': 'multidoc2dial',
        'base_bundle_hash': bundle_hash, 'tree_hash': tree_hash, 'preparation_id': preparation_id,
        'source_hashes': hashes, 'assignments_sha256': sha256(assignments_path),
        'official_validation_read': False, 'test_read': False, 'phases': {},
        'view': view, 'human_reviewed': False, 'near_duplicate_review': 'not_performed',
        'shared_evidence_allowed': True, 'generation_requested': False}
    for phase, (rows, oracle, labels) in prepared.items():
        write_jsonl_new(out / f'{phase}.jsonl', rows)
        oracle.update(phase=phase, preparation_id=preparation_id)
        oracle.pop('oracle_sha256', None); oracle['oracle_sha256'] = digest(oracle)
        write_new(out / f'oracle_{phase}.json', oracle)
        known = sorted(q for q, label in labels.items() if label)
        phase_manifest = {'phase': phase, 'preparation_id': preparation_id, 'sha256': sha256(out / f'{phase}.jsonl'),
            'eligible_count': len(rows), 'assigned_count': len(rows), 'qids': [q['qid'] for q in rows],
            'oracle_file_sha256': sha256(out / f'oracle_{phase}.json'), 'bundle_manifest_sha256': bundle_hash}
        write_new(out / f'{phase}_manifest.json', phase_manifest)
        write_jsonl_new(out / f'routing_labels_{phase}.jsonl', routing_labels(oracle, corpus, tree))
        report['phases'][phase] = {'assigned': len(rows), 'answer_eligible': len(rows),
            'route_eligible_ids': known, 'route_unknown_ids': sorted(set(labels)-set(known)),
            'oracle_summary': oracle['summary']}
    write_new(out / 'joint_manifest.json', report)
    return report
