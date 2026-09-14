"""Offline native GT adapter. No models, QA synthesis, or automatic split selection.

Official doc_text offsets are the authority, not heading names or chunk IDs.
This module is never imported by offline corpus construction or online QA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from doc2skill.config import sha256
from qa_skillopt.data import digest
from qa_skillopt.io import new_directory, read_json, read_jsonl, write_new, write_jsonl_new
from qa_skillopt.oracle import load_corpus, normalize_with_offsets, project_chunk_slice
from qa_skillopt.trajectory import evidence_coverage, diagnose_trajectory


def _unique(rows, key, what):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError('Duplicate ' + what + ': ' + str(identity))
        result[identity] = row
    return result


def corpus_fingerprint(corpus):
    keys = {'documents': 'doc_id', 'sections': 'section_id', 'blocks': 'block_id', 'chunks': 'chunk_id'}
    return digest({kind: sorted(corpus[kind], key=lambda r: r[key]) for kind, key in keys.items()})


def load_inputs(root, split, view='opening-answer'):
    """Read exactly one official split and corpus labels, never test/other QAs."""
    if split not in {'train', 'validation'} or view not in {'all', 'opening-answer'}:
        raise ValueError('Choose an official train/validation split and a supported view')
    root = Path(root).resolve()
    manifest = read_json(root / 'manifest.json')
    names = [f'evaluation/{split}.inputs.jsonl', f'evaluation/{split}.references.jsonl',
             'corpus/native_spans.jsonl', 'corpus/documents.jsonl']
    hashes = {}
    for name in names:
        hashes[name] = sha256(root / name)
        if manifest.get('artifacts', {}).get(name) != hashes[name]:
            raise ValueError('Benchmark input checksum mismatch: ' + name)
    inputs = read_jsonl(root / names[0]); refs = read_jsonl(root / names[1])
    by_input = _unique(inputs, lambda r: r['qid'], 'input qid')
    by_ref = _unique(refs, lambda r: r['qid'], 'reference qid')
    if by_input.keys() != by_ref.keys() or any(r['split'] != split for r in refs):
        raise ValueError('Input/reference split mismatch')
    if any(set(r) != {'qid', 'question', 'history'} for r in inputs):
        raise ValueError('Unexpected gold field in online input')
    keep = {qid for qid, r in by_input.items() if view == 'all' or
            (not r['history'] and by_ref[qid]['response_type'] == 'answer')}
    return ([r for r in inputs if r['qid'] in keep], [r for r in refs if r['qid'] in keep],
            read_jsonl(root / names[2]), read_jsonl(root / names[3]), hashes)


def build_native_oracle(references, native_spans, original_documents, corpus):
    """Map each annotation through original document coordinates to current blocks.

    All annotated spans are jointly required (no invented alternative evidence).
    Only edge/inter-block whitespace may be omitted. Changed text, missing content,
    unknown documents and ambiguous source ownership remain explicitly unknown.
    Mapping does not depend on whether one chunk can contain a whole annotation.
    """
    docs = _unique(original_documents, lambda d: (d['domain'], d['native_doc_id']), 'native document')
    spans = _unique(native_spans, lambda s: (s['domain'], s['native_doc_id'], str(s['id_sp'])), 'native span')
    current_docs = _unique(corpus['documents'], lambda d: d['doc_id'], 'document')
    sections = _unique(corpus['sections'], lambda s: s['section_id'], 'section')
    blocks = _unique(corpus['blocks'], lambda b: b['block_id'], 'block')
    chunks = _unique(corpus['chunks'], lambda c: c['chunk_id'], 'chunk')
    by_doc, by_block, index_spans = defaultdict(list), defaultdict(list), defaultdict(list)
    for block in blocks.values():
        by_doc[block['doc_id']].append(block)
    # A wrong index must not produce plausible but false evidence scores.
    for chunk in chunks.values():
        projected = project_chunk_slice(chunk, 0, len(chunk['text']))
        for span in projected:
            block = blocks.get(span['block_id'])
            if (block is None or block['doc_id'] != chunk['doc_id'] or
                    block['section_id'] != chunk['section_id'] or block.get('page_index') is not None or
                    block.get('source_type') != 'html' or
                    block['text'][span['start_char']:span['end_char']] != span['text']):
                raise ValueError('Chunk disagrees with native source registry')
            by_block[span['block_id']].append((chunk['chunk_id'], span))
            index_spans[chunk['doc_id']].append(span)
        for span in chunk['source_spans']:
            block = blocks[span['block_id']]
            if (span.get('doc_start_char') != block['doc_start_char'] + span['source_start_char'] or
                    span.get('doc_end_char') != block['doc_start_char'] + span['source_end_char']):
                raise ValueError('Chunk document coordinates disagree with block coordinates')

    cache = {}
    def locate(domain, ref):
        key = (domain, ref['doc_id'], str(ref['id_sp']))
        if key in cache:
            return deepcopy(cache[key])
        evidence = {'evidence_id': 'mdd_ev_' + digest(key)[:24], 'native_reference': deepcopy(ref),
                    'status': 'unknown', 'targets': [], 'reason': 'missing_native_span'}
        doc, native = docs.get(key[:2]), spans.get(key)
        if doc is None or native is None:
            return evidence
        did, text = doc['doc_id'], doc['doc_text']
        evidence['gold'] = {'doc_id': did, 'native_doc_id': ref['doc_id'], 'id_sp': str(ref['id_sp']),
                            'page_index': None, 'start_char': native['start_sp'], 'end_char': native['end_sp']}
        a, b = native['start_sp'], native['end_sp']
        current = current_docs.get(did)
        reason = None
        if (hashlib.sha256(text.encode()).hexdigest() != doc['source_sha256'] or not current or
                current.get('source_sha256') != doc['source_sha256'] or
                current.get('doc_text') != text or current.get('native_doc_id') != ref['doc_id'] or
                current.get('domain') != domain or native.get('doc_id') != did):
            reason = 'document_identity_mismatch'
        elif not (type(a) is int and type(b) is int and 0 <= a < b <= len(text)):
            reason = 'invalid_native_offsets'
        elif (native.get('canonical_text') != text[a:b] or
              normalize_with_offsets(native.get('text_sp', ''))[0] != normalize_with_offsets(text[a:b])[0]):
            reason = 'annotation_text_mismatch'
        elif not text[a:b].strip():
            reason = 'empty_annotation'
        if reason:
            evidence['reason'] = reason
            cache[key] = evidence
            return deepcopy(evidence)
        evidence['gold']['quote'] = text[a:b]
        required, pieces, owners, overlaps = [], [], set(), []
        for block in by_doc[did]:
            x, y = block.get('doc_start_char'), block.get('doc_end_char')
            if (type(x) is not int or type(y) is not int or not 0 <= x < y <= len(text) or
                    block['text'] != text[x:y]):
                reason = 'invalid_current_block_coordinates'; break
            left, right = max(a, x), min(b, y)
            if left >= right or not text[left:right].strip():
                continue
            if (block.get('build_role', 'content') != 'content' or
                    sections.get(block['section_id'], {}).get('doc_id') != did):
                reason = 'evidence_not_in_indexable_section'; break
            # Whitespace omitted by chunking does not erase evidence words.
            left += len(text[left:right]) - len(text[left:right].lstrip())
            right = left + len(text[left:right].rstrip())
            overlaps.append((left, right, block))
        cursor = a
        for left, right, block in sorted(overlaps, key=lambda v: (v[0], v[1])):
            if left < cursor:
                reason = 'ambiguous_overlapping_blocks'; break
            if text[cursor:left].strip():
                reason = 'unmapped_evidence_characters'; break
            x = block['doc_start_char']
            required.append({'block_id': block['block_id'], 'page_index': None, 'source_type': 'html',
                'start_char': left-x, 'end_char': right-x, 'doc_start_char': left, 'doc_end_char': right})
            pieces.append(text[left:right]); owners.add(block['section_id']); cursor = right
        if text[cursor:b].strip() or not required:
            reason = reason or 'unmapped_evidence_characters'
        if reason:
            evidence['reason'] = reason
        else:
            hits = set()
            for need in required:
                hits.update(cid for cid, span in by_block[need['block_id']] if
                    max(need['start_char'], span['start_char']) < min(need['end_char'], span['end_char']))
            target = {'doc_id': did, 'document_sha256': doc['source_sha256'], 'page_index': None,
                'source_spans': required, 'section_ids': sorted(owners), 'chunk_ids': sorted(hits),
                'canonical_text': ' '.join(pieces), 'match_method': 'native_document_offsets',
                'chunk_ids_semantics': 'overlapping candidates, not indivisible required chunks'}
            evidence.update(status='mapped', reason='exact_native_coordinates', targets=[target])
        cache[key] = evidence
        return deepcopy(evidence)

    queries = {}
    for row in references:
        if row['qid'] in queries:
            raise ValueError('Duplicate reference qid')
        mapped = [locate(row['domain'], r) for r in row['references']]
        mapped = list({e['evidence_id']: e for e in mapped}.values())
        complete = bool(mapped) and all(e['targets'] for e in mapped)
        query = {'qid': row['qid'], 'evidence': mapped,
            'routes': [{'set_id': 'official_all_spans', 'required_keys': [e['evidence_id'] for e in mapped]}],
            'status': 'mapped' if complete else 'partial' if any(e['targets'] for e in mapped) else 'unknown',
            'complete_route_available': complete}
        coverage = evidence_coverage(query, spans_by_document=index_spans)
        query['index_coverage'] = {'status': coverage['status'],
            'recall': coverage['recall'] if coverage['status'] == 'known' else None,
            'complete': coverage['complete'] if coverage['status'] == 'known' else None}
        queries[row['qid']] = query
    counts = Counter(e['reason'] for q in queries.values() for e in q['evidence'])
    result = {'schema_version': 'multidoc2dial-native-oracle-v1', 'optimizer_only': True,
        'policy': 'Exact official document offsets; whitespace-only omissions; all annotated spans required. '
                  'No fuzzy matching or invented alternatives. Unknown is not failure.',
        'corpus_sha256': corpus_fingerprint(corpus), 'queries': queries,
        'summary': {'query_count': len(queries), 'mapped_query_count': sum(q['complete_route_available'] for q in queries.values()),
                    'index_complete_query_count': sum(q['index_coverage']['complete'] is True for q in queries.values()),
                    'evidence_reason_counts': dict(counts), 'chunk_count': len(chunks)}}
    result['oracle_sha256'] = digest(result)
    return result


def qa_records(inputs, references, oracle):
    """Private judge/adapter records; online uses inputs.jsonl, never this file."""
    by_ref = {r['qid']: r for r in references}
    output = []
    for q in inputs:
        ref = by_ref[q['qid']]
        evidence = [{**e.get('gold', {}), 'evidence_id': e['evidence_id']}
                    for e in oracle['queries'][q['qid']]['evidence'] if 'quote' in e.get('gold', {})]
        output.append({**deepcopy(q), 'answer': ref['reference_answer'], 'question_type': 'qa',
                       'evidence': evidence, 'benchmark_reference': deepcopy(ref)})
    return output


def routing_labels(oracle, corpus, tree=None):
    from qa_skillopt.joint_feedback import required_scope_sets
    sections = {s['section_id']: s for s in corpus['sections']}
    registry = tree or {'section_documents': {s: v['doc_id'] for s, v in sections.items()}}
    for qid, query in oracle['queries'].items():
        required = required_scope_sets(query, registry, corpus['blocks'])
        rows = sorted(set(s for alt in required or [] for s in alt))
        yield {'qid': qid, 'status': 'known' if required else 'unknown',
               'acceptable_required_section_sets': required,
               'sections': [{k: sections[s][k] for k in ('section_id', 'doc_id', 'title', 'path')} for s in rows],
               'navigation_node_ids_by_section': {s: [n for n, v in tree['nodes'].items()
                   if s in v['own_section_ids']] for s in rows} if tree else None}


def export_gt(root, corpus_dir, output, *, split='train', view='opening-answer'):
    inputs, refs, spans, docs, hashes = load_inputs(root, split, view)
    corpus = load_corpus(corpus_dir)
    oracle = build_native_oracle(refs, spans, docs, corpus)
    tree_path = Path(corpus_dir) / 'navigation_tree.json'
    tree = read_json(tree_path) if tree_path.is_file() else None
    if tree:
        from doc2skill.adaptive import validate_tree
        validate_tree(tree)
        if tree['section_documents'] != {s['section_id']: s['doc_id'] for s in corpus['sections']}:
            raise ValueError('Navigation tree and corpus disagree')
    out = new_directory(output, protected=(root, corpus_dir))
    write_jsonl_new(out / 'inputs.jsonl', inputs)
    write_jsonl_new(out / 'references.jsonl', refs)
    write_jsonl_new(out / 'qa.private.jsonl', qa_records(inputs, refs, oracle))
    write_new(out / 'oracle.json', oracle)
    write_jsonl_new(out / 'routing_labels.jsonl', routing_labels(oracle, corpus, tree))
    report = {'schema_version': 'multidoc2dial-gt-export-v1', 'split': split, 'view': view,
        'candidate_only_not_human_reviewed': view == 'opening-answer', 'new_split_created': False,
        'source_hashes': hashes, 'corpus_sha256': oracle['corpus_sha256'],
        'tree_sha256': sha256(tree_path) if tree else None,
        'status': 'mapped_to_tree' if tree and corpus['chunks'] else 'source_mapping_only',
        'generation_requested': False, 'evaluation_performed': False, **oracle['summary']}
    report['unresolved_qids'] = [qid for qid, q in oracle['queries'].items() if not q['complete_route_available']]
    write_new(out / 'report.json', report)
    with (out / 'report.md').open('x') as f:
        f.write('# MultiDoc2Dial GT 對照\n\n'
                f'- 官方 split：{split}；題目視圖：{view}\n'
                f'- 題目：{len(inputs)}；完整來源對齊：{report["mapped_query_count"]}\n'
                f'- 現有全部 chunks 可完整涵蓋證據：{report["index_complete_query_count"]}\n'
                f'- 狀態：{report["status"]}\n\n'
                '這是資料對照，不是模型評估。來源對齊不代表檢索成功或答案正確。\n'
                'opening-answer 只是依原標註筛選的候選，尚未人工確認可獨立回答。\n'
                'source_mapping_only 表示尚未對應完成建置的導航樹／索引，不能直接啟動 SkillOPT。\n'
                'unknown 保留在 oracle.json 並列於 report.json，不當成導航失敗。\n')
    write_new(out / 'manifest.json', {'source_hashes': hashes, 'corpus_sha256': oracle['corpus_sha256'],
        'artifacts': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report


def score_traces(gt_dir, corpus_dir, trajectory_path, output):
    """Explicit read-only evaluation of existing logs. Does not invoke QA/judges."""
    root = Path(gt_dir); manifest = read_json(root / 'manifest.json')
    if sha256(root / 'oracle.json') != manifest['artifacts']['oracle.json']:
        raise ValueError('Oracle export fingerprint mismatch')
    oracle = read_json(root / 'oracle.json'); corpus = load_corpus(corpus_dir)
    if corpus_fingerprint(corpus) != oracle['corpus_sha256']:
        raise ValueError('Trace scoring must use the same fixed corpus as the oracle')
    path = Path(trajectory_path)
    traces = read_jsonl(path) if path.suffix == '.jsonl' else [read_json(path)]
    _unique(traces, lambda t: t['qid'], 'trajectory qid')
    indexed = {(c['doc_id'], c['chunk_id']): c for c in corpus['chunks']}
    reports = []
    for trace in traces:
        if trace['qid'] not in oracle['queries']:
            raise ValueError('Trajectory is outside the selected GT export')
        # A corpus path alone does not prove the trace actually used that index.
        for row in trace.get('rounds') or [trace]:
            for item in [*row.get('items', []), *row.get('generation_items', [])]:
                expected = indexed.get((item.get('doc_id'), item.get('chunk_id')))
                if (expected is None or item.get('text') != expected['text'] or
                        item.get('source_spans') != expected['source_spans']):
                    raise ValueError('Trajectory chunk differs from the fixed index')
        value = diagnose_trajectory(trace, oracle['queries'][trace['qid']], corpus['sections'])
        for row in value['rounds']:
            if row['provenance_errors']:
                raise ValueError('Invalid trajectory source provenance')
            for k, coverage in row.items():
                if k.endswith('_coverage') and isinstance(coverage, dict) and coverage.get('status') != 'known':
                    coverage.update(recall=None, complete=None)
        reports.append(value)
    out = new_directory(output, protected=(gt_dir, corpus_dir, trajectory_path))
    write_jsonl_new(out / 'diagnostics.jsonl', reports)
    known = [r for r in reports if r['oracle_status'] == 'mapped' and r['rounds'][-1]['status'] == 'ok']
    summary = {'case_count': len(reports), 'scorable_count': len(known),
        'unknown_count': sum(r['oracle_status'] != 'mapped' for r in reports),
        'execution_failures': sum(r['failure_stage'] == 'execution_failure' for r in reports),
        'evidence_recall': sum(r['rounds'][-1]['cumulative_retrieved_coverage']['recall'] for r in known)/len(known) if known else None,
        'evidence_complete': sum(r['rounds'][-1]['cumulative_retrieved_coverage']['complete'] for r in known)/len(known) if known else None,
        'trace_sha256': sha256(path), 'oracle_sha256': sha256(root / 'oracle.json'),
        'metric': 'macro fraction of complete official spans in cumulative retrieved items; no answer judgment'}
    write_new(out / 'summary.json', summary)
    return summary
