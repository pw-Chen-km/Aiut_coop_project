"""Versioned corpus-only adaptive build, separate from legacy builds and serving."""
from __future__ import annotations
import json
from pathlib import Path
from .config import read_jsonl, write_json, write_jsonl, sha256, fingerprint
from .pipeline import load_corpus, COLLECTIONS


def build_adaptive(config, *, encoder=None, client=None, token_counter=None, resource_guard=None):
    from .amd import AMDMetadataClient, GPUResourceGuard, ResourceBusy
    from .embedding import E5Encoder
    from .native_review import generate_structure_review
    from .normalization import normalize_structure, write_structure_report
    from .metadata_v2 import generate_metadata_v2
    from .adaptive import plan_navigation, write_navigation
    from .chunking import build_chunks
    from .storage import Store
    from .validation import validate_corpus
    source, out = Path(config['corpus_dir']).resolve(), Path(config['output_dir']).resolve()
    if source == out or source.is_relative_to(out) or out.is_relative_to(source) or out.exists():
        raise ValueError('Adaptive build requires a new, non-overlapping output directory')
    files = [source / (k + '.jsonl') for k in COLLECTIONS]
    if (source / 'native_spans.jsonl').exists():
        files.append(source / 'native_spans.jsonl')
    review_path = Path(config['structure_review']).resolve() if config.get('structure_review') else None
    if review_path:
        files.append(review_path)
    before = {str(p): sha256(p) for p in files}
    implementation = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    signature = fingerprint({'source': before, 'implementation': implementation, 'config': config})
    out.mkdir(parents=True)
    status = {'status': 'building', 'schema_version': 'adaptive-build-v2', 'source_hashes': before,
              'build_signature': signature, 'implementation_hashes': implementation}
    write_json(out / 'build_status.json', status)
    try:
        if client is None:
            client = AMDMetadataClient(config['offline_llm'], out / 'llm_audit')
            write_json(out / 'endpoint_preflight.json', client.transport.preflight())
        resource_guard = resource_guard or GPUResourceGuard(config['gpu_guard'], out / 'resource_checks')
        encoder = encoder or E5Encoder(config['embedding'], local_only=True)
        if token_counter is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['navigation']['tokenizer_path'], local_files_only=True,
                                                       trust_remote_code=False)
            token_counter = lambda s: len(tokenizer.encode(s, add_special_tokens=True))
        original = load_corpus(source)
        audits = []
        corpus = {k: [] for k in COLLECTIONS}
        if review_path:
            corpus, audit = normalize_structure(original, json.loads(review_path.read_text()))
            audits.append(audit)
        else:
            for d in original['documents']:
                did = d['doc_id']; resource_guard.check('structure:' + did)
                single = {k: [r for r in rows if r.get('doc_id') == did] for k, rows in original.items()}
                review, norm, audit = generate_structure_review(single, client, out / 'reviews' / did, general_topics=True)
                audits.append(audit)
                for k in COLLECTIONS:
                    corpus[k].extend(norm[k])
        corpus['chunks'] = build_chunks(None, corpus, config, encoder.tokenizer)
        check = validate_corpus(corpus, encoder.tokenizer, config['chunking'].get('max_tokens', 384))
        if not check['valid']:
            raise ValueError('Invalid adaptive corpus: ' + '; '.join(check['errors']))
        meta, support = generate_metadata_v2(corpus, client,
            {**config['metadata'], 'cache_dir': str(out / 'metadata_cache')}, resource_check=resource_guard.check)
        resource_guard.check('navigation-planning')
        nav = config.get('navigation', {})
        tree, budget = plan_navigation(corpus, meta, client, token_counter,
            max_input_tokens=nav.get('max_input_tokens', 7000), reserve_tokens=nav.get('reserve_tokens', 1536))
        write_navigation(tree, out)
        for k in COLLECTIONS:
            write_jsonl(out / (k + '.jsonl'), corpus[k])
        write_json(out / 'metadata.json', meta)
        write_json(out / 'metadata_support.json', support)
        write_json(out / 'navigation_budget_audit.json', budget)
        write_json(out / 'structure_audit.json', {'documents': audits,
            'block_assignments': [b for a in audits for b in a['block_assignments']]})
        write_structure_report(original, corpus, {'provenance': {'mode': 'llm_endpoint' if not review_path else 'replayed_review'}},
                               out / 'structure_report.md')
        if (source / 'native_spans.jsonl').exists():
            spans = read_jsonl(source / 'native_spans.jsonl')
            for span in spans:
                span['section_ids'] = sorted({b['section_id'] for b in corpus['blocks'] if b['doc_id'] == span['doc_id']
                    and max(span['start_sp'], b['doc_start_char']) < min(span['end_sp'], b['doc_end_char'])})
            write_jsonl(out / 'native_span_sections.jsonl', spans)
        vectors = encoder.encode_documents([c['embedding_text'] for c in corpus['chunks']])
        with Store.create(out / 'corpus.sqlite', {**corpus, 'metadata': {'embedding': encoder.provenance}}, vectors):
            pass
        status.update(status='complete', embedding=encoder.provenance, counts={k: len(v) for k, v in corpus.items()})
    except ResourceBusy as exc:
        status.update(status='paused_resource_busy', error=str(exc))
    except Exception as exc:
        status.update(status='blocked', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        status['source_hashes_unchanged'] = all(sha256(p) == h for p, h in before.items())
        if not status['source_hashes_unchanged']:
            status['status'] = 'blocked'
        write_json(out / 'build_status.json', status)
    if status['status'] == 'complete':
        write_json(out / 'manifest.json', {'schema_version': 'adaptive-build-v2', 'build_signature': signature,
            'status': 'complete', 'artifacts': {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                 if p.is_file() and p.name not in {'manifest.json', 'build_status.json'}}})
    return status
