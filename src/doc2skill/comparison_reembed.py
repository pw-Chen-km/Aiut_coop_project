"""Reindex an immutable preparation without changing its source units or chunks."""
from copy import deepcopy
from pathlib import Path
import shutil
import time

from .config import fingerprint, read_jsonl, sha256, write_json
from .pipeline import load_corpus
from .storage import Store


def reembed_prepared(config, *, encoder=None):
    from .comparison_embedding import comparison_encoder
    from .comparison_pipeline import validate_prepared, _signature, load_sources, COMMON_FILES
    source = Path(config['reuse_prepared_dir']).resolve()
    root = Path(config['output_dir']).resolve()
    old = validate_prepared(source)
    sources = load_sources(config)
    if old['source_hashes'] != {d['source_path']: d['source_sha256'] for d in sources}:
        raise ValueError('Reembedding must use the same original PDFs')
    if old['llm_configuration'] != config['offline_llm']:
        raise ValueError('Reembedding cannot change the descriptor model configuration')
    if root.exists():
        raise FileExistsError('Reembedding requires a new output version')
    root.mkdir(parents=True)
    start = time.time()
    status = {'status': 'running', 'stage': 'reembedding', 'signature': _signature(config, sources),
              'started_at': start}
    write_json(root / 'prepare_status.json', status)
    try:
        encoder = encoder or comparison_encoder(config['embedding'])
        frozen = COMMON_FILES - {'corpus.sqlite', 'corpus.vectors.npy'}
        for relative in frozen:
            shutil.copyfile(source / relative, root / relative)
        corpus = load_corpus(root)
        texts = [c['embedding_text'] for c in corpus['chunks']]
        counts = [len(encoder.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        vectors = encoder.encode_documents(texts)
        with Store.create(root / 'corpus.sqlite', {**corpus, 'metadata': {'embedding': encoder.provenance}}, vectors):
            pass
        hashes = {p: sha256(root / p) for p in COMMON_FILES}
        if any(hashes[p] != old['common_hashes'][p] for p in frozen):
            raise ValueError('Reembedding changed fixed text/descriptors/chunks')
        manifest = deepcopy(old)
        manifest.update(common_hashes=hashes, fingerprint=fingerprint(hashes), signature=status['signature'],
                        embedding=encoder.provenance, elapsed_seconds=time.time() - start, llm_cost={})
        manifest['reuse'] = {'prepared_dir': str(source), 'prepared_fingerprint': old['fingerprint'],
            'prepared_manifest_sha256': sha256(source / 'prepared_manifest.json'),
            'unchanged_artifacts': sorted(frozen), 'original_preparation_llm_cost': old.get('llm_cost'),
            'chunk_token_count_model': old['embedding'], 'new_embedding_token_counts': counts,
            'policy': 'Exact original chunks and descriptors; only retrieval vectors/index replaced'}
        write_json(root / 'prepared_manifest.json', manifest)
        result = validate_prepared(root)
        status.update(status='complete', stage='complete', counts=result['counts'])
        return result
    except Exception as exc:
        status.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        status['elapsed_seconds'] = time.time() - start
        write_json(root / 'prepare_status.json', status)
