"""Budgeted embedding cache and training-only retrieval for weak anchors."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from .prepare import ROOT, PUBLISHED_POOL, sha256, write_jsonl
from wwii_pilot.build_shared_pool import read_jsonl

MODEL = 'text-embedding-3-small'
DIMENSIONS = 1536
INPUT_RATE = .02  # https://developers.openai.com/api/docs/models/text-embedding-3-small
QUERIES = {
 'A23': ['A wartime blockade or shipping disruption deprives a country of imported food, oil, rubber or metals, causing resource shortages.',
         'Economic warfare through blockading imports and restricting supplies during the Second World War.'],
 'A25': ['Military offensives, tanks, aircraft or ships were halted or restricted because fuel supplies ran out or oil was scarce.',
         'Fuel shortages, inadequate oil stocks and petrol logistics constrained World War II military operations.'],
 'A27': ['Winston Churchill gave wartime speeches and broadcasts to rally British public morale, courage and determination to resist Nazi Germany.',
         'King George VI and Queen Elizabeth visited bombed neighborhoods and inspired British civilian morale during the Blitz.'],
 'A28': ['Hitler overruled his generals, forbade retreat, changed military strategy and interfered in the command of World War II operations.',
         'Hitler personally intervened in military decisions against the advice of his commanders, ordering troops to hold positions.'],
}


def decode_vectors(document, count):
    data = document['data']
    if (len(data) != count or any(type(r['index']) is not int for r in data)
            or {r['index'] for r in data} != set(range(count))):
        raise ValueError('embedding response row indices differ')
    vectors = []
    for entry in sorted(data, key=lambda r: r['index']):
        value = entry['embedding']
        vector = (np.frombuffer(base64.b64decode(value, validate=True), dtype='<f4').copy()
                  if isinstance(value, str) else np.asarray(value, dtype=np.float32))
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError('invalid embedding shape or values')
        vectors.append(vector)
    return np.stack(vectors)


def retrieve(api_key, budget=4.50, workers=2, extra_file=None):
    from .label_api import Ledger
    ledger = Ledger(ROOT / 'work/api_labels.sqlite3', budget)
    cache = ROOT / 'work/embeddings'
    cache.mkdir(exist_ok=True)
    pacing_lock = threading.Lock()
    next_slot = [time.monotonic()]

    def embed(texts):
        encoded = json.dumps({'model': MODEL, 'dimensions': DIMENSIONS, 'encoding_format': 'base64',
                              'input': texts}, ensure_ascii=False).encode()
        key = hashlib.sha256(encoded).hexdigest()
        path = cache / (key + '.npy')
        if path.exists():
            vectors = np.load(path)
            if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
                raise ValueError('invalid embedding cache')
            return vectors
        # Pacing by text bytes is conservative for this English-language corpus.
        with pacing_lock:
            slot = max(time.monotonic(), next_slot[0])
            next_slot[0] = slot + sum(len(t.encode()) for t in texts) / 50000
        time.sleep(max(0, slot - time.monotonic()))
        for attempt in range(3):
            rid = ledger.reserve(MODEL, (len(encoded) + 1024) * INPUT_RATE / 1_000_000)
            request = urllib.request.Request('https://api.openai.com/v1/embeddings', data=encoded,
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    document = json.load(response)
            except urllib.error.HTTPError as error:
                ledger.settle(rid, None if error.code >= 500 else 0, 'embedding_http_' + str(error.code))
                if error.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise RuntimeError(f'embedding API HTTP {error.code}; no credential logged') from None
            except (OSError, TimeoutError):
                ledger.settle(rid, None, 'embedding_unknown_network_outcome')
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise RuntimeError('embedding network error; reserved cost retained') from None
            usage = document.get('usage')
            tokens = int(usage['prompt_tokens']) if usage else None
            if tokens is not None and tokens < 0:
                raise ValueError('invalid embedding usage; reservation retained')
            actual = tokens * INPUT_RATE / 1_000_000 if tokens is not None else None
            ledger.settle(rid, actual, 'embedding_received', usage)
            vectors = decode_vectors(document, len(texts))
            temporary = path.with_suffix('.tmp.npy')
            np.save(temporary, vectors)
            temporary.replace(path)
            return vectors

    if extra_file is not None:
        extra = read_jsonl(extra_file)
        if not extra or len({r['chunk_id'] for r in extra}) != len(extra):
            raise ValueError('extra embedding inputs must be nonempty and unique')
        batches = [extra[i:i + 128] for i in range(0, len(extra), 128)]
        values = np.vstack([embed([r['chunk_text'] for r in batch]) for batch in batches])
        np.save(cache / 'extra_vectors.npy', values)
        (cache / 'extra_manifest.json').write_text(json.dumps({'model': MODEL, 'dimensions': DIMENSIONS,
            'ids': [r['chunk_id'] for r in extra], 'input_sha256': sha256(extra_file)}) + '\n')
        return {'embedded_extra': len(extra), 'dimensions': DIMENSIONS, 'budget': ledger.summary()}
    candidates_path = ROOT / 'data/cleaned_candidates.jsonl'
    rows = read_jsonl(candidates_path)
    vector_path = cache / 'vectors.npy'
    metadata_path = cache / 'manifest.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['candidate_sha256'] != sha256(candidates_path) or metadata['model'] != MODEL:
            raise ValueError('embedding cache belongs to a different candidate pool')
        vectors = np.load(vector_path, mmap_mode='r')
    else:
        batches = [rows[i:i + 128] for i in range(0, len(rows), 128)]
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, result in enumerate(executor.map(lambda batch: embed([r['chunk_text'] for r in batch]), batches), 1):
                results.append(result)
                if i % 10 == 0:
                    print(json.dumps({'embedding_batches': i, 'total_batches': len(batches), **ledger.summary()}), flush=True)
        vectors = np.vstack(results)
        np.save(vector_path, vectors)
        metadata = {'model': MODEL, 'dimensions': DIMENSIONS, 'candidate_sha256': sha256(candidates_path),
                    'ids': [r['chunk_id'] for r in rows], 'dtype': 'float32', 'normalization': 'API output; cosine normalization used for retrieval'}
        metadata_path.write_text(json.dumps(metadata) + '\n')
    if vectors.shape != (len(rows), DIMENSIONS):
        raise ValueError('embedding matrix shape differs from candidates')
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    frozen = json.loads((ROOT / 'data/frozen_source_splits.json').read_text())
    existing_path = ROOT / 'data/selected_enriched_candidates.jsonl'
    existing = read_jsonl(existing_path)
    used = {r['chunk_id'] for r in existing}
    counts = read_jsonl(ROOT / 'work/enriched_pool/coverage.jsonl')
    query_pairs = [(code, query) for code in QUERIES for query in QUERIES[code]]
    query_vectors = embed([query for _, query in query_pairs])
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    scores = normed @ query_vectors.T
    additions, retrieval = {}, {}
    for count in counts:
        code, shortfall = count['code'], max(0, 80 - count['draft_positive'])
        if code not in QUERIES or shortfall == 0:
            continue
        indices = [i for i, (c, _) in enumerate(query_pairs) if c == code]
        similarity = scores[:, indices].max(axis=1)
        eligible = [i for i, r in enumerate(rows) if r['chunk_id'] not in used
                    and frozen['source_splits'][r['source_id']] == 'train']
        eligible.sort(key=lambda i: (-float(similarity[i]), rows[i]['chunk_id']))
        for i in eligible[:min(200, shortfall * 3 + 20)]:
            row = rows[i]
            additions[row['chunk_id']] = row
            retrieval.setdefault(row['chunk_id'], {})[code] = round(float(similarity[i]), 6)
    out = ROOT / 'data/selected_semantic_candidates.jsonl'
    if out.exists():
        raise ValueError('semantic retrieval output already exists')
    write_jsonl(out, existing + list(additions.values()))
    report = {'previous': len(existing), 'added': len(additions), 'total': len(existing) + len(additions),
              'retrieval_model': MODEL, 'queries': QUERIES, 'retrieval_scores': retrieval,
              'selection_sha256': sha256(out), 'budget': ledger.summary(),
              'policy': 'Added passages come only from frozen training articles. Retrieval does not assign labels.'}
    (ROOT / 'data/semantic_retrieval_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return {k: v for k, v in report.items() if k not in {'retrieval_scores', 'queries'}}


def export_embeddings(pool=PUBLISHED_POOL):
    pool = Path(pool)
    cache = ROOT / 'work/embeddings'
    manifest = json.loads((cache / 'manifest.json').read_text()) if (cache / 'manifest.json').exists() else {'ids': []}
    if manifest['ids'] and (manifest['model'] != MODEL or manifest['dimensions'] != DIMENSIONS):
        raise ValueError('base embedding settings differ')
    byid = {pid: i for i, pid in enumerate(manifest['ids'])}
    rows = read_jsonl(pool / 'passages.jsonl')
    vectors = np.load(cache / 'vectors.npy', mmap_mode='r') if byid else None
    extra_doc = json.loads((cache / 'extra_manifest.json').read_text()) if (cache / 'extra_manifest.json').exists() else {'ids': []}
    if extra_doc['ids'] and (extra_doc['model'] != MODEL or extra_doc['dimensions'] != DIMENSIONS):
        raise ValueError('extra embedding settings differ')
    extra_ids = {pid: i for i, pid in enumerate(extra_doc['ids'])}
    extra = np.load(cache / 'extra_vectors.npy', mmap_mode='r') if extra_ids else None
    selected = np.empty((len(rows), DIMENSIONS), dtype=np.float32)
    for i, row in enumerate(rows):
        pid = row['chunk_id']
        selected[i] = vectors[byid[pid]] if pid in byid else extra[extra_ids[pid]]
    out = pool / 'training/embeddings.npy'
    np.save(out, selected)
    metadata = {'model': MODEL, 'dimensions': DIMENSIONS, 'dtype': 'float32', 'rows': len(rows),
                'row_ids': [r['chunk_id'] for r in rows], 'sha256': sha256(out),
                'normalization': 'Unmodified API output; fit any standardization on training rows only.'}
    (pool / 'training/embedding_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary_path = pool / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(embeddings_generated=True, embedding_model=MODEL, embedding_dimensions=DIMENSIONS)
    summary['training_file_sha256'].update({name: sha256(pool / 'training' / name)
                                           for name in ['embeddings.npy', 'embedding_manifest.json']})
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    (pool / 'training/export.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {'rows': len(rows), 'dimensions': DIMENSIONS, 'sha256': metadata['sha256']}
