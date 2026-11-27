"""Verify the committed working snapshot without private build files or API calls."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .prepare import PUBLISHED_POOL, canonical_article, sha256
from wwii_pilot.build_shared_pool import passage_id, read_jsonl
from wwii_pilot.refresh_shared_pool import validate_snapshot


def archived_hash(path):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def verify(pool=PUBLISHED_POOL):
    pool = Path(pool).resolve()
    errors = []
    manifest = json.loads((pool / 'bundle_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (pool / name).resolve()
        if pool not in path.parents or not path.is_file():
            errors.append(f'{name}: missing or invalid path')
        elif path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            errors.append(f'{name}: checksum differs')
    raw = pool / 'provenance/raw'
    inputs = json.loads((raw / 'manifest.json').read_text())
    for name, expected in inputs.items():
        actual_hash, size = archived_hash(raw / (name + '.gz'))
        if actual_hash != expected['sha256'] or size != expected['bytes']:
            errors.append(f'{name}: original input archive differs')
    summary = json.loads((pool / 'summary.json').read_text())
    for name, expected in summary['input_sha256'].items():
        path = pool / name if name == 'anchors.json' else pool / 'provenance' / name
        actual = sha256(path) if path.exists() else archived_hash(Path(str(path) + '.gz'))[0]
        if actual != expected:
            errors.append(f'{name}: build input differs')
    for name, expected in summary['training_file_sha256'].items():
        if sha256(pool / 'training' / name) != expected:
            errors.append(f'{name}: training export differs')
    rows = read_jsonl(pool / 'passages.jsonl')
    document = json.loads((pool / 'anchors.json').read_text())
    anchors = document['anchors']
    aids = [a['id'] for a in anchors]
    ids = [r['chunk_id'] for r in rows]
    if len(set(ids)) != len(ids) or len(rows) != summary['retained']:
        errors.append('duplicate IDs or incorrect passage count')
    frozen = json.loads((pool / 'provenance/frozen_source_splits.json').read_text())
    sources, articles = {}, {}
    for row in rows:
        pid = row['chunk_id']
        if pid != passage_id(row['chunk_text']):
            errors.append(f'{pid}: passage text/ID differs')
        if set(row['working_labels']) != set(aids) or any(type(v) is not int or v not in (0, 1) for v in row['working_labels'].values()):
            errors.append(f'{pid}: invalid working labels')
        if set(row['labels']) != set(aids):
            errors.append(f'{pid}: approved-label keys differ')
        if row['review_state'] == 'Reviewed':
            if row['labels'] != row['working_labels']:
                errors.append(f'{pid}: reviewed labels disagree')
        elif any(v is not None for v in row['labels'].values()):
            errors.append(f'{pid}: unapproved labels promoted')
        if row['anchor_version'] != document['anchor_version']:
            errors.append(f'{pid}: anchor version differs')
        if row['split'] not in {'train', 'val', 'test'} or row['split'] != frozen['source_splits'].get(row['source_id']):
            errors.append(f'{pid}: frozen source assignment differs')
        sources.setdefault(row['source_id'], set()).add(row['split'])
        articles.setdefault(canonical_article(row['source_url']), set()).add(row['split'])
        if row.get('source_status') == 'verified_exact_snapshot':
            validate_snapshot(row, pool)
    if any(len(splits) != 1 for splits in [*sources.values(), *articles.values()]):
        errors.append('a source crosses splits')
    if set(frozen['base_retained_ids']) - set(ids):
        errors.append('frozen base passages missing')
    for name in ['texts.csv', 'labels.csv', 'meta.csv']:
        with (pool / 'training' / name).open(newline='') as stream:
            exported = list(csv.DictReader(stream))
        if [e['id'] for e in exported] != ids:
            errors.append(f'{name}: row IDs/order differ')
            continue
        for row, entry in zip(rows, exported):
            if name == 'texts.csv' and entry['text'] != row['chunk_text']:
                errors.append(f'{name}: text differs')
            if name == 'labels.csv' and any(entry[aid] != str(row['working_labels'][aid]) for aid in aids):
                errors.append(f'{name}: labels differ')
            if name == 'meta.csv' and any(entry[key] != row[key] for key in ['source_id', 'source_url', 'split']):
                errors.append(f'{name}: metadata differs')
    metadata = json.loads((pool / 'training/embedding_manifest.json').read_text())
    vectors = np.load(pool / 'training/embeddings.npy', mmap_mode='r')
    if metadata['row_ids'] != ids or vectors.shape != (len(rows), metadata['dimensions']) or not np.isfinite(vectors).all():
        errors.append('embedding shape, values, or row IDs differ')
    if sha256(pool / 'training/embeddings.npy') != metadata['sha256']:
        errors.append('embedding checksum differs')
    return {'passages': len(rows), 'anchors': len(aids), 'sources': len(sources),
            'split_sizes': dict(Counter(r['split'] for r in rows)), 'errors': errors,
            'integrity_passed': not errors, 'human_approved_passages': sum(r['review_state'] == 'Reviewed' for r in rows),
            'limitation': 'Integrity verification does not establish semantic label accuracy or historical truth.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=PUBLISHED_POOL)
    args = parser.parse_args()
    result = verify(args.pool)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
