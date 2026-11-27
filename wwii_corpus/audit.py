"""Read-only checks for exact spans, draft labels, exports, and split leakage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from wwii_pilot.build_shared_pool import fingerprint, passage_id, read_jsonl
from .prepare import ROOT, PUBLISHED_POOL, canonical_article, load_language_tools, sha256


def audit(data=ROOT / 'data', pool=None):
    data = Path(data)
    errors = []
    manifest = json.loads((data / 'raw/manifest.json').read_text())
    for name, entry in manifest.items():
        if sha256(data / 'raw' / name) != entry['sha256']:
            errors.append(f'raw/{name}: hash mismatch')
    parents = {r['id']: r for r in read_jsonl(data / 'raw/chunks.jsonl')}
    rows = read_jsonl(data / 'cleaned_candidates.jsonl')
    original_count = len(rows)
    supplemental = read_jsonl(data / 'supplemental_candidates.jsonl') if (data / 'supplemental_candidates.jsonl').exists() else []
    if supplemental:
        rows.extend(supplemental)
        parents.update({r['id']: r for r in read_jsonl(data / 'supplemental_raw.jsonl')})
    prior = read_jsonl(ROOT.parent / 'wwii_pilot/data/passages.jsonl')
    imported_ids = {r['chunk_id'] for r in supplemental}
    prior = [r for r in prior if r['chunk_id'] not in imported_ids]
    tokenizer, _ = load_language_tools()
    seen_sentences = {fingerprint(s) for r in prior for s in tokenizer.tokenize(r['chunk_text'])}
    seen_ids, source_urls, source_ids = set(), {}, {}
    for row in rows:
        pid, origin = row['chunk_id'], row['origin']
        parent = parents[origin['parent_id']]
        if row.get('source_status') == 'verified_exact_snapshot':
            from wwii_pilot.refresh_shared_pool import validate_snapshot
            validate_snapshot(row, data)
        spans = list(tokenizer.span_tokenize(parent['text']))
        ss = [parent['text'][a:b] for a, b in spans[origin['sentence_start']:origin['sentence_end']]]
        if row['chunk_text'] != parent['text'][origin['char_start']:origin['char_end']] or ss != row['source_sentences']:
            errors.append(f'{pid}: not an exact source span')
        if len(ss) not in (2, 3) or not 25 <= len(row['chunk_text'].split()) <= 70:
            errors.append(f'{pid}: invalid passage length')
        if pid in seen_ids or pid != passage_id(row['chunk_text']):
            errors.append(f'{pid}: duplicate ID or mismatched text')
        seen_ids.add(pid)
        for sentence in ss:
            key = fingerprint(sentence)
            if key in seen_sentences:
                errors.append(f'{pid}: repeated source sentence')
            seen_sentences.add(key)
        canonical = canonical_article(row['source_url'])
        source_urls.setdefault(row['source_id'], set()).add(canonical)
        source_ids.setdefault(canonical, set()).add(row['source_id'])
        if row['review_state'] != 'Unreviewed' or 'labels' in row or 'working_labels' in row:
            errors.append(f'{pid}: candidate has premature labels')
    if any(len(values) != 1 for values in [*source_urls.values(), *source_ids.values()]):
        errors.append('source ID/article identity is not one-to-one')
    pool = Path(pool) if pool is not None else (PUBLISHED_POOL if data.resolve() == (ROOT / 'data').resolve() else data / 'pool')
    pooled = None
    if pool.exists():
        pooled = read_jsonl(pool / 'passages.jsonl')
        anchors = json.loads((pool / 'anchors.json').read_text())['anchors']
        aids = {a['id'] for a in anchors}
        candidate_byid = {r['chunk_id']: r for r in rows}
        sources = {}
        frozen_path = data / 'frozen_source_splits.json'
        frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
        for row in pooled:
            pid = row['chunk_id']
            if pid not in candidate_byid or row['chunk_text'] != candidate_byid[pid]['chunk_text']:
                errors.append(f'{pid}: pool text differs from candidates')
            if set(row['working_labels']) != aids or any(type(v) is not int or v not in (0, 1) for v in row['working_labels'].values()):
                errors.append(f'{pid}: incomplete or invalid draft labels')
            if row['review_state'] != 'AI draft' or set(row['labels'].values()) != {None}:
                errors.append(f'{pid}: unapproved labels promoted')
            if frozen and row['split'] != frozen['source_splits'][row['source_id']]:
                errors.append(f'{pid}: frozen source assignment differs')
            if row.get('source_status') == 'verified_exact_snapshot':
                from wwii_pilot.refresh_shared_pool import validate_snapshot
                validate_snapshot(row, pool)
            sources.setdefault(row['source_page_url'], set()).add(row['split'])
        if any(len(splits) != 1 for splits in sources.values()):
            errors.append('source article crosses splits')
        if frozen and set(frozen.get('base_retained_ids', [])) - {r['chunk_id'] for r in pooled}:
            errors.append('frozen base membership changed')
        for filename, key in [('texts.csv', 'text'), ('labels.csv', None), ('meta.csv', 'source_id')]:
            with (pool / 'training' / filename).open(newline='') as stream:
                exported = list(csv.DictReader(stream))
            if [e['id'] for e in exported] != [r['chunk_id'] for r in pooled]:
                errors.append(f'{filename}: row order/IDs differ')
                continue
            for original, exported_row in zip(pooled, exported):
                if key and exported_row[key] != original['chunk_text' if key == 'text' else key]:
                    errors.append(f'{filename}: data differs')
                if key is None and any(exported_row[a] != str(original['working_labels'][a]) for a in aids):
                    errors.append(f'{filename}: labels differ')
        embedding_manifest = pool / 'training/embedding_manifest.json'
        if embedding_manifest.exists():
            metadata = json.loads(embedding_manifest.read_text())
            vectors = np.load(pool / 'training/embeddings.npy', mmap_mode='r')
            if metadata['row_ids'] != [r['chunk_id'] for r in pooled]:
                errors.append('embedding row IDs differ')
            if vectors.shape != (len(pooled), metadata['dimensions']) or not np.isfinite(vectors).all():
                errors.append('invalid embedding matrix')
            if sha256(pool / 'training/embeddings.npy') != metadata['sha256']:
                errors.append('embedding hash differs')
    return {'cleaned_candidates_checked': original_count, 'supplemental_candidates_checked': len(supplemental),
            'pooled_passages_checked': None if pooled is None else len(pooled),
            'errors': errors, 'integrity_passed': not errors,
            'limitation': 'Integrity checks do not establish semantic label accuracy, historical accuracy, or human approval.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--pool', type=Path)
    args = parser.parse_args()
    result = audit(args.data, args.pool)
    (args.data / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
