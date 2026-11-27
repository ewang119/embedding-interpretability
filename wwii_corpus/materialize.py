"""Join semantic drafts to exact passages and freeze an exploratory split."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from probe.data import SPLIT_NAMES, save_split, stratified_split
from wwii_pilot.build_shared_pool import coverage, read_jsonl
from .duplicates import near_duplicates
from .prepare import ROOT, PUBLISHED_POOL, sha256, write_jsonl


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def duplicate_removals(rows, prior):
    """Prefer existing pilot passages, otherwise the first selected passage."""
    parent = {r['chunk_id']: r['chunk_id'] for r in prior + rows}

    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    pairs = near_duplicates(prior + rows)
    for pair in pairs:
        parent[find(pair['chunk_id_b'])] = find(pair['chunk_id_a'])
    first, removed = {}, []
    old_ids = {r['chunk_id'] for r in prior}
    for row in prior + rows:
        key = find(row['chunk_id'])
        if key not in first:
            first[key] = row['chunk_id']
        elif row['chunk_id'] not in old_ids:
            removed.append({'chunk_id': row['chunk_id'], 'kept_id': first[key],
                            'reason': 'near_duplicate_component'})
    return removed, pairs


def materialize(data=ROOT / 'data', out=None, allow_partial=False, seed=673,
                selection='selected_candidates.jsonl'):
    data = Path(data)
    out = Path(out) if out else data / 'pool'
    if out.exists():
        raise ValueError('output already exists; use a new directory to preserve edits')
    doc = json.loads((data / 'anchors.json').read_text())
    anchors = doc['anchors']
    bycode = {a['code']: a['id'] for a in anchors}
    candidates = read_jsonl(data / selection)
    reviews = read_jsonl(data / 'semantic_reviews.jsonl')
    byid = {r['chunk_id']: r for r in reviews}
    if len(byid) != len(reviews):
        raise ValueError('duplicate review IDs')
    selected_ids = {r['chunk_id'] for r in candidates}
    if set(byid) - selected_ids:
        raise ValueError('reviews include passages outside the selected pool')
    missing = selected_ids - set(byid)
    if missing and not allow_partial:
        raise ValueError('labeling incomplete; resume labeling or explicitly allow a partial pool')
    rows, excluded = [], []
    anchor_hash = sha256(data / 'anchors.json')
    checks_path = data / 'spot_checks.jsonl'
    checks = read_jsonl(checks_path) if checks_path.exists() else []
    checks_byid = {r['chunk_id']: r for r in checks}
    secondary_path = data / 'crosscheck_reviews.jsonl'
    secondary = {r['chunk_id']: r for r in read_jsonl(secondary_path)} if secondary_path.exists() else {}
    frozen_path = data / 'frozen_source_splits.json'
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    base_ids = {r['chunk_id'] for r in read_jsonl(data / 'selected_candidates.jsonl')} if frozen else set()
    if len(checks_byid) != len(checks) or set(checks_byid) - selected_ids:
        raise ValueError('duplicate or unknown spot-check IDs')
    for candidate in candidates:
        review = byid.get(candidate['chunk_id'])
        if review is None:
            excluded.append({'chunk_id': candidate['chunk_id'], 'reason': 'not_yet_labeled'})
            continue
        if (review['text_sha256'] != hashlib.sha256(candidate['chunk_text'].encode()).hexdigest()
                or review['anchor_sha256'] != anchor_hash):
            raise ValueError('text or definitions differ from semantic review')
        if not set(review['positive_codes'] + review['uncertain_codes']) <= set(bycode):
            raise ValueError('unknown anchor code')
        other = secondary.get(candidate['chunk_id'])
        disagreement = []
        quality_disagreement = False
        if other:
            if other['text_sha256'] != review['text_sha256'] or other['anchor_sha256'] != anchor_hash:
                raise ValueError('stale secondary review')
            disagreement = sorted(set(review['positive_codes']) ^ set(other['positive_codes']))
            quality_disagreement = review['usable'] != other['usable']
            review = {**review, 'uncertain_codes': sorted(set(review['uncertain_codes'])
                        | set(other['uncertain_codes']) | set(disagreement))}
        check = checks_byid.get(candidate['chunk_id'])
        if check:
            if (check['text_sha256'] != review['text_sha256'] or check['anchor_sha256'] != anchor_hash
                    or not set(check['set_codes']) <= set(bycode)
                    or not set(check['uncertain_codes']) <= set(bycode)
                    or any(type(v) is not int or v not in (0, 1) for v in check['set_codes'].values())):
                raise ValueError('invalid or stale spot check')
            positive = set(review['positive_codes'])
            for code, value in check['set_codes'].items():
                positive.add(code) if value else positive.discard(code)
            review = {**review, 'positive_codes': sorted(positive),
                      'uncertain_codes': sorted(set(review['uncertain_codes']) | set(check['uncertain_codes'])),
                      'positive_evidence': [e for e in review.get('positive_evidence', []) if e['code'] in positive]}
            if check['reject_reason']:
                review.update(usable=False, quality_reason=check['reject_reason'])
        if not review['usable']:
            excluded.append({'chunk_id': candidate['chunk_id'], 'reason': review['quality_reason']})
            continue
        if frozen and candidate['chunk_id'] not in base_ids and frozen['source_splits'][candidate['source_id']] != 'train':
            excluded.append({'chunk_id': candidate['chunk_id'], 'reason': 'enrichment_from_heldout_article'})
            continue
        labels = {aid: int(code in review['positive_codes']) for code, aid in bycode.items()}
        rows.append({**candidate, 'working_labels': labels, 'draft_labels': dict(labels),
                     'labels': {aid: None for aid in bycode.values()},
                     'review_state': 'AI draft', 'label_method': 'api_semantic_review_all_anchors',
                     'label_model': review['model'], 'label_prompt_version': review['prompt_version'],
                     'positive_evidence': review.get('positive_evidence', []),
                     'secondary_review': other, 'model_disagreement_codes': disagreement,
                     'model_usability_disagreement': quality_disagreement,
                     'uncertain_anchor_ids': [bycode[c] for c in review['uncertain_codes']],
                     'review_note': 'Model-generated decisions; human approval pending.',
                     'origins': [{'origin_key': candidate['origin']['parent_id'],
                                  'origin_file': 'raw/chunks.jsonl', **candidate['origin']}]})
        if check:
            rows[-1]['spot_check'] = check
            rows[-1]['label_method'] = 'api_plus_assistant_semantic_spot_check'
        years = [int(y) for y in re.findall(r'\b(?:18|19|20)\d{2}\b', candidate['chunk_text'])]
        rows[-1]['review_flags'] = (['explicit_pre1933_dates'] if years and max(years) < 1933 else [])
    selected_ids = {r['chunk_id'] for r in rows}
    prior = [r for r in read_jsonl(ROOT.parent / 'wwii_pilot/data/passages.jsonl') if r['chunk_id'] not in selected_ids]
    removed, pairs = duplicate_removals(rows, prior)
    excluded.extend(removed)
    remove_ids = {r['chunk_id'] for r in removed}
    rows = [r for r in rows if r['chunk_id'] not in remove_ids]
    if frozen and set(frozen.get('base_retained_ids', [])) - {r['chunk_id'] for r in rows}:
        raise ValueError('enrichment changed frozen base membership; resolve duplicate components explicitly')
    if not rows:
        raise ValueError('no usable labeled passages')
    aids = [a['id'] for a in anchors]
    Y = np.array([[row['working_labels'][aid] for aid in aids] for row in rows], dtype=np.float32)
    if frozen:
        assignments = [frozen['source_splits'][r['source_id']] for r in rows]
        if set(assignments) - set(SPLIT_NAMES):
            raise ValueError('invalid frozen source assignment')
        parts = [np.flatnonzero(np.asarray(assignments) == name) for name in SPLIT_NAMES]
    else:
        parts = stratified_split(Y, (.7, .1, .2), seed, groups=[r['source_id'] for r in rows])
    if any(len(part) == 0 for part in parts):
        raise ValueError('source grouping produced an empty split')
    split_for = {int(i): name for name, part in zip(SPLIT_NAMES, parts) for i in part}
    for i, row in enumerate(rows):
        row['split'] = split_for[i]
    # Verify both supplied IDs and normalized article URLs remain whole.
    for key in ('source_id', 'source_page_url'):
        groups = {}
        for row in rows:
            groups.setdefault(row[key], set()).add(row['split'])
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError(f'{key} crosses splits')
    out.mkdir(parents=True)
    for row in rows:
        if row.get('source_status') == 'verified_exact_snapshot':
            import shutil
            relative = Path(row['source_snapshot'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid supplemental snapshot path')
            (out / relative.parent).mkdir(exist_ok=True)
            shutil.copyfile(data / relative, out / relative)
    (out / 'anchors.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    write_jsonl(out / 'passages.jsonl', rows)
    write_jsonl(out / 'excluded.jsonl', excluded)
    write_jsonl(out / 'near_duplicates.jsonl', pairs)
    write_jsonl(out / 'spot_checks.jsonl', checks)
    fields = ['passage_id', 'passage_text', 'review_state', 'source_id', 'source_url', 'source_verified', *aids]
    write_csv(out / 'labeling.csv', fields,
              ({'passage_id': r['chunk_id'], 'passage_text': r['chunk_text'], 'review_state': r['review_state'],
                'source_id': r['source_id'], 'source_url': r['source_url'], 'source_verified': 'yes' if r.get('source_verified') else 'no',
                **r['working_labels']} for r in rows))
    counts = coverage(rows, anchors)
    for j, count in enumerate(counts):
        count.update({name: int(Y[part, j].sum()) for name, part in zip(SPLIT_NAMES, parts)})
        count['test_positive_sources'] = len({rows[int(i)]['source_id'] for i in parts[2] if Y[i, j]})
    write_csv(out / 'coverage.csv', list(counts[0]), counts)
    write_jsonl(out / 'coverage.jsonl', counts)
    queue = []
    for row in rows:
        if row['model_usability_disagreement']:
            queue.append({'chunk_id': row['chunk_id'], 'anchor_id': None, 'reason': 'model_usability_disagreement',
                          'draft_label': None, 'split': row['split']})
        for code in row.get('spot_check', {}).get('set_codes', {}):
            aid = bycode[code]
            queue.append({'chunk_id': row['chunk_id'], 'anchor_id': aid, 'reason': 'assistant_spot_check',
                          'draft_label': row['working_labels'][aid], 'split': row['split']})
        for aid in row['uncertain_anchor_ids']:
            queue.append({'chunk_id': row['chunk_id'], 'anchor_id': aid, 'reason': 'model_uncertain',
                          'draft_label': row['working_labels'][aid], 'split': row['split']})
        for code in row['model_disagreement_codes']:
            aid = bycode[code]
            queue.append({'chunk_id': row['chunk_id'], 'anchor_id': aid, 'reason': 'model_disagreement',
                          'draft_label': row['working_labels'][aid], 'split': row['split']})
        if row['review_flags']:
            queue.extend({'chunk_id': row['chunk_id'], 'anchor_id': aid, 'reason': 'temporal_scope_check',
                          'draft_label': 1, 'split': row['split']}
                         for aid, value in row['working_labels'].items() if value)
    # A small deterministic calibration set supplements model-declared uncertainty.
    for aid in aids:
        for value in (0, 1):
            eligible = [r for r in rows if r['working_labels'][aid] == value]
            eligible.sort(key=lambda r: hashlib.sha256(f'{seed}:{aid}:{r["chunk_id"]}'.encode()).hexdigest())
            queue.extend({'chunk_id': r['chunk_id'], 'anchor_id': aid, 'reason': 'calibration_sample',
                          'draft_label': value, 'split': r['split']} for r in eligible[:5])
        code = next(a['code'] for a in anchors if a['id'] == aid)
        difficult_negatives = [r for r in rows if r['working_labels'][aid] == 0
                               and code in r.get('keyword_hints', {}).get('positive', [])
                               + r.get('keyword_hints', {}).get('near_miss', [])]
        difficult_negatives.sort(key=lambda r: hashlib.sha256(f'hard:{seed}:{aid}:{r["chunk_id"]}'.encode()).hexdigest())
        queue.extend({'chunk_id': r['chunk_id'], 'anchor_id': aid, 'reason': 'keyword_candidate_rejected_by_model',
                      'draft_label': 0, 'split': r['split']} for r in difficult_negatives[:5])
    unique_queue = {}
    for entry in queue:
        key = entry['chunk_id'], entry['anchor_id']
        if key not in unique_queue:
            unique_queue[key] = {**entry, 'reasons': [entry['reason']]}
        elif entry['reason'] not in unique_queue[key]['reasons']:
            unique_queue[key]['reasons'].append(entry['reason'])
    write_jsonl(out / 'review_queue.jsonl', unique_queue.values())
    row_byid = {r['chunk_id']: r for r in rows}
    anchor_byid = {a['id']: a for a in anchors}
    readable_queue = []
    for entry in unique_queue.values():
        row = row_byid[entry['chunk_id']]
        anchor = anchor_byid.get(entry['anchor_id'])
        code = anchor['code'] if anchor else 'USABILITY'
        other = row['secondary_review']
        secondary_label = (int(code in other['positive_codes']) if anchor else int(other['usable'])) if other else ''
        readable_queue.append({'chunk_id': row['chunk_id'], 'anchor_code': code,
            'anchor_name': anchor['name'] if anchor else 'Passage usability', 'chunk_text': row['chunk_text'],
            'source_url': row['source_url'], 'split': row['split'],
            'primary_label': entry['draft_label'] if anchor else 1, 'secondary_label': secondary_label,
            'reasons': '; '.join(entry['reasons']), 'reviewed_label': '', 'notes': ''})
    if readable_queue:
        write_csv(out / 'review_queue.csv', list(readable_queue[0]), readable_queue)
    training = out / 'training'
    training.mkdir()
    ids = [r['chunk_id'] for r in rows]
    write_csv(training / 'texts.csv', ['id', 'text'],
              ({'id': r['chunk_id'], 'text': r['chunk_text']} for r in rows))
    write_csv(training / 'labels.csv', ['id', *aids],
              ({'id': r['chunk_id'], **r['working_labels']} for r in rows))
    write_csv(training / 'meta.csv', ['id', 'source_id', 'split', 'source_url'],
              ({'id': r['chunk_id'], 'source_id': r['source_id'], 'split': r['split'],
                'source_url': r['source_url']} for r in rows))
    save_split(training / 'split.json', parts, ids)
    summary = {'selected': len(candidates), 'semantically_labeled': len(reviews), 'retained': len(rows),
               'unlabeled': len(missing), 'anchors': len(aids), 'draft_decisions': len(rows) * len(aids),
               'sources': len({r['source_id'] for r in rows}),
               'excluded_by_reason': dict(Counter(r['reason'] for r in excluded)),
               'split_sizes': {name: len(part) for name, part in zip(SPLIT_NAMES, parts)},
               'requested_split': [.7, .1, .2], 'seed': seed, 'source_overlap_between_splits': 0,
               'split_policy': 'frozen base articles; training-only enrichment' if frozen else 'source-group stratification',
               'minimum_draft_positives': min(c['draft_positive'] for c in counts),
               'anchors_below_75_positives': [c['code'] for c in counts if c['draft_positive'] < 75],
               'anchors_below_5_test_positives': [c['code'] for c in counts if c['test'] < 5],
               'anchors_with_fewer_than_2_positive_test_sources': [c['code'] for c in counts if c['test_positive_sources'] < 2],
               'human_approved_passages': 0, 'source_verified_passages': sum(bool(r.get('source_verified')) for r in rows),
               'provisional_definitions': sum(a['definition_status'] == 'provisional_inferred' for a in anchors),
               'label_models': dict(Counter(r['label_model'] for r in rows)),
               'assistant_spot_checks': len(checks),
               'queued_review_items': len(unique_queue),
               'crosschecked_passages': sum(r['secondary_review'] is not None for r in rows),
               'passages_with_model_disagreements': sum(bool(r['model_disagreement_codes']) for r in rows),
               'model_disagreement_decisions': sum(len(r['model_disagreement_codes']) for r in rows),
               'input_sha256': {name: sha256(data / name) for name in
                                [selection, 'semantic_reviews.jsonl', 'anchors.json']},
               'training_file_sha256': {p.name: sha256(p) for p in training.iterdir()},
               'exploratory_only': True, 'embeddings_generated': False,
               'limitation': 'AI drafts are not evaluation gold. Review labels and source text, finalize definitions, '
                             'and freeze the evaluation protocol before measuring accuracy.'}
    for name in ['crosscheck_reviews.jsonl', 'spot_checks.jsonl', 'frozen_source_splits.json',
                 'supplemental_candidates.jsonl', 'supplemental_raw.jsonl', 'label_protocol.json']:
        if (data / name).exists():
            summary['input_sha256'][name] = sha256(data / name)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (training / 'export.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=PUBLISHED_POOL)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--seed', type=int, default=673)
    parser.add_argument('--selection', default='selected_candidates.jsonl')
    args = parser.parse_args()
    print(json.dumps(materialize(args.data, args.out, args.allow_partial, args.seed, args.selection), indent=2))
