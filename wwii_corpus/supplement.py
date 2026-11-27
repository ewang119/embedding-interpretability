"""Reuse eligible pilot text without copying its labels or changing the pilot."""
import json
from pathlib import Path
import shutil

from .prepare import ROOT, canonical_article, load_language_tools, sha256, source_id, write_jsonl
from wwii_pilot.build_shared_pool import fingerprint, read_jsonl
from wwii_pilot.refresh_shared_pool import validate_snapshot


def supplement():
    data, pilot = ROOT / 'data', ROOT.parent / 'wwii_pilot/data'
    out = data / 'selected_final_candidates.jsonl'
    if out.exists():
        raise ValueError('final selection already exists')
    current = read_jsonl(data / 'selected_semantic_candidates.jsonl')
    byurl = {r['source_page_url']: r['source_id'] for r in current}
    frozen_path = data / 'frozen_source_splits.json'
    frozen = json.loads(frozen_path.read_text())
    tokenizer, _ = load_language_tools()
    additions, raw, skipped = [], [], []
    seen_sentences = {fingerprint(s) for r in current for s in r['source_sentences']}
    targets = {'oil_fuel_constraints': 'A25', 'churchill_rhetoric_public_morale': 'A27'}
    input_hash = sha256(pilot / 'passages.jsonl')
    for row in read_jsonl(pilot / 'passages.jsonl'):
        codes = [code for aid, code in targets.items() if row['working_labels'].get(aid) == 1]
        if not codes or not row.get('source_url'):
            continue
        url = canonical_article(row['source_url'])
        sid = byurl.get(url, source_id(url))
        sentences = row.get('source_sentences') or tokenizer.tokenize(row['chunk_text'])
        reason = ('heldout_article' if frozen['source_splits'].get(sid, 'train') != 'train' else
                  'length' if len(sentences) not in (2, 3) or not 25 <= len(row['chunk_text'].split()) <= 70 else
                  'repeated_sentence' if any(fingerprint(s) in seen_sentences for s in sentences) else None)
        if reason:
            skipped.append({'chunk_id': row['chunk_id'], 'reason': reason})
            continue
        verified = row.get('source_status') == 'verified_exact_snapshot'
        snapshot = {}
        if verified:
            validate_snapshot(row, pilot)
            source_file = pilot / row['source_snapshot']
            relative = Path('supplemental_snapshots') / source_file.name
            (data / relative.parent).mkdir(exist_ok=True)
            shutil.copyfile(source_file, data / relative)
            snapshot = {key: row[key] for key in ['source_html_sha256', 'paragraph_index', 'char_start', 'char_end']}
            snapshot['source_snapshot'] = str(relative)
        pid = 'pilot:' + row['chunk_id']
        origin = {'parent_id': pid, 'char_start': 0, 'char_end': len(row['chunk_text']),
                  'sentence_start': 0, 'sentence_end': len(sentences)}
        additions.append({'chunk_id': row['chunk_id'], 'chunk_text': row['chunk_text'],
            'word_count': len(row['chunk_text'].split()), 'source_id': sid, 'source_url': row['source_url'],
            'source_page_url': url, 'source_title': row.get('source_title') or '', 'source_section': '',
            'source_status': 'verified_exact_snapshot' if verified else 'provided_pilot_text',
            'source_verified': verified, 'source_license': row.get('source_license') or 'not confirmed',
            'source_sentences': sentences, 'origin': origin, 'raw_file_sha256': input_hash,
            'anchor_version': row['anchor_version'], 'keyword_hints': {'positive': [], 'near_miss': []},
            'review_state': 'Unreviewed', 'split': None, 'supplement_retrieval_codes': codes, **snapshot})
        raw.append({'id': pid, 'text': row['chunk_text'], 'original_id': row['chunk_id'], 'input_sha256': input_hash})
        frozen['source_splits'][sid] = 'train'
        seen_sentences.update(fingerprint(s) for s in sentences)
    write_jsonl(data / 'supplemental_candidates.jsonl', additions)
    write_jsonl(data / 'supplemental_raw.jsonl', raw)
    write_jsonl(out, current + additions)
    frozen_path.write_text(json.dumps(frozen, indent=2) + '\n')
    report = {'added': len(additions), 'verified_snapshot_passages': sum(r['source_verified'] for r in additions),
              'total': len(current) + len(additions), 'skipped': skipped, 'pilot_input_sha256': input_hash,
              'selection_sha256': sha256(out),
              'policy': 'Text reused as a training-only supplement; old labels are retrieval hints only and are re-evaluated against all 40 anchors.'}
    (data / 'supplement_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(supplement(), indent=2))
