"""Retrieve extra candidates for low-support anchors; never assign labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .prepare import ROOT, sha256, write_jsonl
from wwii_pilot.build_shared_pool import read_jsonl


def enrich(pool, data=ROOT / 'data', target=75, maximum=600, seed=673):
    if target < 1 or maximum < 1:
        raise ValueError('target and maximum must be positive')
    pool, data = Path(pool), Path(data)
    out = data / 'selected_enriched_candidates.jsonl'
    if out.exists():
        raise ValueError('enriched selection already exists')
    counts = read_jsonl(pool / 'coverage.jsonl')
    short = {r['code']: target - r['draft_positive'] for r in counts if r['draft_positive'] < target}
    original = read_jsonl(data / 'selected_candidates.jsonl')
    used = {r['chunk_id'] for r in original}
    remaining = [r for r in read_jsonl(data / 'cleaned_candidates.jsonl') if r['chunk_id'] not in used]
    groups = {}
    rank = lambda row: hashlib.sha256(f'{seed}:enrich:{row["chunk_id"]}'.encode()).hexdigest()
    for code in short:
        bysource = defaultdict(list)
        for row in remaining:
            hints = row['keyword_hints']
            if code in hints['positive'] or code in hints['near_miss']:
                bysource[row['source_id']].append(row)
        for source in bysource:
            bysource[source].sort(key=lambda r: (code not in r['keyword_hints']['positive'], rank(r)))
        # Round-robin articles within each deficient anchor's retrieval pool.
        ordered = []
        for depth in range(max(map(len, bysource.values()), default=0)):
            for source in sorted(bysource):
                if depth < len(bysource[source]):
                    ordered.append(bysource[source][depth])
        groups[code] = ordered[:max(30, short[code] * 4)]
    selected = {}
    for depth in range(max(map(len, groups.values()), default=0)):
        for code in sorted(groups, key=lambda c: (-short[c], c)):
            if depth < len(groups[code]) and len(selected) < maximum:
                row = groups[code][depth]
                selected.setdefault(row['chunk_id'], {'row': row, 'retrieval_anchors': []})['retrieval_anchors'].append(code)
    additions = [entry['row'] for entry in selected.values()]
    write_jsonl(out, original + additions)
    report = {'original': len(original), 'added': len(additions), 'total': len(original) + len(additions),
              'target_draft_positives': target, 'shortfalls_before': short,
              'selection_sha256': sha256(out),
              'added_ids': {pid: sorted(set(entry['retrieval_anchors'])) for pid, entry in selected.items()},
              'limitation': 'Keyword hints guide retrieval only. Every added passage still needs semantic labeling '
                            'against all anchors. Enrichment changes label prevalence and does not guarantee coverage.'}
    (data / 'enrichment_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--target', type=int, default=75)
    parser.add_argument('--maximum', type=int, default=600)
    args = parser.parse_args()
    result = enrich(args.pool, args.data, args.target, args.maximum)
    print(json.dumps({k: v for k, v in result.items() if k != 'added_ids'}, indent=2))
