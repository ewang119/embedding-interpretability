"""Print current checkpoint coverage without accessing API credentials."""
from collections import Counter
import json

from .label_api import MODEL, PROMPT_VERSION, Ledger
from .prepare import ROOT, sha256


def status(model='gpt-5-mini-2025-08-07'):
    anchors_path = ROOT / 'data/anchors.json'
    anchors = json.loads(anchors_path.read_text())['anchors']
    ledger = Ledger(ROOT / 'work/api_labels.sqlite3')
    anchor_hash = sha256(anchors_path)
    results = [json.loads(row[0]) for row in ledger.db.execute('SELECT result FROM judgments')]
    results = [r for r in results if r['prompt_version'] == PROMPT_VERSION and r['anchor_sha256'] == anchor_hash
               and r['model'] == model]
    counts = Counter(code for r in results if r['usable'] for code in r['positive_codes'])
    report = {'completed': len(results), 'usable': sum(r['usable'] for r in results),
              'budget': ledger.summary(),
              'coverage': [{'code': a['code'], 'name': a['name'], 'draft_positive': counts[a['code']]}
                           for a in anchors]}
    (ROOT / 'work/live_coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(status(), indent=2))
