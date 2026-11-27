import json

import pytest

from wwii_corpus.prepare import write_jsonl
from wwii_corpus.select import select


def test_selection_is_reproducible_balances_sources_and_ignores_hints(tmp_path):
    candidates = tmp_path / 'candidates.jsonl'
    rows = [{'chunk_id': f'{s}{i}', 'source_id': s, 'keyword_hints': {}}
            for s in 'ABC' for i in range(5)]
    write_jsonl(candidates, rows)
    one, two = tmp_path / 'one.jsonl', tmp_path / 'two.jsonl'
    report = select(candidates, one, target=6)
    for row in rows:
        row['keyword_hints'] = {'positive': ['A01']}
    write_jsonl(candidates, rows)
    select(candidates, two, target=6)
    ids = lambda path: [json.loads(line)['chunk_id'] for line in path.read_text().splitlines()]
    assert ids(one) == ids(two)
    assert report['minimum_per_source'] == report['maximum_per_source'] == 2
    with pytest.raises(ValueError, match='already exists'):
        select(candidates, one)


def test_selection_rejects_empty_or_duplicate_input(tmp_path):
    candidates, out = tmp_path / 'input.jsonl', tmp_path / 'out.jsonl'
    for rows in [[], [{'chunk_id': 'x', 'source_id': 's'}] * 2]:
        write_jsonl(candidates, rows)
        with pytest.raises(ValueError, match='nonempty'):
            select(candidates, out)
