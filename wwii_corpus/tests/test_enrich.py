import json

from wwii_corpus.enrich import enrich
from wwii_corpus.prepare import write_jsonl
from wwii_pilot.build_shared_pool import read_jsonl


def test_enrichment_retrieves_underrepresented_concepts_without_labeling(tmp_path):
    pool, data = tmp_path / 'pool', tmp_path / 'data'
    pool.mkdir()
    data.mkdir()
    write_jsonl(pool / 'coverage.jsonl', [{'code': 'A01', 'draft_positive': 74},
                                         {'code': 'A02', 'draft_positive': 100}])
    original = {'chunk_id': 'old', 'source_id': 'old', 'keyword_hints': {'positive': [], 'near_miss': []}}
    remaining = [{'chunk_id': f'new{i}', 'source_id': f's{i}',
                  'keyword_hints': {'positive': ['A01'], 'near_miss': []}} for i in range(3)]
    unrelated = {'chunk_id': 'unrelated', 'source_id': 'u', 'keyword_hints': {'positive': ['A02'], 'near_miss': []}}
    write_jsonl(data / 'selected_candidates.jsonl', [original])
    write_jsonl(data / 'cleaned_candidates.jsonl', [original, *remaining, unrelated])
    report = enrich(pool, data, maximum=2)
    rows = read_jsonl(data / 'selected_enriched_candidates.jsonl')
    assert report['added'] == 2
    assert rows[0] == original
    assert 'unrelated' not in {r['chunk_id'] for r in rows}
    assert all('labels' not in r and 'working_labels' not in r for r in rows)
