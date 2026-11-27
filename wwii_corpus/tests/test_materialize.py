import hashlib
import json

import pytest

from wwii_corpus import materialize as materialization
from wwii_corpus.prepare import sha256, write_jsonl


def fixture_pool(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    anchors = {'anchor_version': 'fixture', 'anchors': [
        {'code': 'A01', 'id': 'one', 'name': 'One', 'definition_status': 'user_supplied'},
        {'code': 'A02', 'id': 'two', 'name': 'Two', 'definition_status': 'provisional_inferred'}]}
    (data / 'anchors.json').write_text(json.dumps(anchors))
    rows, reviews = [], []
    for i in range(12):
        text = f'Fixture historical passage number {i}.'
        rows.append({'chunk_id': str(i), 'chunk_text': text, 'source_id': f's{i//2}',
                     'source_url': f'https://example.org/{i//2}', 'source_page_url': f'https://example.org/{i//2}',
                     'origin': {'parent_id': str(i)}, 'anchor_version': 'fixture'})
        reviews.append({'chunk_id': str(i), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'anchor_sha256': sha256(data / 'anchors.json'), 'positive_codes': ['A01'],
                        'uncertain_codes': ['A02'] if i == 0 else [], 'usable': i != 11,
                        'quality_reason': 'context_missing' if i == 11 else 'ok',
                        'model': 'fixture-model', 'prompt_version': 1})
    write_jsonl(data / 'selected_candidates.jsonl', rows)
    write_jsonl(data / 'semantic_reviews.jsonl', reviews)
    return data, reviews


def test_materialized_labels_remain_drafts_and_articles_stay_whole(tmp_path, monkeypatch):
    data, _ = fixture_pool(tmp_path)
    monkeypatch.setattr(materialization, 'duplicate_removals', lambda rows, prior: ([], []))
    result = materialization.materialize(data)
    rows = materialization.read_jsonl(data / 'pool/passages.jsonl')
    assert result['retained'] == 11
    assert result['draft_decisions'] == 22
    assert result['exploratory_only'] is True
    assert all(set(r['labels'].values()) == {None} for r in rows)
    assert all(r['working_labels'] == {'one': 1, 'two': 0} for r in rows)
    sources = {}
    for row in rows:
        sources.setdefault(row['source_id'], set()).add(row['split'])
    assert all(len(splits) == 1 for splits in sources.values())
    with pytest.raises(ValueError, match='already exists'):
        materialization.materialize(data)


def test_incomplete_and_stale_reviews_are_rejected(tmp_path):
    data, reviews = fixture_pool(tmp_path)
    write_jsonl(data / 'semantic_reviews.jsonl', reviews[:-1])
    with pytest.raises(ValueError, match='incomplete'):
        materialization.materialize(data)
    reviews[0]['text_sha256'] = 'stale'
    write_jsonl(data / 'semantic_reviews.jsonl', reviews)
    with pytest.raises(ValueError, match='differ'):
        materialization.materialize(data)


def test_near_duplicate_components_prefer_the_existing_pilot():
    original = 'The factories produced aircraft throughout the war. Skilled workers assembled engines and wings.'
    prior = [{'chunk_id': 'old', 'chunk_text': original}]
    rows = [{'chunk_id': 'new', 'chunk_text': original.replace('Skilled', 'Experienced')},
            {'chunk_id': 'other', 'chunk_text': 'Entirely different account of wartime diplomacy and peace negotiations.'}]
    removed, pairs = materialization.duplicate_removals(rows, prior)
    assert removed == [{'chunk_id': 'new', 'kept_id': 'old', 'reason': 'near_duplicate_component'}]
    assert len(pairs) == 1


def test_spot_check_corrects_only_the_draft_and_preserves_raw_api_review(tmp_path, monkeypatch):
    data, reviews = fixture_pool(tmp_path)
    raw_before = (data / 'semantic_reviews.jsonl').read_bytes()
    check = {'chunk_id': '0', 'text_sha256': reviews[0]['text_sha256'],
             'anchor_sha256': sha256(data / 'anchors.json'), 'set_codes': {'A01': 0},
             'uncertain_codes': ['A02'], 'reject_reason': None,
             'reason': 'Fixture correction', 'human_approved': False}
    write_jsonl(data / 'spot_checks.jsonl', [check])
    monkeypatch.setattr(materialization, 'duplicate_removals', lambda rows, prior: ([], []))
    result = materialization.materialize(data)
    first = materialization.read_jsonl(data / 'pool/passages.jsonl')[0]
    assert first['working_labels']['one'] == 0
    assert set(first['labels'].values()) == {None}
    assert first['review_state'] == 'AI draft'
    assert result['assistant_spot_checks'] == 1
    assert (data / 'semantic_reviews.jsonl').read_bytes() == raw_before


def test_secondary_disagreement_is_queued_without_replacing_primary_labels(tmp_path, monkeypatch):
    data, reviews = fixture_pool(tmp_path)
    other = {**reviews[0], 'positive_codes': ['A02'], 'usable': False, 'quality_reason': 'fragment'}
    write_jsonl(data / 'crosscheck_reviews.jsonl', [other])
    monkeypatch.setattr(materialization, 'duplicate_removals', lambda rows, prior: ([], []))
    result = materialization.materialize(data)
    first = materialization.read_jsonl(data / 'pool/passages.jsonl')[0]
    assert first['working_labels'] == {'one': 1, 'two': 0}
    assert set(first['uncertain_anchor_ids']) == {'one', 'two'}
    assert first['model_usability_disagreement'] is True
    assert result['model_disagreement_decisions'] == 2
    queue = materialization.read_jsonl(data / 'pool/review_queue.jsonl')
    assert any(r['chunk_id'] == '0' and r['reason'] == 'model_usability_disagreement' for r in queue)


def test_frozen_split_excludes_enrichment_from_heldout_articles(tmp_path, monkeypatch):
    data, reviews = fixture_pool(tmp_path)
    candidates = materialization.read_jsonl(data / 'selected_candidates.jsonl')
    extra = {**candidates[8], 'chunk_id': 'extra', 'chunk_text': 'A new heldout-article passage.'}
    review = {**reviews[8], 'chunk_id': 'extra',
              'text_sha256': hashlib.sha256(extra['chunk_text'].encode()).hexdigest()}
    write_jsonl(data / 'enriched.jsonl', [*candidates, extra])
    write_jsonl(data / 'semantic_reviews.jsonl', [*reviews, review])
    (data / 'frozen_source_splits.json').write_text(json.dumps({
        'source_splits': {'s0': 'train', 's1': 'train', 's2': 'val', 's3': 'val', 's4': 'test', 's5': 'test'},
        'base_retained_ids': [str(i) for i in range(11)]}))
    monkeypatch.setattr(materialization, 'duplicate_removals', lambda rows, prior: ([], []))
    result = materialization.materialize(data, selection='enriched.jsonl')
    assert result['retained'] == 11
    assert result['excluded_by_reason']['enrichment_from_heldout_article'] == 1
    rows = materialization.read_jsonl(data / 'pool/passages.jsonl')
    assert {r['chunk_id'] for r in rows if r['split'] == 'test'} == {'8', '9', '10'}
