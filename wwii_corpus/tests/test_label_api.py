import json

import pytest

from wwii_corpus.label_api import (
    BudgetExceeded, Ledger, MODEL, RATES, cache_key, cost_from_usage,
    make_payload, validate_response,
)


def test_budget_keeps_unknown_cost_reserved_and_releases_rejected_cost(tmp_path):
    ledger = Ledger(tmp_path / 'ledger.sqlite3', budget=0.1)
    rid = ledger.reserve(MODEL, 0.08)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(MODEL, 0.03)
    ledger.settle(rid, None, 'unknown_network_outcome')
    assert ledger.charged() == pytest.approx(0.08)
    assert ledger.summary()['unsettled_requests_at_reserved_maximum'] == 1
    ledger.settle(rid, 0, 'http_400')
    assert ledger.charged() == 0


def test_cost_uses_reported_cache_tokens():
    usage = {'prompt_tokens': 1000, 'completion_tokens': 100,
             'prompt_tokens_details': {'cached_tokens': 500}}
    assert cost_from_usage(usage, RATES[MODEL]) == pytest.approx(0.00041)
    with pytest.raises(ValueError):
        cost_from_usage({**usage, 'prompt_tokens': 100}, RATES[MODEL])


def test_prompts_omit_source_identity_and_keyword_hints():
    row = {'chunk_id': 'SECRET_ID', 'chunk_text': 'Quoted passage.',
           'source_title': 'SECRET_SOURCE', 'keyword_hints': {'positive': ['SECRET_HINT']}}
    anchors = [{'code': 'A01', 'name': 'Industry', 'definition': 'Production.'}]
    payload = make_payload([row], anchors)
    content = json.dumps(payload)
    assert all(secret not in content for secret in ['SECRET_ID', 'SECRET_SOURCE', 'SECRET_HINT'])
    assert json.loads(payload['messages'][1]['content']) == [{'index': 0, 'text': 'Quoted passage.'}]
    assert payload['store'] is False


def test_response_requires_every_passage_and_known_unique_codes():
    good = {'index': 0, 'positive': ['A01'], 'evidence': [{'code': 'A01', 'quote': 'passage'}],
            'uncertain': [], 'usable': True, 'reason': 'ok'}
    assert validate_response(json.dumps({'items': [good]}), 1, ['A01']) == [good]
    for bad in [{**good, 'positive': ['A01', 'A01']}, {**good, 'positive': ['A02']},
                {**good, 'index': 1}, {**good, 'usable': False}, {**good, 'index': True}]:
        with pytest.raises(ValueError):
            validate_response(json.dumps({'items': [bad]}), 1, ['A01'])
    flagged = validate_response(json.dumps({'items': [good]}), 1, ['A01'], [{'chunk_text': 'different text'}])[0]
    assert flagged['uncertain'] == ['A01']
    assert flagged['evidence'][0]['exact_match'] is False


def test_cache_invalidates_when_definitions_model_or_text_changes():
    row = {'chunk_id': 'a', 'chunk_text': 'text'}
    key = cache_key(row, 'anchors-v1', MODEL)
    assert key != cache_key(row, 'anchors-v2', MODEL)
    assert key != cache_key({**row, 'chunk_text': 'changed'}, 'anchors-v1', MODEL)
    assert key != cache_key(row, 'anchors-v1', 'different-model')


def test_wire_schema_attaches_evidence_to_its_positive_code():
    wire = {'items': {'0': {'positive': [{'code': 'A01', 'quote': 'factory output'}],
                            'uncertain': [], 'usable': True, 'reason': 'ok'}}}
    result = validate_response(json.dumps(wire), 1, ['A01'], [{'chunk_text': 'Rising factory output.'}])[0]
    assert result['positive'] == ['A01']
    assert result['evidence'][0]['exact_match'] is True


def test_separate_ledger_connections_share_the_cost_cap(tmp_path):
    path = tmp_path / 'ledger.sqlite3'
    first, second = Ledger(path, .1), Ledger(path, .1)
    first.reserve(MODEL, .08)
    with pytest.raises(BudgetExceeded):
        second.reserve(MODEL, .03)
    assert second.charged() == pytest.approx(.08)
