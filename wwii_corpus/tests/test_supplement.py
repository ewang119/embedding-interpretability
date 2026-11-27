import json

from nltk.tokenize.punkt import PunktSentenceTokenizer

from wwii_corpus import supplement as supplementation
from wwii_corpus.prepare import write_jsonl
from wwii_pilot.build_shared_pool import read_jsonl


def test_pilot_supplement_preserves_inputs_omits_old_labels_and_skips_heldout(tmp_path, monkeypatch):
    root = tmp_path / 'project/wwii_corpus'
    data, pilot = root / 'data', root.parent / 'wwii_pilot/data'
    data.mkdir(parents=True)
    pilot.mkdir(parents=True)
    text = ('The army lacked enough fuel to move its vehicles along the supply route. '
            'Military commanders halted their planned offensive until additional shipments could reach the front.')
    old = [{'chunk_id': pid, 'chunk_text': text, 'source_url': url, 'anchor_version': 'fixture',
            'working_labels': {'oil_fuel_constraints': 1, 'churchill_rhetoric_public_morale': 0}}
           for pid, url in [('new', 'https://example.org/new'), ('heldout', 'https://example.org/test')]]
    write_jsonl(pilot / 'passages.jsonl', old)
    before = (pilot / 'passages.jsonl').read_bytes()
    write_jsonl(data / 'selected_semantic_candidates.jsonl', [
        {'source_page_url': 'https://example.org/test', 'source_id': 'test',
         'source_sentences': ['Existing unrelated sentence.']}])
    (data / 'frozen_source_splits.json').write_text(json.dumps({'source_splits': {'test': 'test'}}))
    monkeypatch.setattr(supplementation, 'ROOT', root)
    monkeypatch.setattr(supplementation, 'load_language_tools', lambda: (PunktSentenceTokenizer(), None))
    result = supplementation.supplement()
    rows = read_jsonl(data / 'supplemental_candidates.jsonl')
    assert result['added'] == 1
    assert result['skipped'] == [{'chunk_id': 'heldout', 'reason': 'heldout_article'}]
    assert rows[0]['review_state'] == 'Unreviewed'
    assert 'working_labels' not in rows[0] and 'labels' not in rows[0]
    assert (pilot / 'passages.jsonl').read_bytes() == before
