import base64
import json

import numpy as np
import pytest

from wwii_corpus import semantic_retrieval as retrieval


def test_embedding_decoding_uses_returned_indices_for_row_alignment(monkeypatch):
    monkeypatch.setattr(retrieval, 'DIMENSIONS', 3)
    one, two = np.array([1, 2, 3], dtype='<f4'), np.array([4, 5, 6], dtype='<f4')
    encode = lambda x: base64.b64encode(x.tobytes()).decode()
    doc = {'data': [{'index': 1, 'embedding': encode(two)}, {'index': 0, 'embedding': encode(one)}]}
    result = retrieval.decode_vectors(doc, 2)
    np.testing.assert_array_equal(result, np.stack([one, two]))
    doc['data'][1]['index'] = 1
    with pytest.raises(ValueError, match='indices'):
        retrieval.decode_vectors(doc, 2)


def test_embedding_decoder_rejects_wrong_dimensions_and_nonfinite_values(monkeypatch):
    monkeypatch.setattr(retrieval, 'DIMENSIONS', 3)
    for value in [[1, 2], [float('nan'), 1, 2], [0, 0, 0]]:
        with pytest.raises(ValueError, match='shape or values'):
            retrieval.decode_vectors({'data': [{'index': 0, 'embedding': value}]}, 1)


def test_training_export_aligns_base_and_extra_vectors_by_passage_id(tmp_path, monkeypatch):
    root = tmp_path / 'corpus'
    cache, pool = root / 'work/embeddings', root / 'data/pool'
    cache.mkdir(parents=True)
    (pool / 'training').mkdir(parents=True)
    monkeypatch.setattr(retrieval, 'ROOT', root)
    monkeypatch.setattr(retrieval, 'DIMENSIONS', 3)
    np.save(cache / 'vectors.npy', np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    np.save(cache / 'extra_vectors.npy', np.array([[7, 8, 9]], dtype=np.float32))
    for name, ids in [('manifest.json', ['a', 'b']), ('extra_manifest.json', ['c'])]:
        (cache / name).write_text(json.dumps({'model': retrieval.MODEL, 'dimensions': 3, 'ids': ids}))
    (pool / 'passages.jsonl').write_text(''.join(json.dumps({'chunk_id': pid}) + '\n' for pid in ['b', 'c', 'a']))
    (pool / 'summary.json').write_text(json.dumps({'training_file_sha256': {}}))
    result = retrieval.export_embeddings(pool)
    np.testing.assert_array_equal(np.load(pool / 'training/embeddings.npy'), [[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    assert result['rows'] == 3
    assert json.loads((pool / 'training/embedding_manifest.json').read_text())['row_ids'] == ['b', 'c', 'a']
