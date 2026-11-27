import random

from wwii_corpus.duplicates import near_duplicates
from wwii_pilot.build_shared_pool import near_duplicates as exhaustive


def test_prefix_index_matches_exhaustive_search():
    rng = random.Random(673)
    vocabulary = [f'word{i}' for i in range(35)]
    texts = [' '.join(rng.sample(vocabulary, rng.randint(4, 25))) for _ in range(200)]
    texts += [text.replace('word1 ', 'replacement ') for text in texts[:40]]
    texts += [texts[0], 'completely unrelated passage words']
    rows = [{'chunk_id': str(i), 'chunk_text': text} for i, text in enumerate(texts)]
    for threshold in (.5, .88, 1):
        expected = [{k: v for k, v in pair.items() if k != 'action'}
                    for pair in exhaustive(rows, threshold)]
        assert near_duplicates(rows, threshold) == expected
