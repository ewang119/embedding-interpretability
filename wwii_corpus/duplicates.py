"""Exact prefix filtering for the pilot's existing near-duplicate definition."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import math

from wwii_pilot.build_shared_pool import fingerprint


def near_duplicates(rows, threshold=.88, jaccard=.65):
    """Return the same pairs as exhaustive Jaccard + character matching.

    Globally ordered set prefixes must intersect for a pair to exceed the
    Jaccard threshold. The prefix index skips impossible pairs without making
    approximate similarity decisions.
    """
    if not 0 < jaccard <= 1 or not 0 <= threshold <= 1:
        raise ValueError('invalid similarity threshold')
    texts = [fingerprint(row['chunk_text']) for row in rows]
    words = [set(text.split()) for text in texts]
    frequency = Counter(word for values in words for word in values)
    index = defaultdict(list)
    results = []
    for i, values in enumerate(words):
        ordered = sorted(values, key=lambda word: (frequency[word], word))
        prefix = ordered[:len(values) - math.ceil(jaccard * len(values)) + 1]
        possible = {j for word in prefix for j in index[word]}
        for j in sorted(possible):
            if not jaccard * len(values) <= len(words[j]) <= len(values) / jaccard:
                continue
            union = len(values | words[j])
            if not union or len(values & words[j]) / union < jaccard:
                continue
            ratio = SequenceMatcher(None, texts[i], texts[j], autojunk=False).ratio()
            if ratio >= threshold:
                results.append({'chunk_id_a': rows[j]['chunk_id'], 'chunk_id_b': rows[i]['chunk_id'],
                                'similarity': round(ratio, 4)})
        for word in prefix:
            index[word].append(i)
    return results
