"""Build a small real-text dataset to check the probe pipeline before the
team's real embeddings exist.

- Chunks: Reuters-21578 articles (one article = one chunk; median ~145 words).
- Concepts: 40 Reuters topics, the 3rd to 42nd largest. `earn` and `acq`
  are skipped because they are huge and broad.
- Sampling: rarest topic first, add articles until each topic has about
  `--per-topic` positives (or all it has). Gives about 1,000 chunks,
  matching the plan.
- Embeddings: LSA (TF-IDF then truncated SVD). This is a stand-in, not a
  neural model: it needs no download and it is a fair first test that the
  pipeline works on real text, real label overlap and correlated positions.
  Swap in the real model's embeddings by passing them to `probe.run`.

    python -m nltk.downloader reuters      # once
    python experiments/build_reuters_lsa.py --out experiments/data/reuters_lsa
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import reuters
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def pick_topics(skip=("earn", "acq"), n_topics=40):
    counts = collections.Counter(t for f in reuters.fileids() for t in reuters.categories(f))
    ranked = [t for t, _ in counts.most_common() if t not in skip]
    return ranked[:n_topics], counts


def sample_docs(topics, per_topic, seed):
    rng = np.random.default_rng(seed)
    by_topic = {t: [f for f in reuters.fileids() if t in reuters.categories(f)] for t in topics}
    chosen, have = [], collections.Counter()
    for t in sorted(topics, key=lambda t: len(by_topic[t])):         # rarest first
        pool = [f for f in by_topic[t] if f not in set(chosen)]
        rng.shuffle(pool)
        for f in pool:
            if have[t] >= per_topic:
                break
            chosen.append(f)
            have.update(c for c in reuters.categories(f) if c in topics)
    return sorted(chosen)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/data/reuters_lsa")
    ap.add_argument("--per-topic", type=int, default=40)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    topics, counts = pick_topics()
    docs = sample_docs(topics, args.per_topic, args.seed)

    # TF-IDF + SVD fit on the whole corpus (no labels used), then applied to the sample.
    all_ids = reuters.fileids()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    T = tfidf.fit_transform([reuters.raw(f) for f in all_ids])
    svd = TruncatedSVD(n_components=args.dim, random_state=args.seed).fit(T)
    row = {f: i for i, f in enumerate(all_ids)}
    X = svd.transform(T[[row[f] for f in docs]]).astype(np.float32)

    Y = np.array([[int(t in reuters.categories(f)) for t in topics] for f in docs])
    labels = pd.DataFrame(Y, columns=topics)
    labels.insert(0, "id", docs)

    np.save(out / "embeddings.npy", X)
    labels.to_csv(out / "labels.csv", index=False)
    info = {
        "n_chunks": len(docs), "dim": args.dim, "topics": topics,
        "positives_per_topic": dict(zip(topics, Y.sum(0).tolist())),
        "corpus_docs_per_topic": {t: counts[t] for t in topics},
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(f"{len(docs)} chunks, {len(topics)} topics, positives {Y.sum(0).min()}-{Y.sum(0).max()}")


if __name__ == "__main__":
    main()
