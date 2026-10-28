"""Fake embeddings with concepts planted in known positions.

Lets us check the pipeline finds what we put there before real embeddings
exist, and shows the difference between frozen and retrain ablation.

Concept layout (each planted group is `group` positions wide):
  concepts 0..D-2 : "local"     one group of positions shifts when the concept is present
  concept  D-1    : "redundant" two separate groups both carry it, so ablating
                    one group should NOT remove the concept when the probe is retrained
"""

from __future__ import annotations

import numpy as np


def make_synthetic(n=1500, d=64, n_concepts=4, group=3, prevalence=0.15, effect=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = (rng.random((n, n_concepts)) < prevalence).astype(np.float32)

    perm = rng.permutation(d)
    planted, cursor = {}, 0
    for c in range(n_concepts):
        n_groups = 2 if c == n_concepts - 1 else 1
        groups = []
        for _ in range(n_groups):
            groups.append(perm[cursor: cursor + group].tolist())
            cursor += group
        planted[c] = groups
        for g in groups:
            X[np.ix_(Y[:, c] == 1, g)] += effect
    concepts = [f"local_{c}" for c in range(n_concepts - 1)] + ["redundant"]
    return X, Y, concepts, planted
